//! Particle-filter (MCL) relocalize against a saved global grid.
//!
//! Consumes one ToF scan per call, narrows a cloud of (x, y, yaw)
//! hypotheses over a few seconds. The runtime wires it under
//! `pending_relocalize`: while the cloud is spread, no submap ingestion;
//! once it collapses (and stays collapsed for a few frames), the runtime
//! snaps `tracked` to `best()` and resumes regular SLAM.
//!
//! Sensor model: per-beam residual against the grid's distance field,
//! Gaussian likelihood with `beam_sigma_m`, residuals clamped at
//! `beam_clamp_m` so a single bad beam can't flatten the weight.
//!
//! Motion model: odometry-driven body-frame translation + yaw delta with
//! Gaussian noise scaled by travelled distance + |yaw delta|. Same
//! convention as v1's MCL.
//!
//! Resampling: low-variance (systematic) sampler triggered when the
//! effective sample size falls below `resample_ess_frac * N`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::grid::OccupancyGrid;
use crate::submap::Pose2;

#[derive(Debug, Clone)]
pub struct MclConfig {
    pub n_particles: usize,
    /// Motion-model noise: σ_xy = `sigma_xy_per_m * |Δxy|
    /// + sigma_xy_per_rad * |Δyaw|`. Same for yaw.
    pub sigma_xy_per_m: f32,
    pub sigma_xy_per_rad: f32,
    pub sigma_yaw_per_m: f32,
    pub sigma_yaw_per_rad: f32,
    /// Gaussian std on per-beam distance-field residual.
    pub beam_sigma_m: f32,
    /// Clamp per-beam residual at this (saturation guard).
    pub beam_clamp_m: f32,
    /// Skip the update when fewer than this many beams are valid.
    pub min_beams_used: u32,
    /// Resample when effective sample size < `frac * N`.
    pub resample_ess_frac: f32,
    /// Lock criteria — cloud spread + best-particle residual.
    pub locked_xy_std_m: f32,
    pub locked_yaw_std_rad: f32,
    pub locked_max_residual_m: f32,
    /// Require N consecutive frames meeting the lock criteria.
    pub locked_min_frames: u32,
    /// Tiny "exploration" noise injected on every predict — helps the
    /// cloud not collapse onto a single point too fast on quiet odom.
    pub jitter_xy_m: f32,
    pub jitter_yaw_rad: f32,
    /// Fraction of particles replaced with fresh uniform samples after
    /// each resample. Probes for missed posterior peaks; without this,
    /// a wrong-but-plausible cluster can capture the cloud and never
    /// release it.
    pub random_inject_frac: f32,
    /// Fixed-point log-odds threshold above which a cell counts as a
    /// wall for the distance-field likelihood. `OCC_THRESHOLD` (= 0)
    /// reproduces the previous behaviour; bumping it up filters
    /// transient noise — useful when the saved map is fuzzy. 200 ≈ "a
    /// cell needed 3+ net hits to be a wall".
    pub wall_threshold_fp: i16,
    /// Tempering factor for the observation likelihood. log-likelihoods
    /// are multiplied by this before normalising. 1.0 = raw (very peaky
    /// with ~64 beams; the cloud collapses to a single cluster after
    /// one frame, which is usually wrong with a 45° FOV); 0.0 = ignore
    /// the scan entirely. Lower values keep competing modes alive long
    /// enough that subsequent motion can disambiguate. 0.3 is a
    /// reasonable default for ~64-beam VL53L5CX scans.
    pub likelihood_temper: f32,
    /// Minimum translation (metres) that must accumulate via `predict`
    /// between successive streak increments. A stationary duck whose
    /// cloud happens to be narrow won't lock — motion is required to
    /// validate the hypothesis.
    pub lock_min_motion_per_streak_m: f32,
    /// Alternative path to a streak increment: a yaw rotation of at
    /// least this much (radians) since the last increment.
    pub lock_min_rotation_per_streak_rad: f32,
    /// Skip resampling for the first N updates. Lets weight evidence
    /// accumulate multiplicatively across frames so the cloud doesn't
    /// collapse on frame 1 to whichever wrong cluster fit best. After
    /// the grace period, normal ESS-based resampling resumes.
    pub min_updates_before_resample: u32,
}

impl Default for MclConfig {
    fn default() -> Self {
        Self {
            n_particles: 800,
            sigma_xy_per_m: 0.10,
            sigma_xy_per_rad: 0.05,
            sigma_yaw_per_m: 0.05,
            sigma_yaw_per_rad: 0.10,
            beam_sigma_m: 0.20,
            beam_clamp_m: 0.50,
            random_inject_frac: 0.05,
            min_beams_used: 16,
            resample_ess_frac: 0.5,
            locked_xy_std_m: 0.15,
            locked_yaw_std_rad: 8.0_f32.to_radians(),
            locked_max_residual_m: 0.12,
            // 25 frames at 15 Hz ≈ 1.7 s of consistent narrow cloud.
            // Faster than this and a one-frame fluke can declare lock.
            locked_min_frames: 25,
            jitter_xy_m: 0.005,
            jitter_yaw_rad: 0.005,
            wall_threshold_fp: 200,
            // Soften the posterior. With 64 beams and σ=0.20 m the
            // raw posterior spans ~200 nats; temper=0.15 compresses
            // that to ~30, so multiple competing modes survive instead
            // of collapsing onto whichever cluster fit slightly best.
            likelihood_temper: 0.15,
            // Motion gating: see `relocalize_start_odom` plumbing in
            // the runtime. These values are still consulted as a
            // *lower bound* inside MCL but the runtime applies a
            // stricter net-displacement check on top.
            lock_min_motion_per_streak_m: 0.05,
            lock_min_rotation_per_streak_rad: 0.05,
            // Don't resample on frames 1..5 — let weights multiply
            // across updates so the cloud commits only after multiple
            // frames of consistent evidence.
            min_updates_before_resample: 5,
        }
    }
}

pub struct Localizer {
    cfg: MclConfig,
    particles: Vec<Pose2>,
    weights:   Vec<f32>,
    rng: StdRng,
    /// Last computed best-particle residual (NaN until the first update).
    last_residual_m: f32,
    /// Frames in a row that satisfied the lock criteria.
    locked_streak: u32,
    /// Translation (metres) received via `predict` since the lock-streak
    /// counter last advanced. Used to gate streak increments on real
    /// motion, so a stationary duck can never accidentally lock.
    motion_since_streak_m: f32,
    /// Same idea for rotation.
    rotation_since_streak_rad: f32,
    /// Number of `update` calls so far. Used by the
    /// `min_updates_before_resample` grace period.
    update_count: u32,
}

impl Localizer {
    pub fn new(cfg: MclConfig, seed: u64) -> Self {
        let n = cfg.n_particles.max(1);
        Self {
            cfg,
            particles: vec![(0.0, 0.0, 0.0); n],
            weights:   vec![1.0; n],
            rng: StdRng::seed_from_u64(seed),
            last_residual_m: f32::NAN,
            locked_streak: 0,
            motion_since_streak_m: 0.0,
            rotation_since_streak_rad: 0.0,
            update_count: 0,
        }
    }

    pub fn n_particles(&self) -> usize { self.particles.len() }
    pub fn last_residual_m(&self) -> f32 { self.last_residual_m }
    pub fn locked_streak(&self) -> u32 { self.locked_streak }

    /// Spread particles uniformly over `grid`'s free cells with random
    /// yaw. Falls back to (0, 0, *) if the grid has no free cells (e.g.
    /// blank). Sets all weights equal.
    pub fn seed_uniform(&mut self, grid: &OccupancyGrid) {
        let cfg = grid.cfg();
        let cell = cfg.cell;
        let w = grid.width(); let h = grid.height();
        let mut free: Vec<(f32, f32)> = Vec::with_capacity(w * h / 4);
        for i in 0..h {
            for j in 0..w {
                if grid.is_known_free(i, j) {
                    let cx = cfg.x_range.0 + (j as f32 + 0.5) * cell;
                    let cy = cfg.y_range.0 + (i as f32 + 0.5) * cell;
                    free.push((cx, cy));
                }
            }
        }
        let two_pi = 2.0 * std::f32::consts::PI;
        for p in self.particles.iter_mut() {
            if free.is_empty() {
                let yaw = self.rng.gen::<f32>() * two_pi - std::f32::consts::PI;
                *p = (0.0, 0.0, yaw);
            } else {
                let (cx, cy) = free[self.rng.gen_range(0..free.len())];
                let jx = (self.rng.gen::<f32>() - 0.5) * cell;
                let jy = (self.rng.gen::<f32>() - 0.5) * cell;
                let yaw = self.rng.gen::<f32>() * two_pi - std::f32::consts::PI;
                *p = (cx + jx, cy + jy, yaw);
            }
        }
        self.reset_weights();
        self.locked_streak = 0;
        self.last_residual_m = f32::NAN;
        self.motion_since_streak_m = 0.0;
        self.rotation_since_streak_rad = 0.0;
        self.update_count = 0;
    }

    /// Seed particles from a small list of candidate poses (e.g. the
    /// top-K from brute-force search). Particles are split evenly across
    /// seeds with Gaussian noise (σ_xy=`spread_xy_m`, σ_yaw=`spread_yaw_rad`).
    pub fn seed_around(&mut self, seeds: &[Pose2], spread_xy_m: f32, spread_yaw_rad: f32) {
        if seeds.is_empty() { return; }
        let nx = Normal::new(0.0_f32, spread_xy_m).unwrap();
        let ny = Normal::new(0.0_f32, spread_xy_m).unwrap();
        let nt = Normal::new(0.0_f32, spread_yaw_rad).unwrap();
        for (i, p) in self.particles.iter_mut().enumerate() {
            let s = seeds[i % seeds.len()];
            *p = (
                s.0 + nx.sample(&mut self.rng),
                s.1 + ny.sample(&mut self.rng),
                wrap_pi(s.2 + nt.sample(&mut self.rng)),
            );
        }
        self.reset_weights();
        self.locked_streak = 0;
        self.last_residual_m = f32::NAN;
        self.motion_since_streak_m = 0.0;
        self.rotation_since_streak_rad = 0.0;
        self.update_count = 0;
    }

    /// Mixed seed: a fraction of particles around `seeds` (Gaussian
    /// noise) and the rest uniformly over the grid's free cells. Use
    /// this to combine a brute-force candidate with exploration — if
    /// the brute-force pose is wrong, the uniform cloud still has a
    /// chance to win after a few frames of motion.
    pub fn seed_mixed(
        &mut self,
        seeds: &[Pose2],
        seeds_frac: f32,
        grid: &OccupancyGrid,
        spread_xy_m: f32,
        spread_yaw_rad: f32,
    ) {
        // Start uniform, then overwrite the front `frac * N` particles
        // with seeded ones.
        self.seed_uniform(grid);
        if seeds.is_empty() { return; }
        let frac = seeds_frac.clamp(0.0, 1.0);
        let n_seed = ((self.particles.len() as f32) * frac) as usize;
        if n_seed == 0 { return; }
        let nx = Normal::new(0.0_f32, spread_xy_m).unwrap();
        let ny = Normal::new(0.0_f32, spread_xy_m).unwrap();
        let nt = Normal::new(0.0_f32, spread_yaw_rad).unwrap();
        for i in 0..n_seed {
            let s = seeds[i % seeds.len()];
            self.particles[i] = (
                s.0 + nx.sample(&mut self.rng),
                s.1 + ny.sample(&mut self.rng),
                wrap_pi(s.2 + nt.sample(&mut self.rng)),
            );
        }
        // reset_weights / streak / residual already zeroed by seed_uniform.
    }

    /// Apply a body-frame motion delta with noise.
    pub fn predict(&mut self, dx_b: f32, dy_b: f32, dyaw: f32) {
        let trans = (dx_b * dx_b + dy_b * dy_b).sqrt();
        let rot   = dyaw.abs();
        // Track real motion so lock gating can require it.
        self.motion_since_streak_m += trans;
        self.rotation_since_streak_rad += rot;
        let sigma_xy = self.cfg.sigma_xy_per_m * trans
                     + self.cfg.sigma_xy_per_rad * rot
                     + self.cfg.jitter_xy_m;
        let sigma_yaw = self.cfg.sigma_yaw_per_m * trans
                      + self.cfg.sigma_yaw_per_rad * rot
                      + self.cfg.jitter_yaw_rad;
        let nx = Normal::new(0.0_f32, sigma_xy.max(1e-6)).unwrap();
        let ny = Normal::new(0.0_f32, sigma_xy.max(1e-6)).unwrap();
        let nt = Normal::new(0.0_f32, sigma_yaw.max(1e-6)).unwrap();
        for p in self.particles.iter_mut() {
            // Compose body-frame delta in particle frame.
            let cy = p.2.cos(); let sy = p.2.sin();
            let dx_w = cy * dx_b - sy * dy_b + nx.sample(&mut self.rng);
            let dy_w = sy * dx_b + cy * dy_b + ny.sample(&mut self.rng);
            p.0 += dx_w;
            p.1 += dy_w;
            p.2 = wrap_pi(p.2 + dyaw + nt.sample(&mut self.rng));
        }
    }

    /// Reweight particles against a single scan and resample if needed.
    pub fn update(
        &mut self,
        grid: &mut OccupancyGrid,
        angles_body: &[f32],
        ranges_horiz: &[f32],
    ) {
        if angles_body.is_empty() || angles_body.len() != ranges_horiz.len() { return; }
        let n_valid = ranges_horiz.iter().filter(|r| r.is_finite() && **r > 0.0).count();
        if (n_valid as u32) < self.cfg.min_beams_used { return; }

        let field = grid.distance_field(self.cfg.wall_threshold_fp).to_vec();
        let cfg_g = *grid.cfg();
        let w = grid.width(); let h = grid.height();
        let cell = cfg_g.cell;
        let x_min = cfg_g.x_range.0;
        let y_min = cfg_g.y_range.0;
        let two_sigma2 = 2.0 * self.cfg.beam_sigma_m * self.cfg.beam_sigma_m;
        let clamp = self.cfg.beam_clamp_m;

        let mut log_w: Vec<f32> = Vec::with_capacity(self.particles.len());
        for p in &self.particles {
            let mut sum = 0.0_f32;
            let mut n = 0u32;
            for (a, &r) in angles_body.iter().zip(ranges_horiz) {
                if !r.is_finite() || r <= 0.0 { continue; }
                let theta = p.2 + a;
                let hx = p.0 + r * theta.cos();
                let hy = p.1 + r * theta.sin();
                let j = ((hx - x_min) / cell) as i32;
                let i = ((hy - y_min) / cell) as i32;
                if i < 0 || j < 0 || (i as usize) >= h || (j as usize) >= w { continue; }
                let d = field[(i as usize) * w + (j as usize)].min(clamp);
                sum += -(d * d) / two_sigma2;
                n += 1;
            }
            // Sum (not mean) so weight contrast is high enough that
            // ESS-based resampling fires when the cloud needs to
            // collapse. Softness comes from `beam_sigma_m`.
            if n == 0 { log_w.push(-1e6); }
            else      { log_w.push(sum); }
        }
        // Temper the log-likelihoods before normalizing (raise the
        // posterior to a power < 1). Keeps competing modes alive past
        // the first update — with 64 beams and σ=0.20 m the raw
        // posterior is peaky enough that one frame collapses the cloud
        // to whichever cluster happened to fit best, regardless of
        // whether it's right.
        let temper = self.cfg.likelihood_temper.max(1e-6);
        // Stabilize: subtract the max before exponentiating.
        let max_lw = log_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // Multiplicative weight update — keep prior evidence around so
        // the cloud commits only after multiple frames agree. After
        // resampling fires, weights are reset uniform anyway, so this
        // collapses to the usual single-frame likelihood update from
        // that point on.
        let mut sum_w = 0.0_f32;
        for (w_out, &lw) in self.weights.iter_mut().zip(log_w.iter()) {
            let factor = ((lw - max_lw) * temper).exp();
            *w_out *= factor;
            sum_w += *w_out;
        }
        if sum_w > 0.0 {
            for w_out in self.weights.iter_mut() { *w_out /= sum_w; }
        } else {
            self.reset_weights();
        }
        self.update_count += 1;

        // Best particle residual for the lock check.
        if let Some((best_idx, _)) = self.weights.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let p = self.particles[best_idx];
            let mut sum = 0.0_f32;
            let mut n = 0u32;
            for (a, &r) in angles_body.iter().zip(ranges_horiz) {
                if !r.is_finite() || r <= 0.0 { continue; }
                let theta = p.2 + a;
                let hx = p.0 + r * theta.cos();
                let hy = p.1 + r * theta.sin();
                let j = ((hx - x_min) / cell) as i32;
                let i = ((hy - y_min) / cell) as i32;
                if i < 0 || j < 0 || (i as usize) >= h || (j as usize) >= w { continue; }
                let d = field[(i as usize) * w + (j as usize)].min(clamp);
                sum += d;
                n += 1;
            }
            self.last_residual_m = if n > 0 { sum / (n as f32) } else { f32::NAN };
        }

        // Resample if ESS too low — but only after the grace period.
        // During grace, weights keep accumulating evidence across
        // frames; resampling early throws that away.
        let ess: f32 = 1.0 / self.weights.iter().map(|w| w * w).sum::<f32>().max(1e-12);
        if self.update_count >= self.cfg.min_updates_before_resample
            && ess < self.cfg.resample_ess_frac * (self.particles.len() as f32) {
            self.systematic_resample();
            // Replace `random_inject_frac` of the (now duplicated)
            // resampled particles with fresh uniform samples drawn from
            // the grid's free cells. Keeps exploration alive without
            // melting the rest of the posterior.
            let n_inject = ((self.particles.len() as f32)
                            * self.cfg.random_inject_frac.clamp(0.0, 1.0))
                           as usize;
            if n_inject > 0 {
                self.inject_uniform(grid, n_inject);
            }
        }

        // Update lock streak based on cloud spread + best-particle
        // residual AND demonstrated motion since the last streak step.
        // A stationary duck whose cloud collapsed on a wrong cluster
        // would otherwise streak forever and lock — motion is the only
        // signal that can disambiguate a 45° FOV.
        let xy_std = self.position_std();
        let yaw_std = self.yaw_std();
        let res_ok = self.last_residual_m.is_finite()
            && self.last_residual_m <= self.cfg.locked_max_residual_m;
        let spread_ok = xy_std <= self.cfg.locked_xy_std_m
                     && yaw_std <= self.cfg.locked_yaw_std_rad;
        let motion_ok = self.motion_since_streak_m
                          >= self.cfg.lock_min_motion_per_streak_m
                     || self.rotation_since_streak_rad
                          >= self.cfg.lock_min_rotation_per_streak_rad;
        if res_ok && spread_ok && motion_ok {
            self.locked_streak += 1;
            self.motion_since_streak_m = 0.0;
            self.rotation_since_streak_rad = 0.0;
        } else if !res_ok || !spread_ok {
            // Bad fit / wide cloud → reset streak. Keep motion
            // accumulating so a stationary duck that suddenly walks
            // doesn't have to wait for a fresh motion budget.
            self.locked_streak = 0;
        }
        // res_ok && spread_ok but no motion yet → don't increment, but
        // don't reset either. Cloud stays tentatively narrow until the
        // duck moves enough to validate.
    }

    /// Highest-weight particle.
    pub fn best(&self) -> Pose2 {
        self.weights.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| self.particles[i])
            .unwrap_or((0.0, 0.0, 0.0))
    }

    /// Weighted-mean particle (alternative to `best()` when the cloud
    /// is multi-modal — collapses to the centre of mass).
    pub fn weighted_mean(&self) -> Pose2 {
        let mut x = 0.0_f32; let mut y = 0.0_f32;
        let mut sx = 0.0_f32; let mut cy = 0.0_f32;
        let mut sw = 0.0_f32;
        for (p, &w) in self.particles.iter().zip(self.weights.iter()) {
            x += w * p.0;
            y += w * p.1;
            sx += w * p.2.sin();
            cy += w * p.2.cos();
            sw += w;
        }
        if sw <= 0.0 { return (0.0, 0.0, 0.0); }
        (x / sw, y / sw, sx.atan2(cy))
    }

    /// 1D std-dev across the cloud's xy positions, equally weighted.
    pub fn position_std(&self) -> f32 {
        let n = self.particles.len() as f32;
        if n < 2.0 { return 0.0; }
        let (mut mx, mut my) = (0.0_f32, 0.0_f32);
        for p in &self.particles { mx += p.0; my += p.1; }
        mx /= n; my /= n;
        let mut var = 0.0_f32;
        for p in &self.particles {
            let dx = p.0 - mx; let dy = p.1 - my;
            var += dx * dx + dy * dy;
        }
        (var / n).sqrt()
    }

    pub fn yaw_std(&self) -> f32 {
        let n = self.particles.len() as f32;
        if n < 2.0 { return 0.0; }
        let mut sx = 0.0_f32; let mut cy = 0.0_f32;
        for p in &self.particles { sx += p.2.sin(); cy += p.2.cos(); }
        // Circular variance → effective std (Mardia).
        let r = ((sx * sx + cy * cy).sqrt()) / n;
        if r >= 1.0 { 0.0 } else { (-2.0 * r.ln()).sqrt() }
    }

    /// True iff the lock criteria have held for `locked_min_frames`
    /// consecutive frames.
    pub fn is_locked(&self) -> bool {
        self.locked_streak >= self.cfg.locked_min_frames
    }

    fn reset_weights(&mut self) {
        let n = self.particles.len() as f32;
        for w in self.weights.iter_mut() { *w = 1.0 / n; }
    }

    /// Replace the front `count` particles with fresh uniform samples
    /// over the grid's free cells. Weights left untouched (they get
    /// reset on the next resample anyway).
    fn inject_uniform(&mut self, grid: &OccupancyGrid, count: usize) {
        let cfg = grid.cfg();
        let cell = cfg.cell;
        let w = grid.width(); let h = grid.height();
        // Sample free cells one at a time; if we hit too many occupied
        // cells in a row just give up and leave the particle alone.
        let mut placed = 0usize;
        let mut tries = 0usize;
        let two_pi = 2.0 * std::f32::consts::PI;
        while placed < count && tries < count * 20 {
            tries += 1;
            let i = self.rng.gen_range(0..h);
            let j = self.rng.gen_range(0..w);
            if !grid.is_known_free(i, j) { continue; }
            let cx = cfg.x_range.0 + (j as f32 + 0.5) * cell;
            let cy = cfg.y_range.0 + (i as f32 + 0.5) * cell;
            let yaw = self.rng.gen::<f32>() * two_pi - std::f32::consts::PI;
            self.particles[placed] = (cx, cy, yaw);
            placed += 1;
        }
    }

    fn systematic_resample(&mut self) {
        let n = self.particles.len();
        if n == 0 { return; }
        let mut cum: Vec<f32> = Vec::with_capacity(n);
        let mut acc = 0.0_f32;
        for &w in &self.weights { acc += w; cum.push(acc); }
        if acc <= 0.0 {
            self.reset_weights();
            return;
        }
        let step = acc / n as f32;
        let u0: f32 = self.rng.gen::<f32>() * step;
        let mut new_particles: Vec<Pose2> = Vec::with_capacity(n);
        let mut k = 0;
        for i in 0..n {
            let u = u0 + i as f32 * step;
            while k < n - 1 && cum[k] < u { k += 1; }
            new_particles.push(self.particles[k]);
        }
        self.particles = new_particles;
        self.reset_weights();
    }
}

#[inline]
fn wrap_pi(a: f32) -> f32 {
    use std::f32::consts::PI;
    let two_pi = 2.0 * PI;
    let mut y = (a + PI).rem_euclid(two_pi) - PI;
    if y == PI { y = -PI; }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{GridConfig, OccupancyGrid};

    fn make_room() -> OccupancyGrid {
        // 4×4 m room with asymmetric divider. Cast rays from many
        // origins so free cells densely fill the inside, mimicking what
        // a real-robot map looks like after a walk-around.
        let mut g = OccupancyGrid::new(GridConfig {
            x_range: (-2.5, 2.5), y_range: (-2.5, 2.5), cell: 0.05,
        });
        let perim = 200;
        let mut walls: Vec<(f32, f32)> = Vec::new();
        for i in 0..perim {
            let t = -2.0 + 4.0 * (i as f32 / (perim - 1) as f32);
            walls.push(( t,  2.0));
            walls.push(( t, -2.0));
            walls.push(( 2.0, t));
            walls.push((-2.0, t));
        }
        for i in 0..perim / 2 {
            let t = 2.0 * (i as f32 / ((perim / 2) as f32));
            walls.push((t, 0.5));  // asymmetric divider
        }
        let origins: Vec<(f32, f32)> = {
            let mut v = Vec::new();
            let mut y = -1.5_f32;
            while y <= 1.5 {
                let mut x = -1.5_f32;
                while x <= 1.5 {
                    v.push((x, y));
                    x += 0.30;
                }
                y += 0.30;
            }
            v
        };
        for (ox, oy) in &origins {
            for (wx, wy) in &walls {
                g.integrate_ray(*ox, *oy, *wx, *wy, true);
            }
        }
        g
    }

    fn fake_scan(grid: &mut OccupancyGrid, pose: Pose2, n_beams: usize)
        -> (Vec<f32>, Vec<f32>) {
        let mut a = Vec::new(); let mut r = Vec::new();
        let half = std::f32::consts::FRAC_PI_2;  // ±90° fan
        for k in 0..n_beams {
            let aa = -half + (k as f32 / (n_beams - 1) as f32) * 2.0 * half;
            let rr = grid.cast_ray(pose.0, pose.1, pose.2 + aa, 4.0);
            if rr > 0.0 {
                a.push(aa);
                r.push(rr);
            }
        }
        (a, r)
    }

    #[test]
    fn seed_uniform_lands_inside_free_space() {
        let grid = make_room();
        let mut mcl = Localizer::new(MclConfig {
            n_particles: 200,
            ..MclConfig::default()
        }, 0);
        mcl.seed_uniform(&grid);
        // All particles should be inside the grid bounds; most should
        // be in cells we marked as known-free (we accept ~95% — the cell
        // jitter inside `seed_uniform` can land just over the edge of
        // the closest occupied cell).
        let mut inside_free = 0usize;
        for p in &mcl.particles {
            if let Some((i, j)) = grid.world_to_idx(p.0, p.1) {
                if grid.is_known_free(i, j) { inside_free += 1; }
            }
        }
        assert!(inside_free as f32 / mcl.n_particles() as f32 > 0.95,
                "{} of {} particles fell outside known-free",
                mcl.n_particles() - inside_free, mcl.n_particles());
    }

    #[test]
    fn predict_advances_particles_with_noise() {
        let mut mcl = Localizer::new(MclConfig {
            n_particles: 200,
            jitter_xy_m: 0.0,
            jitter_yaw_rad: 0.0,
            ..MclConfig::default()
        }, 0);
        // Seed all particles at origin facing +x.
        for p in mcl.particles.iter_mut() { *p = (0.0, 0.0, 0.0); }
        mcl.predict(1.0, 0.0, 0.0);
        // Mean should be close to +1 m on x.
        let mut mx = 0.0; for p in &mcl.particles { mx += p.0; }
        mx /= mcl.n_particles() as f32;
        assert!((mx - 1.0).abs() < 0.10, "mean x after predict = {mx}");
        // Spread should be > 0 (noise injected).
        assert!(mcl.position_std() > 0.0);
    }

    #[test]
    fn update_concentrates_weight_around_truth_seed() {
        let mut grid = make_room();
        let truth = (0.5_f32, -0.5_f32, 0.0_f32);
        // Seed half the particles around truth, half on a decoy 1.5 m
        // away. After one update the truth half should dominate.
        let mut mcl = Localizer::new(MclConfig {
            n_particles: 400,
            ..MclConfig::default()
        }, 0);
        let half = mcl.n_particles() / 2;
        let nx = Normal::new(0.0_f32, 0.05).unwrap();
        let nt = Normal::new(0.0_f32, 0.02).unwrap();
        for i in 0..mcl.n_particles() {
            let s = if i < half { truth }
                    else        { (-1.0, 0.5, std::f32::consts::PI) };
            mcl.particles[i] = (
                s.0 + nx.sample(&mut mcl.rng),
                s.1 + nx.sample(&mut mcl.rng),
                wrap_pi(s.2 + nt.sample(&mut mcl.rng)),
            );
        }
        mcl.reset_weights();
        let (a, r) = fake_scan(&mut grid, truth, 64);
        // Run several updates so the cloud commits past the
        // `min_updates_before_resample` grace period.
        for _ in 0..8 {
            mcl.update(&mut grid, &a, &r);
        }

        // After update + (internal) resample, weights are uniform — the
        // signal of "truth dominated" lives in particle *positions*, not
        // weights. Count particles near the truth pose vs the decoy.
        let near = |p: (f32, f32, f32), c: (f32, f32, f32), r: f32| {
            let dx = p.0 - c.0; let dy = p.1 - c.1;
            (dx * dx + dy * dy).sqrt() < r
        };
        let decoy = (-1.0_f32, 0.5_f32, std::f32::consts::PI);
        let n_truth = mcl.particles.iter().filter(|p| near(**p, truth, 0.30)).count();
        let n_decoy = mcl.particles.iter().filter(|p| near(**p, decoy, 0.30)).count();
        assert!(n_truth > 3 * n_decoy.max(1),
                "truth cluster didn't dominate: near_truth={n_truth} \
                 near_decoy={n_decoy}");
    }
}
