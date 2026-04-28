//! Monte Carlo Localization on the 2D occupancy grid.
//!
//! Port of `microduck_maploc/sim/localizer.py`. The Python version is the
//! design reference; numerical knobs match unless noted.
//!
//! Particle state is `(x, y, yaw)` stored in a flat `Vec<f32>` of length
//! `3 * n_particles` for cache-friendly iteration. Per-beam ray casts
//! across the whole particle cloud are parallelised with rayon — that's
//! the hot loop; everything else is per-particle scalar work that runs
//! single-threaded.
//!
//! API mirrors the Python:
//!   * `predict(dx_b, dy_b, dyaw)` — odometry motion model with noise.
//!   * `update(angles, ranges)`    — measurement update + augmented-MCL.
//!   * `best()`, `position_std()`, `last_residual_m()` — readouts.

use std::f32::consts::PI;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand::distributions::Distribution;
use rand_distr::Normal;
use rayon::prelude::*;

use crate::accumulator::BufferedBeam;
use crate::grid::OccupancyGrid;

#[inline]
fn wrap_pi(a: f32) -> f32 {
    let mut x = a;
    while x >  PI { x -= 2.0 * PI; }
    while x < -PI { x += 2.0 * PI; }
    x
}

#[derive(Debug, Clone, Copy)]
pub struct MclConfig {
    pub n_particles: usize,
    // Motion noise — sigma = a*|trans| + b*|rot|. Tuned for "trust odom":
    // small per-step noise so a converged cloud stays tight while it
    // drifts along odometry; the measurement update only nudges.
    pub sigma_xy_per_m:    f32,
    pub sigma_xy_per_rad:  f32,
    pub sigma_yaw_per_rad: f32,
    pub sigma_yaw_per_m:   f32,
    // Measurement model.
    pub beam_sigma:    f32,
    pub z_max:         f32,
    pub n_beams_used:  usize,
    /// Beams predicted at max_range contribute a fixed mild penalty
    /// instead of a huge quadratic cost (`-beam_logw_floor`).
    pub beam_logw_floor: f32,
    /// Don't score beams whose actual or predicted range is essentially
    /// at max_range — they carry no localization signal.
    pub skip_max_margin: f32,
    pub neff_threshold_frac: f32,
    // Textbook augmented-MCL (Thrun et al., Probabilistic Robotics §8.3.7).
    // Track slow / fast EWMAs of the average particle weight; whenever
    // `w_fast` drops well below `w_slow` it means the recent fit just got
    // noticeably worse than the long-term baseline (= we got lost). The
    // injection probability per particle is max(0, 1 - w_fast/w_slow),
    // which auto-decays once `w_slow` catches up — no manual thresholds.
    pub alpha_slow: f32,
    pub alpha_fast: f32,
    /// Cap injection per resample step. Never replace 100% — we'd lose
    /// any tracking info. ~50% leaves room for a correct cluster to win
    /// back the population once it's sampled.
    pub max_inject_frac: f32,
    /// Reserved (kept for parity with Python's call sites).
    pub inject_std_m: f32,
}

impl Default for MclConfig {
    fn default() -> Self {
        Self {
            n_particles: 2000,
            sigma_xy_per_m:    0.03,
            sigma_xy_per_rad:  0.02,
            sigma_yaw_per_rad: 0.04,
            sigma_yaw_per_m:   0.01,
            beam_sigma:    0.25,
            z_max:         4.0,
            n_beams_used:  16,
            beam_logw_floor: -2.5,
            skip_max_margin: 0.10,
            neff_threshold_frac: 0.5,
            alpha_slow: 0.02,
            alpha_fast: 0.30,
            max_inject_frac: 0.50,
            inject_std_m:    0.30,
        }
    }
}

/// Parameters for `Localizer::global_relocalize_field`. Defaults match
/// the Python sim's `Localizer.global_relocalize_field` parameters.
#[derive(Debug, Clone, Copy)]
pub struct FieldRelocConfig {
    pub cell_subsample: u32,
    pub n_yaw_bins:     u32,
    pub beam_subsample: u32,
    pub sigma_m:        f32,
    pub top_k_frac:     f32,
    pub jitter_xy:      f32,
    pub jitter_yaw:     f32,
    /// Distance-field threshold in fixed-point. 150 = log_odds 1.5
    /// (≈2 confirmed hits) — matches the Python planner / relocalize.
    pub occ_threshold_fp: i16,
}

impl Default for FieldRelocConfig {
    fn default() -> Self {
        Self {
            cell_subsample: 3,
            n_yaw_bins:     24,
            beam_subsample: 32,
            sigma_m:        0.20,
            top_k_frac:     0.005,
            jitter_xy:      0.05,
            jitter_yaw:     0.10,
            occ_threshold_fp: 150,
        }
    }
}

/// One particle: x, y, yaw.
#[derive(Debug, Clone, Copy)]
pub struct Particle { pub x: f32, pub y: f32, pub yaw: f32 }

pub struct Localizer {
    cfg:        MclConfig,
    particles:  Vec<Particle>,
    weights:    Vec<f32>,
    rng:        StdRng,
    last_residual_m: f32,
    /// EWMAs of the average particle weight per update — the textbook
    /// augmented-MCL lost-detection signal.
    w_slow: f32,
    w_fast: f32,
    w_avg_initialized: bool,
}

impl Localizer {
    pub fn new(grid: &OccupancyGrid, cfg: MclConfig, rng_seed: u64) -> Self {
        let n = cfg.n_particles;
        let mut s = Self {
            cfg,
            particles: vec![Particle { x: 0.0, y: 0.0, yaw: 0.0 }; n],
            weights:   vec![1.0 / n as f32; n],
            rng:       StdRng::seed_from_u64(rng_seed),
            last_residual_m: f32::NAN,
            w_slow: 0.0,
            w_fast: 0.0,
            w_avg_initialized: false,
        };
        s.reset_uniform(grid);
        s
    }

    pub fn cfg(&self) -> &MclConfig { &self.cfg }
    pub fn particles(&self) -> &[Particle] { &self.particles }
    pub fn weights(&self) -> &[f32] { &self.weights }
    pub fn last_residual_m(&self) -> f32 { self.last_residual_m }

    // ── State init ─────────────────────────────────────────────────────────

    /// Spread particles over known-free cells (kidnap state). Falls back
    /// to grid bounds if too few free cells exist.
    pub fn reset_uniform(&mut self, grid: &OccupancyGrid) {
        let cell = grid.cell();
        let cfg  = grid.cfg();
        let free = collect_free_cells(grid);
        let n = self.cfg.n_particles;
        for k in 0..n {
            let (x, y) = if free.len() >= 50 {
                let (i, j) = free[self.rng.gen_range(0..free.len())];
                let jit_x: f32 = self.rng.gen_range(-0.5..0.5);
                let jit_y: f32 = self.rng.gen_range(-0.5..0.5);
                (cfg.x_range.0 + (j as f32 + 0.5) * cell + jit_x * cell,
                 cfg.y_range.0 + (i as f32 + 0.5) * cell + jit_y * cell)
            } else {
                (self.rng.gen_range(cfg.x_range.0..cfg.x_range.1),
                 self.rng.gen_range(cfg.y_range.0..cfg.y_range.1))
            };
            let yaw: f32 = self.rng.gen_range(-PI..PI);
            self.particles[k] = Particle { x, y, yaw };
        }
        self.weights.iter_mut().for_each(|w| *w = 1.0 / n as f32);
        // New cloud → forget the old fit-quality EWMA so the next update
        // seeds them fresh and we don't immediately fire injection on
        // the first tick.
        self.w_avg_initialized = false;
        self.w_slow = 0.0;
        self.w_fast = 0.0;
    }

    /// Tight cluster around a known pose (e.g. on dock undocking).
    pub fn reset_known(&mut self, x: f32, y: f32, yaw: f32) {
        let n = self.cfg.n_particles;
        let nx = Normal::new(x as f64,   0.05).unwrap();
        let ny = Normal::new(y as f64,   0.05).unwrap();
        let nyaw = Normal::new(yaw as f64, 0.05).unwrap();
        for k in 0..n {
            self.particles[k] = Particle {
                x:   nx.sample(&mut self.rng) as f32,
                y:   ny.sample(&mut self.rng) as f32,
                yaw: nyaw.sample(&mut self.rng) as f32,
            };
        }
        self.weights.iter_mut().for_each(|w| *w = 1.0 / n as f32);
        // New cloud → forget the old fit-quality EWMA so the next update
        // seeds them fresh and we don't immediately fire injection on
        // the first tick.
        self.w_avg_initialized = false;
        self.w_slow = 0.0;
        self.w_fast = 0.0;
    }

    // ── Predict ────────────────────────────────────────────────────────────

    /// Apply odometry delta in body frame, with noise, to every particle.
    pub fn predict(&mut self, dx_body: f32, dy_body: f32, dyaw: f32) {
        let cfg = self.cfg;
        let trans = (dx_body * dx_body + dy_body * dy_body).sqrt();
        let sigma_xy  = (cfg.sigma_xy_per_m  * trans + cfg.sigma_xy_per_rad  * dyaw.abs()).max(1e-3);
        let sigma_yaw = (cfg.sigma_yaw_per_m * trans + cfg.sigma_yaw_per_rad * dyaw.abs()).max(1e-3);
        let nxy  = Normal::new(0.0_f64, sigma_xy as f64).unwrap();
        let nyaw = Normal::new(0.0_f64, sigma_yaw as f64).unwrap();
        for p in self.particles.iter_mut() {
            let c = p.yaw.cos();
            let s = p.yaw.sin();
            p.x += c * dx_body - s * dy_body + nxy.sample(&mut self.rng) as f32;
            p.y += s * dx_body + c * dy_body + nxy.sample(&mut self.rng) as f32;
            p.yaw += dyaw + nyaw.sample(&mut self.rng) as f32;
            // Wrap into [-pi, pi].
            p.yaw = ((p.yaw + PI).rem_euclid(2.0 * PI)) - PI;
        }
    }

    // ── Update ─────────────────────────────────────────────────────────────

    /// Measurement update from a 2D scan in the body frame.
    ///
    /// `angles` and `ranges` are parallel arrays; ranges that are NaN,
    /// non-finite, ≤ 0, or ≥ z_max are dropped. From the remaining
    /// beams up to `n_beams_used` are subsampled evenly.
    pub fn update(&mut self, grid: &OccupancyGrid,
                  angles: &[f32], ranges: &[f32]) {
        debug_assert_eq!(angles.len(), ranges.len());
        let cfg = self.cfg;
        let z_near = cfg.z_max - cfg.skip_max_margin;
        let sig2 = 2.0 * cfg.beam_sigma * cfg.beam_sigma;

        // Filter to valid beams + subsample.
        let valid_idx: Vec<usize> = (0..angles.len())
            .filter(|&k| {
                let r = ranges[k];
                r.is_finite() && r > 0.0 && r < cfg.z_max
            })
            .collect();
        if valid_idx.is_empty() { return; }
        let used: Vec<usize> = if valid_idx.len() <= cfg.n_beams_used {
            valid_idx
        } else {
            let n = cfg.n_beams_used;
            (0..n)
                .map(|i| valid_idx[((i as f32 / (n - 1).max(1) as f32) * (valid_idx.len() - 1) as f32).round() as usize])
                .collect()
        };

        // For each used beam, cast from every particle in parallel.
        // log_w accumulates per particle (Vec<f32>). `scored` keeps the
        // (a, r, pred_per_particle) so we can compute the clean residual
        // for the best particle after weights are settled.
        let n = self.cfg.n_particles;
        let mut log_w = vec![0.0_f32; n];
        let mut scored: Vec<(f32, f32, Vec<f32>)> = Vec::with_capacity(used.len());

        for &k in &used {
            let a = angles[k];
            let r = ranges[k];
            if r >= z_near { continue; }
            let pred: Vec<f32> = self.particles
                .par_iter()
                .map(|p| grid.cast_ray(p.x, p.y, p.yaw + a, cfg.z_max))
                .collect();
            // Per-particle cost: gaussian on known beams, flat penalty
            // on unknown ones (pred ≈ z_max).
            log_w.iter_mut().zip(&pred).for_each(|(lw, &pp)| {
                let cost = if pp >= z_near {
                    -cfg.beam_logw_floor   // (positive: subtracted from log_w)
                } else {
                    let d = pp - r;
                    d * d / sig2
                };
                *lw -= cost;
            });
            scored.push((a, r, pred));
        }

        if scored.is_empty() { return; }

        // Normalize weights via subtract-max + exp.
        let log_w_max = log_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut w_unnorm = vec![0.0_f32; n];
        for (i, &lw) in log_w.iter().enumerate() {
            w_unnorm[i] = (lw - log_w_max).exp();
        }
        let s: f32 = w_unnorm.iter().sum();
        if !(s > 0.0 && s.is_finite()) {
            self.last_residual_m = f32::NAN;
            return;
        }
        // Textbook augmented-MCL: track average particle weight (post-
        // max-subtract — bounded in (0, 1]) on slow + fast EWMAs. A
        // sustained drop in `w_fast` relative to `w_slow` means the
        // recent fit is much worse than the long-term baseline → lost.
        let w_avg = s / n as f32;
        if !self.w_avg_initialized {
            self.w_slow = w_avg;
            self.w_fast = w_avg;
            self.w_avg_initialized = true;
        } else {
            self.w_slow += cfg.alpha_slow * (w_avg - self.w_slow);
            self.w_fast += cfg.alpha_fast * (w_avg - self.w_fast);
        }
        for (w, &u) in self.weights.iter_mut().zip(w_unnorm.iter()) {
            *w = u / s;
        }

        // Clean residual: best particle's RMS error over known beams only.
        let best = self.weights.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let (mut sq, mut n_known) = (0.0_f32, 0u32);
        for (_a, r, pred) in &scored {
            let p_b = pred[best];
            if p_b >= z_near { continue; }
            let d = p_b - r;
            sq += d * d;
            n_known += 1;
        }
        self.last_residual_m = if n_known > 0 {
            (sq / n_known as f32).sqrt()
        } else { f32::NAN };

        if self.neff() < cfg.neff_threshold_frac * n as f32 {
            self.systematic_resample();
        }

        // Augmented-MCL injection: probabilistic, capped, no manual gates.
        // p_inject per particle = max(0, 1 - w_fast/w_slow). When the
        // filter is well-locked, w_fast ≈ w_slow → no injection. When
        // the recent fit dips (kidnap, walking off the mapped region,
        // etc.), w_fast drops faster than w_slow → some particles get
        // replaced with random samples. Once w_slow catches up, tapers.
        let p_inject = if self.w_slow > 0.0 {
            (1.0 - self.w_fast / self.w_slow).max(0.0)
        } else { 0.0 };
        let n_inject = (p_inject * n as f32) as usize;
        let cap = (cfg.max_inject_frac * n as f32) as usize;
        let n_inject = n_inject.min(cap);
        if n_inject > 0 {
            self.inject_random(grid, n_inject);
        }
    }

    // ── Update from accumulated wide scan ──────────────────────────────────

    /// MCL update from a buffered wide scan (multiple viewpoints).
    /// Each beam uses the localizer's belief at capture to compute
    /// per-particle pose-at-capture under the constant-offset
    /// approximation. Mirrors the Python `Localizer.update_accumulated`.
    pub fn update_accumulated(
        &mut self,
        grid: &OccupancyGrid,
        beams: &[BufferedBeam],
        current_est_pose: (f32, f32, f32),
    ) {
        if beams.is_empty() { return; }
        let cfg = self.cfg;
        let z_near = cfg.z_max - cfg.skip_max_margin;
        let sig2 = 2.0 * cfg.beam_sigma * cfg.beam_sigma;

        // Subsample buffered beams. ~32 strided beams keep wide
        // diversity while bounding the per-flush cost.
        const MAX_BEAMS_IN_UPDATE: usize = 32;
        let beams_used: Vec<&BufferedBeam> = if beams.len() > MAX_BEAMS_IN_UPDATE {
            let stride = (beams.len() / MAX_BEAMS_IN_UPDATE).max(1);
            beams.iter().step_by(stride).take(MAX_BEAMS_IN_UPDATE).collect()
        } else {
            beams.iter().collect()
        };

        let (cur_x, cur_y, cur_yaw) = current_est_pose;
        let n = self.cfg.n_particles;
        let mut log_w = vec![0.0_f32; n];

        // Cache (beam, pred-per-particle) for the post-update residual.
        let mut scored: Vec<(&BufferedBeam, Vec<f32>)> = Vec::with_capacity(beams_used.len());
        for beam in beams_used {
            if beam.range_m >= z_near { continue; }
            let dx_off   = beam.est_origin.0 - cur_x;
            let dy_off   = beam.est_origin.1 - cur_y;
            let dyaw_off = wrap_pi(beam.est_yaw - cur_yaw);
            let pred: Vec<f32> = self.particles.par_iter().map(|p| {
                let px = p.x + dx_off;
                let py = p.y + dy_off;
                let theta = p.yaw + dyaw_off + beam.angle_body;
                grid.cast_ray(px, py, theta, cfg.z_max)
            }).collect();
            log_w.iter_mut().zip(&pred).for_each(|(lw, &pp)| {
                let cost = if pp >= z_near {
                    -cfg.beam_logw_floor
                } else {
                    let d = pp - beam.range_m;
                    d * d / sig2
                };
                *lw -= cost;
            });
            scored.push((beam, pred));
        }
        if scored.is_empty() { return; }

        let log_w_max = log_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for (w, lw) in self.weights.iter_mut().zip(&log_w) {
            *w = (*lw - log_w_max).exp();
        }
        let s: f32 = self.weights.iter().sum();
        if !(s > 0.0 && s.is_finite()) {
            self.last_residual_m = f32::NAN;
            return;
        }
        for w in self.weights.iter_mut() { *w /= s; }

        // Clean residual: best particle's RMS error over known beams.
        let best = self.weights.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let (mut sq, mut n_known) = (0.0_f32, 0u32);
        for (beam, pred) in &scored {
            let p_b = pred[best];
            if p_b >= z_near { continue; }
            let d = p_b - beam.range_m;
            sq += d * d;
            n_known += 1;
        }
        self.last_residual_m = if n_known > 0 {
            (sq / n_known as f32).sqrt()
        } else { f32::NAN };

        if self.neff() < cfg.neff_threshold_frac * n as f32 {
            self.systematic_resample();
        }

        // Wide-scan flush failed to find a good fit → re-disperse.
        const WIDE_RESID_BAD_M: f32 = 0.50;
        if self.last_residual_m.is_finite() && self.last_residual_m > WIDE_RESID_BAD_M {
            self.inject_random(grid, n / 2);
        }
    }

    // ── Global relocalize (likelihood field) ───────────────────────────────

    /// Coarse global scan-match using the precomputed distance field.
    ///
    /// Replaces the particle cloud with samples drawn from the top-K
    /// hypotheses on a (free-cell × yaw) grid, scored against the
    /// accumulated wide scan with the textbook field model
    /// `exp(-d² / 2σ²)` where `d` is the distance from each beam's
    /// endpoint to the nearest mapped obstacle. This is the kidnap
    /// recovery primitive — cheap (one O(1) lookup per beam per
    /// hypothesis instead of an 80-step ray march), so we can call it
    /// after every accumulator flush while lost.
    ///
    /// Defaults match the Python `Localizer.global_relocalize_field`
    /// in `microduck_maploc/sim/localizer.py`.
    pub fn global_relocalize_field(
        &mut self,
        grid: &mut OccupancyGrid,
        beams: &[BufferedBeam],
        current_est_pose: (f32, f32, f32),
    ) {
        self.global_relocalize_field_with(
            grid, beams, current_est_pose,
            FieldRelocConfig::default(),
        );
    }

    pub fn global_relocalize_field_with(
        &mut self,
        grid: &mut OccupancyGrid,
        beams: &[BufferedBeam],
        current_est_pose: (f32, f32, f32),
        params: FieldRelocConfig,
    ) {
        if beams.is_empty() { return; }
        let cfg = self.cfg;
        let z_near = cfg.z_max - cfg.skip_max_margin;

        // Hypothesis grid (free cells × yaw bins), subsampled.
        let mut free_cells = collect_free_cells(grid);
        if free_cells.is_empty() { return; }
        if params.cell_subsample > 1 {
            free_cells = free_cells.into_iter()
                .step_by(params.cell_subsample as usize).collect();
        }
        let n_cells = free_cells.len();
        let n_yaw = params.n_yaw_bins as usize;
        let n_hyp = n_cells * n_yaw;
        let cell = grid.cell();
        let cfg_g = grid.cfg();
        let x0 = cfg_g.x_range.0;
        let y0 = cfg_g.y_range.0;
        let mut hx  = vec![0.0_f32; n_hyp];
        let mut hy  = vec![0.0_f32; n_hyp];
        let mut hyw = vec![0.0_f32; n_hyp];
        for j in 0..n_yaw {
            let yaw_b = (j as f32 + 0.5) * (2.0 * PI / n_yaw as f32) - PI;
            for (k, &(i, c)) in free_cells.iter().enumerate() {
                let idx = j * n_cells + k;
                hx[idx]  = x0 + (c as f32 + 0.5) * cell;
                hy[idx]  = y0 + (i as f32 + 0.5) * cell;
                hyw[idx] = yaw_b;
            }
        }

        // Beam subsample — global match doesn't need every beam.
        let beams_used: Vec<&BufferedBeam> = if beams.len() > params.beam_subsample as usize {
            let stride = (beams.len() / params.beam_subsample as usize).max(1);
            beams.iter().step_by(stride).take(params.beam_subsample as usize).collect()
        } else {
            beams.iter().collect()
        };

        let (cur_x, cur_y, cur_yaw) = current_est_pose;
        // Snapshot grid metadata BEFORE the mut-borrowing distance_field
        // call — we can't immutably reach into `grid` while the field
        // slice is alive.
        let h = grid.height();
        let w = grid.width();
        let field = grid.distance_field(params.occ_threshold_fp);
        let cell_inv = 1.0 / cell;
        let sig2 = 2.0 * params.sigma_m * params.sigma_m;

        let mut log_w = vec![0.0_f64; n_hyp];
        let unmapped_d = params.sigma_m * 3.0;   // out-of-bounds penalty distance
        for beam in &beams_used {
            if beam.range_m >= z_near { continue; }
            let dx_off   = beam.est_origin.0 - cur_x;
            let dy_off   = beam.est_origin.1 - cur_y;
            let dyaw_off = wrap_pi(beam.est_yaw - cur_yaw);
            for k in 0..n_hyp {
                let theta = hyw[k] + dyaw_off + beam.angle_body;
                let ex = (hx[k] + dx_off) + beam.range_m * theta.cos();
                let ey = (hy[k] + dy_off) + beam.range_m * theta.sin();
                let jj = ((ex - x0) * cell_inv) as i32;
                let ii = ((ey - y0) * cell_inv) as i32;
                let d = if ii >= 0 && jj >= 0 && (ii as usize) < h && (jj as usize) < w {
                    field[(ii as usize) * w + (jj as usize)]
                } else {
                    unmapped_d
                };
                log_w[k] -= (d as f64) * (d as f64) / (sig2 as f64);
            }
        }

        // Normalise via subtract-max + exp.
        let log_w_max = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut w: Vec<f64> = log_w.iter().map(|lw| (*lw - log_w_max).exp()).collect();
        let s: f64 = w.iter().sum();
        if !(s > 0.0 && s.is_finite()) { return; }
        for v in w.iter_mut() { *v /= s; }

        // Sample new particles from the top-K hypotheses, weighted.
        let n = self.cfg.n_particles;
        let k = (50_usize).max(((params.top_k_frac as f64 * n_hyp as f64) as usize).min(n_hyp));
        let mut indexed: Vec<(usize, f64)> = (0..n_hyp).map(|i| (i, w[i])).collect();
        // Partial sort: pick top-k by weight (descending). For small k vs n_hyp
        // a heap or quickselect would be faster; std's `select_nth_unstable_by`
        // gives us O(n) selection.
        if k < n_hyp {
            indexed.select_nth_unstable_by(n_hyp - k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let top: Vec<(usize, f64)> = indexed.into_iter().rev().take(k).collect();
        let top_sum: f64 = top.iter().map(|(_, w)| *w).sum();
        if !(top_sum > 0.0) { return; }
        let mut cum = Vec::with_capacity(top.len());
        let mut acc = 0.0_f64;
        for (_, w) in &top {
            acc += w / top_sum;
            cum.push(acc);
        }

        let xy_noise  = Normal::new(0.0_f64, params.jitter_xy as f64).unwrap();
        let yaw_noise = Normal::new(0.0_f64, params.jitter_yaw as f64).unwrap();
        for slot in 0..n {
            let r: f64 = self.rng.gen();
            // Sequential search through CDF — short array (k ≤ a few thousand).
            let pos = cum.partition_point(|c| *c < r).min(top.len() - 1);
            let h_idx = top[pos].0;
            let jx: f64 = xy_noise.sample(&mut self.rng);
            let jy: f64 = xy_noise.sample(&mut self.rng);
            let jyaw: f64 = yaw_noise.sample(&mut self.rng);
            let mut yaw = hyw[h_idx] + jyaw as f32;
            yaw = ((yaw + PI).rem_euclid(2.0 * PI)) - PI;
            self.particles[slot] = Particle {
                x:   hx[h_idx] + jx as f32,
                y:   hy[h_idx] + jy as f32,
                yaw,
            };
        }
        let inv = 1.0 / n as f32;
        for w_p in self.weights.iter_mut() { *w_p = inv; }
        // Reset augmented-MCL EWMAs so the new cloud isn't immediately
        // judged "lost" against an old fit baseline.
        self.w_avg_initialized = false;
        self.w_slow = 0.0;
        self.w_fast = 0.0;
    }

    // ── Estimates ──────────────────────────────────────────────────────────

    /// Weighted mean (xy) + circular mean (yaw).
    pub fn best(&self) -> (f32, f32, f32) {
        let mut x = 0.0_f32; let mut y = 0.0_f32;
        let mut sy = 0.0_f32; let mut cy = 0.0_f32;
        for (p, &w) in self.particles.iter().zip(&self.weights) {
            x  += p.x   * w;
            y  += p.y   * w;
            sy += p.yaw.sin() * w;
            cy += p.yaw.cos() * w;
        }
        (x, y, sy.atan2(cy))
    }

    /// Weighted xy standard deviation. High = cloud dispersed = lost.
    pub fn position_std(&self) -> f32 {
        let (bx, by, _) = self.best();
        let mut var = 0.0_f32;
        for (p, &w) in self.particles.iter().zip(&self.weights) {
            let dx = p.x - bx; let dy = p.y - by;
            var += w * (dx * dx + dy * dy);
        }
        var.max(0.0).sqrt()
    }

    /// UNweighted xy std — physical particle dispersion. Right "are we
    /// lost?" signal: stays large until resample has actually collapsed
    /// the cloud, regardless of how spiky the weights are.
    pub fn cloud_spread(&self) -> f32 {
        let n = self.particles.len() as f32;
        if n == 0.0 { return 0.0; }
        let mut mx = 0.0_f32; let mut my = 0.0_f32;
        for p in &self.particles { mx += p.x; my += p.y; }
        mx /= n; my /= n;
        let mut var = 0.0_f32;
        for p in &self.particles {
            let dx = p.x - mx; let dy = p.y - my;
            var += dx * dx + dy * dy;
        }
        (var / n).max(0.0).sqrt()
    }

    // ── Internals ──────────────────────────────────────────────────────────

    fn neff(&self) -> f32 {
        let s2: f32 = self.weights.iter().map(|w| w * w).sum();
        1.0 / (s2 + 1e-30)
    }

    fn inject_random(&mut self, grid: &OccupancyGrid, k: usize) {
        if k == 0 { return; }
        let n = self.cfg.n_particles;
        let cell = grid.cell();
        let cfg  = grid.cfg();
        let free = collect_free_cells(grid);
        // Sample k DISTINCT particle indices to overwrite.
        let mut idxs: Vec<usize> = (0..n).collect();
        // partial Fisher-Yates: only first k positions
        for i in 0..k {
            let j = self.rng.gen_range(i..n);
            idxs.swap(i, j);
        }
        for &slot in &idxs[..k] {
            let (x, y) = if free.len() >= 50 {
                let (ii, jj) = free[self.rng.gen_range(0..free.len())];
                let jx: f32 = self.rng.gen_range(-0.5..0.5);
                let jy: f32 = self.rng.gen_range(-0.5..0.5);
                (cfg.x_range.0 + (jj as f32 + 0.5) * cell + jx * cell,
                 cfg.y_range.0 + (ii as f32 + 0.5) * cell + jy * cell)
            } else {
                (self.rng.gen_range(cfg.x_range.0..cfg.x_range.1),
                 self.rng.gen_range(cfg.y_range.0..cfg.y_range.1))
            };
            let yaw: f32 = self.rng.gen_range(-PI..PI);
            self.particles[slot] = Particle { x, y, yaw };
        }
        // Even out weights — fresh particles haven't been measured yet.
        let inv = 1.0 / n as f32;
        for w in self.weights.iter_mut() { *w = inv; }
    }

    fn systematic_resample(&mut self) {
        let n = self.cfg.n_particles;
        let r0: f32 = self.rng.gen_range(0.0..1.0);
        // CDF.
        let mut cum = vec![0.0_f32; n];
        let mut acc = 0.0_f32;
        for (i, &w) in self.weights.iter().enumerate() {
            acc += w;
            cum[i] = acc;
        }
        if let Some(last) = cum.last_mut() { *last = 1.0; }   // numerical safety
        // Sample positions and look them up.
        let mut new_particles = Vec::with_capacity(n);
        let mut j = 0usize;
        for i in 0..n {
            let pos = (i as f32 + r0) / n as f32;
            while j + 1 < n && cum[j] < pos { j += 1; }
            new_particles.push(self.particles[j]);
        }
        self.particles = new_particles;
        let inv = 1.0 / n as f32;
        for w in self.weights.iter_mut() { *w = inv; }
    }
}

/// Indices of cells that have been explicitly observed as free.
fn collect_free_cells(grid: &OccupancyGrid) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..grid.height() {
        for j in 0..grid.width() {
            if grid.is_known_free(i, j) {
                out.push((i, j));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{GridConfig, OccupancyGrid};

    fn small_grid_with_walls() -> OccupancyGrid {
        // 1.5×1.5 m grid; box of walls forming a small enclosed room
        // around the origin so MCL has something to localize against.
        let mut g = OccupancyGrid::new(GridConfig {
            x_range: (-0.75, 0.75), y_range: (-0.75, 0.75), cell: 0.05,
        });
        // Hammer all four walls of a box at ±0.6 from origin.
        for t in 0..30 {
            let p = -0.6 + (t as f32) * 0.04;
            for _ in 0..6 {
                g.integrate_ray(0.0, 0.0,  0.6,  p, true);
                g.integrate_ray(0.0, 0.0, -0.6,  p, true);
                g.integrate_ray(0.0, 0.0,    p,  0.6, true);
                g.integrate_ray(0.0, 0.0,    p, -0.6, true);
            }
        }
        g
    }

    /// Eight-beam horizontal scan from (x, y, yaw) — what MCL would
    /// receive on the wire.
    fn fake_scan(grid: &OccupancyGrid, x: f32, y: f32, yaw: f32) -> (Vec<f32>, Vec<f32>) {
        let n = 8;
        let half = std::f32::consts::PI * 0.5; // 90° fan, just to see walls
        let step = (2.0 * half) / n as f32;
        let mut angs = Vec::with_capacity(n);
        let mut rngs = Vec::with_capacity(n);
        for i in 0..n {
            let a = -half + (i as f32 + 0.5) * step;
            angs.push(a);
            let r = grid.cast_ray(x, y, yaw + a, 4.0);
            rngs.push(r);
        }
        (angs, rngs)
    }

    #[test]
    fn predict_keeps_known_pose_tight() {
        let g = small_grid_with_walls();
        let mut loc = Localizer::new(&g, MclConfig::default(), 0);
        loc.reset_known(0.0, 0.0, 0.0);
        // Drift forward a small bit in body frame.
        for _ in 0..10 {
            loc.predict(0.02, 0.0, 0.0);
        }
        let std = loc.position_std();
        assert!(std < 0.10, "expected tight cloud after small predicts, got std={std}");
    }

    #[test]
    fn update_pulls_estimate_to_truth() {
        let g = small_grid_with_walls();
        let mut loc = Localizer::new(&g, MclConfig::default(), 1);
        loc.reset_known(0.05, -0.05, 0.0);   // slightly off
        let (a, r) = fake_scan(&g, 0.0, 0.0, 0.0);
        for _ in 0..3 {
            loc.update(&g, &a, &r);
        }
        let (bx, by, _) = loc.best();
        let err = (bx * bx + by * by).sqrt();
        assert!(err < 0.10, "estimate should drift toward truth, got ({bx}, {by})");
    }

    #[test]
    fn global_relocalize_field_runs_and_seeds_particles() {
        // Build a small mapped room, run global_relocalize_field with
        // a fake wide scan, verify particles land in the mapped area.
        let mut g = small_grid_with_walls();
        let mut loc = Localizer::new(&g, MclConfig { n_particles: 200, ..MclConfig::default() }, 3);
        // Fake beams: synthesise them from cast_ray at the origin.
        let mut beams: Vec<BufferedBeam> = Vec::new();
        for k in 0..16 {
            let a = -std::f32::consts::PI + (k as f32) * (std::f32::consts::PI / 8.0);
            let r = g.cast_ray(0.0, 0.0, a, 4.0);
            beams.push(BufferedBeam {
                angle_body: a, range_m: r,
                est_origin: (0.0, 0.0), est_yaw: 0.0,
            });
        }
        loc.global_relocalize_field(&mut g, &beams, (0.0, 0.0, 0.0));
        // Particles should now cluster within the apartment bounds.
        for p in loc.particles() {
            assert!(p.x >= -0.8 && p.x <= 0.8, "x out of bounds: {}", p.x);
            assert!(p.y >= -0.8 && p.y <= 0.8, "y out of bounds: {}", p.y);
        }
    }

    #[test]
    fn best_handles_uniform_weights() {
        // Weighted mean of a uniform cloud near origin should be near origin.
        let g = small_grid_with_walls();
        let loc = Localizer::new(&g, MclConfig { n_particles: 200, ..MclConfig::default() }, 7);
        let (x, y, _) = loc.best();
        // Particles seeded in known-free cells around origin → mean near 0.
        assert!(x.abs() < 0.6 && y.abs() < 0.6, "uniform best off: ({x}, {y})");
    }
}
