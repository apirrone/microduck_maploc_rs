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

use crate::grid::OccupancyGrid;

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
    // Augmented MCL — searching mode (cloud dispersed): inject every tick.
    pub inject_std_m:    f32,
    pub inject_resid_m:  f32,
    pub inject_frac_max: f32,
    // Tracking mode (cloud tight): only inject after sustained catastrophe.
    pub track_resid_catastrophe_m: f32,
    pub track_inject_frac:         f32,
    pub track_catastrophe_persist: u32,
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
            inject_std_m:    0.30,
            inject_resid_m:  0.20,
            inject_frac_max: 0.40,
            track_resid_catastrophe_m: 1.5,
            track_inject_frac:         0.05,
            track_catastrophe_persist: 8,
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
    catastrophe_count: u32,
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
            catastrophe_count: 0,
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
        self.catastrophe_count = 0;
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
        self.catastrophe_count = 0;
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
        for (w, lw) in self.weights.iter_mut().zip(&log_w) {
            *w = (*lw - log_w_max).exp();
        }
        let s: f32 = self.weights.iter().sum();
        if !(s > 0.0 && s.is_finite()) {
            self.last_residual_m = f32::NAN;
            return;
        }
        for w in self.weights.iter_mut() { *w /= s; }

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

        // Two-mode injection — searching vs tracking with persistence.
        let std_now = self.position_std();
        if !self.last_residual_m.is_finite() { return; }
        if std_now > cfg.inject_std_m {
            // SEARCHING — inject every tick proportional to residual ramp.
            if self.last_residual_m > cfg.inject_resid_m {
                let ramp = (self.last_residual_m - cfg.inject_resid_m) / cfg.inject_resid_m;
                let frac = (cfg.inject_frac_max * ramp).min(cfg.inject_frac_max);
                self.inject_random(grid, (frac * n as f32) as usize);
            }
            self.catastrophe_count = 0;
        } else {
            // TRACKING — only after sustained catastrophe.
            if self.last_residual_m > cfg.track_resid_catastrophe_m {
                self.catastrophe_count += 1;
            } else {
                self.catastrophe_count = 0;
            }
            if self.catastrophe_count >= cfg.track_catastrophe_persist {
                self.inject_random(grid, (cfg.track_inject_frac * n as f32) as usize);
                self.catastrophe_count = 0;
            }
        }
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
    fn best_handles_uniform_weights() {
        // Weighted mean of a uniform cloud near origin should be near origin.
        let g = small_grid_with_walls();
        let loc = Localizer::new(&g, MclConfig { n_particles: 200, ..MclConfig::default() }, 7);
        let (x, y, _) = loc.best();
        // Particles seeded in known-free cells around origin → mean near 0.
        assert!(x.abs() < 0.6 && y.abs() < 0.6, "uniform best off: ({x}, {y})");
    }
}
