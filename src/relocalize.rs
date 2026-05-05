//! Brute-force relocalize against a saved global map.
//!
//! Used for kidnapped-robot recovery: given a single ToF scan and a
//! prior occupancy grid (e.g. the global render of a loaded session),
//! search over `(x, y, yaw)` candidates and return the pose with the
//! smallest mean per-beam residual against the grid's distance field.
//!
//! Cost is O(N_free × N_yaw × N_beams). On a 4×4 m room at 5 cm cells,
//! that's ~6400 free cells × 36 yaw bins × 64 beams = ~15 M lookups —
//! a few hundred ms of one-shot search at the start of a session, then
//! we go back to the regular live SLAM pipeline.
//!
//! Two-stage search: a coarse pass over the full grid at `cfg.coarse_xy_stride`
//! cells / `cfg.coarse_yaw_bins` bins, followed by a refinement pass at
//! single-cell / fine-bin resolution around the best coarse candidate.

use crate::grid::{OccupancyGrid, OCC_THRESHOLD};

#[derive(Debug, Clone)]
pub struct RelocalizeConfig {
    /// Coarse stride over (x, y) free cells.
    pub coarse_xy_stride: usize,
    /// Coarse number of yaw bins covering 0..2π.
    pub coarse_yaw_bins: usize,
    /// Half-width of the local refinement window (cells).
    pub refine_xy_radius: usize,
    /// Refinement yaw bins covering ±`refine_yaw_half_rad`.
    pub refine_yaw_bins: usize,
    pub refine_yaw_half_rad: f32,
    /// Acceptance threshold on the mean per-beam residual (metres).
    pub max_mean_residual_m: f32,
    /// Minimum number of valid beams a candidate must explain.
    pub min_beams_used: u32,
    /// Per-beam residual is clamped to this so a single very-far beam
    /// can't dominate the mean (matches the Hector saturation trick).
    pub clamp_m: f32,
}

impl Default for RelocalizeConfig {
    fn default() -> Self {
        Self {
            coarse_xy_stride:    2,
            coarse_yaw_bins:     36,
            refine_xy_radius:    4,
            refine_yaw_bins:     11,
            refine_yaw_half_rad: 10.0_f32.to_radians(),
            max_mean_residual_m: 0.20,
            min_beams_used:      16,
            clamp_m:             0.50,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RelocalizeResult {
    pub pose:            (f32, f32, f32),
    pub mean_residual_m: f32,
    pub n_beams_used:    u32,
    /// True iff `mean_residual_m <= cfg.max_mean_residual_m` and
    /// `n_beams_used >= cfg.min_beams_used`.
    pub accepted:        bool,
}

/// Score a single (cx, cy, yaw) candidate against a precomputed
/// distance field. Returns `(sum_clamped_residual, n_beams_used)`.
#[inline]
fn score_candidate(
    cx: f32, cy: f32, yaw: f32,
    angles_body: &[f32], ranges: &[f32],
    field: &[f32], w: usize, h: usize,
    x_min: f32, y_min: f32, cell: f32,
    clamp_m: f32,
) -> (f32, u32) {
    let mut sum = 0.0_f32;
    let mut n = 0u32;
    for (a, &r) in angles_body.iter().zip(ranges) {
        if !r.is_finite() || r <= 0.0 { continue; }
        let theta = yaw + a;
        let hx = cx + r * theta.cos();
        let hy = cy + r * theta.sin();
        let j = ((hx - x_min) / cell) as i32;
        let i = ((hy - y_min) / cell) as i32;
        if i < 0 || j < 0 || (i as usize) >= h || (j as usize) >= w {
            continue;
        }
        let d = field[(i as usize) * w + (j as usize)];
        sum += d.min(clamp_m);
        n += 1;
    }
    (sum, n)
}

/// Search the grid for the best matching pose. Returns the global
/// minimum-residual candidate (ignoring acceptance — caller checks
/// `accepted` to decide whether to use the pose).
pub fn relocalize_against_grid(
    grid: &mut OccupancyGrid,
    angles_body: &[f32],
    ranges: &[f32],
    cfg: &RelocalizeConfig,
) -> Option<RelocalizeResult> {
    if angles_body.is_empty() || ranges.is_empty() { return None; }
    if angles_body.len() != ranges.len() { return None; }

    // Take a copy of the distance field so we can drop the mutable
    // borrow before the search loop accesses immutable grid methods.
    let field = grid.distance_field(OCC_THRESHOLD).to_vec();
    let cfg_g = *grid.cfg();
    let w = grid.width(); let h = grid.height();
    let cell = cfg_g.cell;
    let x_min = cfg_g.x_range.0;
    let y_min = cfg_g.y_range.0;

    let two_pi = 2.0 * std::f32::consts::PI;
    let coarse_yaw_bins = cfg.coarse_yaw_bins.max(1);
    let dyaw = two_pi / coarse_yaw_bins as f32;

    // Phase 1 — coarse pass over free cells × yaw bins.
    let mut best: Option<RelocalizeResult> = None;
    let stride = cfg.coarse_xy_stride.max(1);
    for ci in (0..h).step_by(stride) {
        for cj in (0..w).step_by(stride) {
            if !grid.is_known_free(ci, cj) { continue; }
            let cx = x_min + (cj as f32 + 0.5) * cell;
            let cy = y_min + (ci as f32 + 0.5) * cell;
            for yi in 0..coarse_yaw_bins {
                let yaw = -std::f32::consts::PI + yi as f32 * dyaw;
                let (sum, n) = score_candidate(
                    cx, cy, yaw, angles_body, ranges,
                    &field, w, h, x_min, y_min, cell, cfg.clamp_m);
                if n < cfg.min_beams_used { continue; }
                let mean = sum / (n as f32);
                if best.as_ref().map_or(true, |b| mean < b.mean_residual_m) {
                    best = Some(RelocalizeResult {
                        pose: (cx, cy, yaw),
                        mean_residual_m: mean,
                        n_beams_used: n,
                        accepted: false,
                    });
                }
            }
        }
    }
    let mut best = best?;

    // Phase 2 — local refinement around the coarse winner.
    let (bx, by, byaw) = best.pose;
    let bj = ((bx - x_min) / cell) as i32;
    let bi = ((by - y_min) / cell) as i32;
    let r_xy = cfg.refine_xy_radius as i32;
    let yaw_bins = cfg.refine_yaw_bins.max(1);
    let yaw_step = if yaw_bins == 1 { 0.0 }
                   else { 2.0 * cfg.refine_yaw_half_rad / (yaw_bins - 1) as f32 };
    for ddi in -r_xy..=r_xy {
        for ddj in -r_xy..=r_xy {
            let ci = bi + ddi; let cj = bj + ddj;
            if ci < 0 || cj < 0
                || (ci as usize) >= h || (cj as usize) >= w { continue; }
            if !grid.is_known_free(ci as usize, cj as usize) { continue; }
            let cx = x_min + (cj as f32 + 0.5) * cell;
            let cy = y_min + (ci as f32 + 0.5) * cell;
            for yi in 0..yaw_bins {
                let yaw = byaw - cfg.refine_yaw_half_rad + yi as f32 * yaw_step;
                let (sum, n) = score_candidate(
                    cx, cy, yaw, angles_body, ranges,
                    &field, w, h, x_min, y_min, cell, cfg.clamp_m);
                if n < cfg.min_beams_used { continue; }
                let mean = sum / (n as f32);
                if mean < best.mean_residual_m {
                    best = RelocalizeResult {
                        pose: (cx, cy, yaw),
                        mean_residual_m: mean,
                        n_beams_used: n,
                        accepted: false,
                    };
                }
            }
        }
    }

    best.accepted = best.mean_residual_m <= cfg.max_mean_residual_m
        && best.n_beams_used >= cfg.min_beams_used;
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{GridConfig, OccupancyGrid};

    fn make_test_room() -> OccupancyGrid {
        // 4×4 m square room, walls at ±2 m, room is "L"-shaped to break
        // 4-fold symmetry: a chunk is missing in the +x/+y quadrant so
        // the relocalize candidate can be unambiguous.
        let mut g = OccupancyGrid::new(GridConfig {
            x_range: (-2.5, 2.5), y_range: (-2.5, 2.5), cell: 0.05,
        });
        // Outer walls.
        let n = 200;
        for i in 0..n {
            let t = -2.0 + 4.0 * (i as f32 / (n - 1) as f32);
            g.integrate_ray(0.0, 0.0, t,  2.0, true);
            g.integrate_ray(0.0, 0.0, t, -2.0, true);
            g.integrate_ray(0.0, 0.0,  2.0, t, true);
            g.integrate_ray(0.0, 0.0, -2.0, t, true);
        }
        // Asymmetric divider.
        for i in 0..n / 2 {
            let t = 0.0 + 2.0 * (i as f32 / ((n / 2) as f32));
            g.integrate_ray(0.0, 0.0, t, 0.5, true);
        }
        g
    }

    #[test]
    fn relocalize_finds_known_pose() {
        let mut grid = make_test_room();
        // Simulated scan: at ground truth (0.5, -0.5, 0.0) cast 36 beams
        // and use the grid raycasts as ranges.
        let truth = (0.5_f32, -0.5_f32, 0.0_f32);
        let n_beams = 36;
        let mut angles = Vec::new();
        let mut ranges = Vec::new();
        for k in 0..n_beams {
            // ±90° fan in body frame.
            let a = -std::f32::consts::FRAC_PI_2
                + (k as f32 / (n_beams - 1) as f32) * std::f32::consts::PI;
            let r = grid.cast_ray(truth.0, truth.1, truth.2 + a, 4.0);
            if r > 0.0 {
                angles.push(a);
                ranges.push(r);
            }
        }
        let cfg = RelocalizeConfig::default();
        let res = relocalize_against_grid(&mut grid, &angles, &ranges, &cfg)
            .expect("relocalize returns a candidate");
        assert!(res.accepted,
                "expected accepted, got mean_residual={:.3} m, n={}",
                res.mean_residual_m, res.n_beams_used);
        // Should land within a couple of cells of ground truth and within
        // a few degrees of the right yaw.
        let dx = res.pose.0 - truth.0;
        let dy = res.pose.1 - truth.1;
        let dist = (dx * dx + dy * dy).sqrt();
        assert!(dist < 0.20,
                "position {:.2} m off truth ({:?} vs {:?})",
                dist, res.pose, truth);
        let dyaw = (res.pose.2 - truth.2).abs();
        assert!(dyaw < 0.20 || (2.0 * std::f32::consts::PI - dyaw) < 0.20,
                "yaw {:.1}° off truth", dyaw.to_degrees());
    }
}
