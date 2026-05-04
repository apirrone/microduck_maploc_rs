//! Hector-style scan-to-map ICP.
//!
//! Gauss-Newton minimization of the sum of squared distances from each
//! beam endpoint to the nearest mapped obstacle. The grid's distance
//! field gives O(1) per-beam residual + bilinear gradient, so each
//! iteration is cheap and ~10 iterations converge.
//!
//! Why this exists alongside MCL: MCL is a particle filter — great for
//! relocalising in a *known* map (uniform cloud → scan likelihood pulls
//! it home), bad for tracking + mapping a *partial* map (high residuals
//! in unmapped regions trigger kidnap injection / global relocalize and
//! the filter snaps to wrong-but-also-consistent clusters). Scan
//! matching is the textbook tracker: bounded local search, no particle
//! cloud, no chance of teleporting. Use this for SLAM mode (build +
//! drift-correct), use MCL for relocalize-from-uniform mode.
//!
//! Cost: ~1–3 ms per scan on a Pi 4 with 64 valid beams; should run
//! comfortably on a Pi Zero 2 W.

use crate::grid::OccupancyGrid;

/// Hyperparameters for [`match_scan`].
#[derive(Debug, Clone, Copy)]
pub struct ScanMatchConfig {
    pub max_iters: u32,
    /// Convergence: stop when |Δ| drops below all three thresholds.
    pub eps_translation_m: f32,
    pub eps_rotation_rad:  f32,
    /// Levenberg damping added to the Hessian diagonal each step.
    /// 0 = pure Gauss-Newton.
    pub lambda: f32,
    /// Per-beam residual saturation. Beams whose endpoint is more than
    /// `sigma_m` from any wall contribute a capped distance instead of
    /// blowing up the cost — keeps unmapped regions from dominating.
    pub sigma_m: f32,
    /// Occupancy threshold (fixed-point log-odds) used by
    /// `OccupancyGrid::distance_field`. Matches the global-relocalize
    /// default.
    pub occ_threshold_fp: i16,
    /// Optional Gaussian regularizer pulling the optimized pose toward
    /// `prior_pose` (typically the odometry-predicted pose). 0 = off.
    /// Stops the matcher from wandering when there's not enough scan
    /// signal to constrain all 3 DoF (e.g. looking at a long corridor).
    pub prior_sigma_xy:  f32,
    pub prior_sigma_yaw: f32,
}

impl Default for ScanMatchConfig {
    fn default() -> Self {
        Self {
            max_iters: 12,
            eps_translation_m: 1e-3,
            eps_rotation_rad:  1e-3,
            lambda:            1e-3,
            sigma_m:           0.30,
            occ_threshold_fp:  150,
            prior_sigma_xy:    0.50,
            prior_sigma_yaw:   0.50,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScanMatchResult {
    pub pose:        (f32, f32, f32),
    /// Mean per-beam residual in metres (sqrt of average squared distance).
    pub residual_m:  f32,
    pub iterations:  u32,
    pub converged:   bool,
    /// Number of beams that contributed (NaN/zero ranges skipped).
    pub n_beams_used: u32,
}

/// Scan-to-map ICP. Iteratively shifts `initial_pose` so beam endpoints
/// land on mapped obstacles, minimizing sum of squared distances from
/// each endpoint to its nearest wall (looked up via the grid's distance
/// field with bilinear interpolation).
///
/// `angles_body` and `ranges` are parallel — one entry per beam, in the
/// body frame, horizontal-plane projected. Non-finite or non-positive
/// ranges are skipped.
///
/// `prior_pose` (if `Some`) is a Gaussian anchor pulling the result
/// toward that pose with the configured sigmas. Pass the odometry-
/// predicted pose to keep the matcher honest when scan signal is weak.
pub fn match_scan(
    grid: &mut OccupancyGrid,
    angles_body: &[f32],
    ranges:      &[f32],
    initial_pose: (f32, f32, f32),
    prior_pose:   Option<(f32, f32, f32)>,
    cfg: &ScanMatchConfig,
) -> ScanMatchResult {
    debug_assert_eq!(angles_body.len(), ranges.len());

    // Snapshot grid metadata before we (mutably) touch the field.
    let cell    = grid.cell();
    let cell_inv = 1.0 / cell;
    let cfg_g   = grid.cfg().clone();
    let h       = grid.height();
    let w       = grid.width();
    let field   = grid.distance_field(cfg.occ_threshold_fp);

    // Bilinear-interpolated distance + gradient at world point.
    // Returns (d, ∂d/∂x, ∂d/∂y). Out-of-bounds: saturate to sigma, no gradient.
    let sample = |fx: f32, fy: f32| -> (f32, f32, f32) {
        let cx = (fx - cfg_g.x_range.0) * cell_inv - 0.5;
        let cy = (fy - cfg_g.y_range.0) * cell_inv - 0.5;
        let i0 = cy.floor() as i32;
        let j0 = cx.floor() as i32;
        if i0 < 0 || j0 < 0
            || (i0 + 1) as usize >= h
            || (j0 + 1) as usize >= w
        {
            return (cfg.sigma_m, 0.0, 0.0);
        }
        let i0u = i0 as usize;
        let j0u = j0 as usize;
        let fx_frac = cx - j0 as f32;
        let fy_frac = cy - i0 as f32;
        let d00 = field[ i0u      * w + j0u    ];
        let d01 = field[ i0u      * w + j0u + 1];
        let d10 = field[(i0u + 1) * w + j0u    ];
        let d11 = field[(i0u + 1) * w + j0u + 1];
        let d = (1.0 - fx_frac) * (1.0 - fy_frac) * d00
              +        fx_frac  * (1.0 - fy_frac) * d01
              + (1.0 - fx_frac) *        fy_frac  * d10
              +        fx_frac  *        fy_frac  * d11;
        let dd_dx = ((d01 - d00) * (1.0 - fy_frac) + (d11 - d10) * fy_frac) * cell_inv;
        let dd_dy = ((d10 - d00) * (1.0 - fx_frac) + (d11 - d01) * fx_frac) * cell_inv;
        (d, dd_dx, dd_dy)
    };

    let (mut x, mut y, mut yaw) = initial_pose;
    let mut last_residual_m = f32::INFINITY;
    let mut iterations = 0u32;
    let mut converged  = false;
    let mut last_n_used = 0u32;

    for iter in 0..cfg.max_iters {
        iterations = iter + 1;
        // 3x3 Hessian + 3x1 gradient (J^T·J  and  J^T·r).
        let mut hm = [[0.0_f32; 3]; 3];
        let mut gv = [0.0_f32; 3];
        let mut residual_sum_sq = 0.0_f32;
        let mut n_used = 0u32;

        for (a_b, &r) in angles_body.iter().zip(ranges) {
            if !r.is_finite() || r <= 0.0 { continue; }
            let theta = yaw + a_b;
            let (sin_t, cos_t) = theta.sin_cos();
            let ex = x + r * cos_t;
            let ey = y + r * sin_t;
            let (d, dd_dx, dd_dy) = sample(ex, ey);
            // Saturate so one out-of-map beam doesn't dominate.
            let d_sat = d.min(cfg.sigma_m);
            residual_sum_sq += d_sat * d_sat;
            n_used += 1;
            // Jacobian of d w.r.t. (x, y, yaw):
            //   ∂d/∂x   = ∂d/∂ex · ∂ex/∂x   = ∂d/∂ex
            //   ∂d/∂y   = ∂d/∂ey · ∂ey/∂y   = ∂d/∂ey
            //   ∂d/∂yaw = ∂d/∂ex · (-r·sin θ) + ∂d/∂ey · (r·cos θ)
            let j0 = dd_dx;
            let j1 = dd_dy;
            let j2 = dd_dx * (-r * sin_t) + dd_dy * (r * cos_t);
            // Symmetric outer-product accumulation.
            let js = [j0, j1, j2];
            for a in 0..3 {
                for b in 0..3 {
                    hm[a][b] += js[a] * js[b];
                }
                gv[a] += js[a] * d_sat;
            }
        }
        if n_used == 0 { break; }
        last_n_used = n_used;

        // Gaussian pose-prior regularizer: adds (1/σ²) * Δpose² to the
        // cost, which contributes (1/σ²) on the diagonal and (1/σ²)*Δ
        // to the gradient.
        if let Some((px, py, pyaw)) = prior_pose {
            if cfg.prior_sigma_xy > 0.0 {
                let inv2 = 1.0 / (cfg.prior_sigma_xy * cfg.prior_sigma_xy);
                let dx = x - px;
                let dy = y - py;
                hm[0][0] += inv2;
                hm[1][1] += inv2;
                gv[0]    += inv2 * dx;
                gv[1]    += inv2 * dy;
                residual_sum_sq += inv2 * (dx * dx + dy * dy);
            }
            if cfg.prior_sigma_yaw > 0.0 {
                let inv2 = 1.0 / (cfg.prior_sigma_yaw * cfg.prior_sigma_yaw);
                let dyaw = wrap_pi(yaw - pyaw);
                hm[2][2] += inv2;
                gv[2]    += inv2 * dyaw;
                residual_sum_sq += inv2 * dyaw * dyaw;
            }
        }

        last_residual_m = (residual_sum_sq / n_used.max(1) as f32).sqrt();

        // Levenberg damping.
        hm[0][0] += cfg.lambda;
        hm[1][1] += cfg.lambda;
        hm[2][2] += cfg.lambda;

        let delta = match solve_3x3(&hm, &gv) {
            Some(d) => [-d[0], -d[1], -d[2]],
            None    => break,  // singular Hessian, give up
        };
        x   += delta[0];
        y   += delta[1];
        yaw  = wrap_pi(yaw + delta[2]);

        if delta[0].abs() < cfg.eps_translation_m
           && delta[1].abs() < cfg.eps_translation_m
           && delta[2].abs() < cfg.eps_rotation_rad
        {
            converged = true;
            break;
        }
    }

    ScanMatchResult {
        pose: (x, y, yaw),
        residual_m: last_residual_m,
        iterations,
        converged,
        n_beams_used: last_n_used,
    }
}

fn solve_3x3(a: &[[f32; 3]; 3], b: &[f32; 3]) -> Option<[f32; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-9 { return None; }
    let inv_det = 1.0 / det;
    let inv = [
        [(a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det,
         (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det,
         (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det],
        [(a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det,
         (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det,
         (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det],
        [(a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det,
         (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det,
         (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det],
    ];
    Some([
        inv[0][0] * b[0] + inv[0][1] * b[1] + inv[0][2] * b[2],
        inv[1][0] * b[0] + inv[1][1] * b[1] + inv[1][2] * b[2],
        inv[2][0] * b[0] + inv[2][1] * b[1] + inv[2][2] * b[2],
    ])
}

fn wrap_pi(a: f32) -> f32 {
    use std::f32::consts::PI;
    let two_pi = 2.0 * PI;
    let mut y = (a + PI).rem_euclid(two_pi) - PI;
    if y == PI { y = -PI; }
    y
}
