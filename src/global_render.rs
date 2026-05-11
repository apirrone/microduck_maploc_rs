//! Composite all submap grids into a single global occupancy grid.
//!
//! Each submap is a small local grid anchored at its world pose. To get
//! a global view we:
//!
//!   1. Compute a world-frame bounding box covering every submap's grid
//!      corners (after applying the submap's anchor pose).
//!   2. Allocate a fresh `OccupancyGrid` covering that bbox + a margin.
//!   3. For each submap, walk its non-zero cells; for each cell, transform
//!      its centre to world coordinates via the anchor pose and add the
//!      cell's log-odds to the corresponding global cell. Sums are
//!      clamped to `[LO_MIN, LO_MAX]` by `add_log_odds_at_world`.
//!
//! Phase 4 ships a naive O(total cells) implementation. That's perfectly
//! fine at our scales (a 4 m × 4 m × 5 cm submap is 6400 cells; 50
//! submaps = 320 k cells = a few ms on a Pi).

use crate::grid::{GridConfig, OccupancyGrid};
use crate::submap::Submap;

#[derive(Debug, Clone, Copy)]
pub struct GlobalRenderConfig {
    /// Cell size of the rendered global grid.
    pub cell_m: f32,
    /// Padding around the union bbox, metres.
    pub margin_m: f32,
    /// Per-submap cells are only composited when their |log-odds| is
    /// above this threshold. 0 = include every barely-positive cell
    /// (legacy, fuzzy walls). ~150 hides one-off ToF flickers without
    /// erasing genuine walls.
    pub min_hit_threshold_fp: i16,
}

impl Default for GlobalRenderConfig {
    fn default() -> Self {
        Self { cell_m: 0.05, margin_m: 0.5, min_hit_threshold_fp: 150 }
    }
}

/// Render the union of all submaps into a fresh global grid. Returns
/// `None` when given no submaps (no bbox to render). When all submaps
/// have empty grids, the resulting global grid is just an empty grid
/// covering their anchors.
pub fn render_global<'a>(
    submaps: impl IntoIterator<Item = &'a Submap>,
    cfg: &GlobalRenderConfig,
) -> Option<OccupancyGrid> {
    let submaps: Vec<&Submap> = submaps.into_iter().collect();
    if submaps.is_empty() { return None; }

    // Step 1 — bbox over all submap grid corners (in world frame).
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for s in &submaps {
        let cfg_l = s.grid().cfg();
        let (ax, ay, ayaw) = s.anchor_pose();
        let ca = ayaw.cos();
        let sa = ayaw.sin();
        for (xl, yl) in [
            (cfg_l.x_range.0, cfg_l.y_range.0),
            (cfg_l.x_range.0, cfg_l.y_range.1),
            (cfg_l.x_range.1, cfg_l.y_range.0),
            (cfg_l.x_range.1, cfg_l.y_range.1),
        ] {
            let xw = ax + ca * xl - sa * yl;
            let yw = ay + sa * xl + ca * yl;
            if xw < min_x { min_x = xw; }
            if xw > max_x { max_x = xw; }
            if yw < min_y { min_y = yw; }
            if yw > max_y { max_y = yw; }
        }
    }
    let m = cfg.margin_m;
    let global_cfg = GridConfig {
        x_range: (min_x - m, max_x + m),
        y_range: (min_y - m, max_y + m),
        cell:    cfg.cell_m,
    };
    let mut global = OccupancyGrid::new(global_cfg);

    // Step 2 — for each submap, transform each non-zero local cell
    // into world coords and add its log-odds to the global cell.
    for s in &submaps {
        let cfg_l = s.grid().cfg();
        let cell_l = cfg_l.cell;
        let lw = s.grid().width();
        let lh = s.grid().height();
        let log = s.grid().log_raw();
        let (ax, ay, ayaw) = s.anchor_pose();
        let ca = ayaw.cos();
        let sa = ayaw.sin();
        for i in 0..lh {
            for j in 0..lw {
                let v = log[i * lw + j];
                if v == 0 { continue; }
                // Skip per-submap cells whose |log-odds| is below the
                // configured threshold — keeps the global render from
                // accumulating sub-threshold noise into a wall once it
                // crosses zero in aggregate.
                if v.unsigned_abs() < cfg.min_hit_threshold_fp.unsigned_abs() {
                    continue;
                }
                let xl = cfg_l.x_range.0 + (j as f32 + 0.5) * cell_l;
                let yl = cfg_l.y_range.0 + (i as f32 + 0.5) * cell_l;
                let xw = ax + ca * xl - sa * yl;
                let yw = ay + sa * xl + ca * yl;
                global.add_log_odds_at_world(xw, yw, v);
            }
        }
    }
    Some(global)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::GridConfig;
    use crate::submap::Submap;

    #[test]
    fn render_two_overlapping_submaps_combines_their_marks() {
        let cfg = GridConfig {
            x_range: (-1.0, 1.0),
            y_range: (-1.0, 1.0),
            cell:    0.05,
        };
        // Submap A at (0, 0, 0): mark a wall at world (0.5, 0). Repeat
        // the same beam so the cell crosses the render confidence
        // threshold (a single hit would otherwise be filtered as noise).
        let mut a = Submap::new_at((0.0, 0.0, 0.0), cfg);
        for _ in 0..5 { a.integrate_scan((0.0, 0.0, 0.0), &[0.0], &[0.5]); }
        // Submap B anchored at (0.5, 0, 0): same world wall is at LOCAL (0, 0).
        let mut b = Submap::new_at((0.5, 0.0, 0.0), cfg);
        for _ in 0..5 { b.integrate_scan((0.5, 0.0, 0.0), &[0.0], &[0.05]); }

        let g = render_global([&a, &b], &GlobalRenderConfig::default()).unwrap();
        // World cell ~ (0.5, 0) should be marked occupied (positive log-odds)
        // — both submaps contributed there.
        let (i, j) = g.world_to_idx(0.5, 0.0).expect("inside global bbox");
        assert!(g.log_at(i, j) > 0,
                "world (0.5, 0) wasn't marked occupied (log = {})",
                g.log_at(i, j));
    }
}
