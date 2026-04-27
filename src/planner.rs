//! A* path planning on the occupancy grid.
//!
//! Uses 8-connected neighbours with octile heuristic. Obstacles are
//! "inflated" by `radius_cells` so the path keeps the robot's body
//! clear of walls (we plan for a point but the duck is ~10 cm wide).
//!
//! `plan` returns world-frame waypoints, simplified by greedy
//! line-of-sight reduction so the duck doesn't get a noisy zig-zag from
//! the 8-connected grid expansion.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::grid::{OccupancyGrid, LOG_SCALE};

/// Octile distance × 1000 — fits comfortably in i32 for ~10 m grids.
const COST_STRAIGHT: i32 = 1000;
const COST_DIAGONAL: i32 = 1414;

#[derive(Debug, Clone, Copy)]
pub struct PlannerConfig {
    /// Inflate obstacles by this many cells. With 5 cm cells and a
    /// ~10 cm duck radius, 2 is the right floor; bump to 3 for a
    /// safety margin around real walls.
    pub inflate_cells: u8,
    /// Cap A* expansions so a hopeless query (e.g. goal in an enclosed
    /// region) fails quickly rather than burning the runtime.
    pub max_expansions: usize,
    /// Require log_odds above this for a cell to count as a planning
    /// obstacle. A single ToF hit saturates a stray cell to ~+0.85 ≈ 85
    /// in fixed-point — without a threshold, those cells get inflated and
    /// can close doorways. ~150 (= 1.5) demands ~2 confirmed hits.
    pub occ_threshold_fp: i16,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            inflate_cells: 3,
            max_expansions: 100_000,
            occ_threshold_fp: (1.5 * LOG_SCALE) as i16,
        }
    }
}

/// Plan a path from `(sx, sy)` to `(gx, gy)` in world coords. Returns
/// `None` if no path exists or the goal is in an obstacle. The first
/// waypoint is the start; the last is the goal (or as close as we can
/// get if the goal cell itself is occupied — we snap to the nearest
/// free cell within `inflate_cells * 2`).
pub fn plan(
    grid: &OccupancyGrid,
    start: (f32, f32),
    goal:  (f32, f32),
    cfg:   PlannerConfig,
) -> Option<Vec<(f32, f32)>> {
    // Build inflated obstacle mask once per query (cheap on apartment-
    // sized grids). Could cache by version if we ever feel the cost.
    let blocked = inflated_obstacles(grid, cfg.inflate_cells as i32, cfg.occ_threshold_fp);
    let w = grid.width(); let h = grid.height();

    let s = grid.world_to_idx(start.0, start.1)?;
    let g = match grid.world_to_idx(goal.0, goal.1) {
        Some(g) => g,
        None => return None,
    };

    // If the start is blocked (probably duck radius vs wall), fall
    // back to nearest free cell — otherwise A* never explores.
    let s = nearest_free(&blocked, w, h, s, cfg.inflate_cells as i32 * 2)?;
    let g = nearest_free(&blocked, w, h, g, cfg.inflate_cells as i32 * 2)?;

    let path = astar(&blocked, w, h, s, g, cfg.max_expansions)?;
    let smoothed = simplify_path(&blocked, w, h, &path);
    let cell = grid.cell();
    let x0 = grid.cfg().x_range.0;
    let y0 = grid.cfg().y_range.0;
    Some(smoothed.into_iter().map(|(i, j)| {
        (x0 + (j as f32 + 0.5) * cell, y0 + (i as f32 + 0.5) * cell)
    }).collect())
}

fn inflated_obstacles(grid: &OccupancyGrid, inflate: i32,
                      occ_threshold_fp: i16) -> Vec<bool> {
    let w = grid.width(); let h = grid.height();
    let mut blocked = vec![false; w * h];
    for i in 0..h {
        for j in 0..w {
            if grid.log_at(i, j) > occ_threshold_fp {
                for di in -inflate..=inflate {
                    for dj in -inflate..=inflate {
                        let ii = i as i32 + di; let jj = j as i32 + dj;
                        if ii < 0 || jj < 0 || ii >= h as i32 || jj >= w as i32 { continue; }
                        // Stay roughly circular — drop corners outside the radius.
                        if di * di + dj * dj > inflate * inflate { continue; }
                        blocked[ii as usize * w + jj as usize] = true;
                    }
                }
            }
        }
    }
    blocked
}

fn nearest_free(blocked: &[bool], w: usize, h: usize,
                start: (usize, usize), max_radius: i32) -> Option<(usize, usize)> {
    let (i0, j0) = start;
    if !blocked[i0 * w + j0] { return Some(start); }
    for r in 1..=max_radius {
        for di in -r..=r {
            for dj in -r..=r {
                if di.abs() != r && dj.abs() != r { continue; } // ring only
                let ii = i0 as i32 + di; let jj = j0 as i32 + dj;
                if ii < 0 || jj < 0 || ii >= h as i32 || jj >= w as i32 { continue; }
                let (i, j) = (ii as usize, jj as usize);
                if !blocked[i * w + j] { return Some((i, j)); }
            }
        }
    }
    None
}

#[inline]
fn octile(a: (usize, usize), b: (usize, usize)) -> i32 {
    let di = (a.0 as i32 - b.0 as i32).abs();
    let dj = (a.1 as i32 - b.1 as i32).abs();
    let (lo, hi) = if di < dj { (di, dj) } else { (dj, di) };
    COST_STRAIGHT * (hi - lo) + COST_DIAGONAL * lo
}

fn astar(
    blocked: &[bool], w: usize, h: usize,
    start: (usize, usize), goal: (usize, usize),
    max_expansions: usize,
) -> Option<Vec<(usize, usize)>> {
    let n = w * h;
    let mut g_score = vec![i32::MAX; n];
    let mut came_from: Vec<u32> = vec![u32::MAX; n];
    let mut closed = vec![false; n];
    let mut heap: BinaryHeap<Reverse<(i32, u32)>> = BinaryHeap::new();

    let s_idx = (start.0 * w + start.1) as u32;
    let g_idx = (goal.0  * w + goal.1)  as u32;
    g_score[s_idx as usize] = 0;
    heap.push(Reverse((octile(start, goal), s_idx)));

    let neighbours: [(i32, i32, i32); 8] = [
        (-1,  0, COST_STRAIGHT), (1,  0, COST_STRAIGHT),
        ( 0, -1, COST_STRAIGHT), (0,  1, COST_STRAIGHT),
        (-1, -1, COST_DIAGONAL), (-1, 1, COST_DIAGONAL),
        ( 1, -1, COST_DIAGONAL), ( 1, 1, COST_DIAGONAL),
    ];
    let mut expansions = 0;
    while let Some(Reverse((_, cur))) = heap.pop() {
        if cur == g_idx {
            // Reconstruct.
            let mut path = Vec::new();
            let mut node = cur;
            loop {
                let i = (node as usize) / w;
                let j = (node as usize) % w;
                path.push((i, j));
                if node == s_idx { break; }
                let prev = came_from[node as usize];
                if prev == u32::MAX { return None; }
                node = prev;
            }
            path.reverse();
            return Some(path);
        }
        if closed[cur as usize] { continue; }
        closed[cur as usize] = true;
        expansions += 1;
        if expansions > max_expansions { return None; }

        let ci = (cur as usize) / w;
        let cj = (cur as usize) % w;
        let cur_g = g_score[cur as usize];
        for &(di, dj, step_cost) in &neighbours {
            let ni = ci as i32 + di; let nj = cj as i32 + dj;
            if ni < 0 || nj < 0 || ni >= h as i32 || nj >= w as i32 { continue; }
            let nidx = (ni as usize) * w + nj as usize;
            if blocked[nidx] || closed[nidx] { continue; }
            // Forbid diagonal moves that "squeeze through" two blocked corners.
            if di != 0 && dj != 0 {
                if blocked[ci * w + (nj as usize)] || blocked[(ni as usize) * w + cj] {
                    continue;
                }
            }
            let tentative = cur_g.saturating_add(step_cost);
            if tentative < g_score[nidx] {
                g_score[nidx] = tentative;
                came_from[nidx] = cur;
                let f = tentative.saturating_add(octile((ni as usize, nj as usize), goal));
                heap.push(Reverse((f, nidx as u32)));
            }
        }
    }
    None
}

/// Greedy line-of-sight simplification — walk forward as long as the
/// straight line stays free, then start a new segment. Cheap and good
/// enough; we don't need full shortcut smoothing for a small map.
fn simplify_path(blocked: &[bool], w: usize, h: usize,
                 path: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if path.len() < 3 { return path.to_vec(); }
    let mut out = Vec::with_capacity(path.len() / 2);
    out.push(path[0]);
    let mut anchor = 0usize;
    let mut last_visible = anchor + 1;
    while last_visible + 1 < path.len() {
        let next = last_visible + 1;
        if line_of_sight(blocked, w, h, path[anchor], path[next]) {
            last_visible = next;
        } else {
            out.push(path[last_visible]);
            anchor = last_visible;
            last_visible = anchor + 1;
        }
    }
    out.push(*path.last().unwrap());
    out
}

fn line_of_sight(blocked: &[bool], w: usize, h: usize,
                 a: (usize, usize), b: (usize, usize)) -> bool {
    let (mut ai, mut aj) = (a.0 as i32, a.1 as i32);
    let (bi, bj) = (b.0 as i32, b.1 as i32);
    let di = (bi - ai).abs(); let dj = (bj - aj).abs();
    let si = if ai < bi { 1 } else { -1 };
    let sj = if aj < bj { 1 } else { -1 };
    let mut err = di - dj;
    loop {
        if ai < 0 || aj < 0 || ai >= h as i32 || aj >= w as i32 { return false; }
        if blocked[ai as usize * w + aj as usize] { return false; }
        if ai == bi && aj == bj { return true; }
        let e2 = 2 * err;
        if e2 > -dj { err -= dj; ai += si; }
        if e2 <  di { err += di; aj += sj; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{GridConfig, OccupancyGrid};

    fn small_grid_with_wall() -> OccupancyGrid {
        // 1.5×1.5 m grid with 5 cm cells = 30×30. A vertical wall at
        // x=0 from y=-0.4 to +0.4, with a gap further north.
        let mut g = OccupancyGrid::new(GridConfig {
            x_range: (-0.75, 0.75), y_range: (-0.75, 0.75), cell: 0.05,
        });
        for y_i in 0..16 {
            let y = -0.40 + y_i as f32 * 0.05;
            // hammer the same cell so log-odds saturate above threshold
            for _ in 0..10 { g.integrate_ray(-0.7, y, 0.0, y, true); }
        }
        g
    }

    #[test]
    fn plans_around_wall() {
        let g = small_grid_with_wall();
        let path = plan(&g, (-0.6, 0.0), (0.6, 0.0),
                        PlannerConfig { inflate_cells: 1, max_expansions: 50_000,
                                        ..Default::default() })
            .expect("path should exist via north gap");
        assert!(path.len() >= 2);
        // Should detour north (positive y) before crossing.
        assert!(path.iter().any(|&(_, y)| y > 0.30),
                "expected detour around wall, got {path:?}");
    }

    #[test]
    fn no_path_returns_none() {
        // Goal outside the grid.
        let g = small_grid_with_wall();
        let p = plan(&g, (0.0, 0.0), (10.0, 10.0), PlannerConfig::default());
        assert!(p.is_none());
    }
}
