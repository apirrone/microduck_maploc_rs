//! Submap — local 2D occupancy grid + the world pose at which it was
//! anchored.
//!
//! All scans integrated into a submap are placed in *its local frame*:
//! local X/Y of (0, 0, 0) corresponds to the duck's pose at submap
//! creation. The global map is reconstructed by composing each submap's
//! local grid through its anchor pose (see `global_render`).
//!
//! For Phase 3 (single submap) we just create one Submap at world
//! origin and never close it. From Phase 4 onward `submap_manager`
//! owns the lifecycle.

use crate::grid::{GridConfig, OccupancyGrid};

/// Pose in SE(2): `(x, y, yaw)`.
pub type Pose2 = (f32, f32, f32);

/// One scan retained for loop-closure scan matching. Stored in the
/// submap's *local* frame (so it's independent of any later anchor
/// changes during pose-graph optimization).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RawScan {
    pub pose_in_submap: Pose2,
    pub angles_body:  Vec<f32>,
    pub ranges_horiz: Vec<f32>,
}

#[inline]
fn wrap_pi(a: f32) -> f32 {
    use std::f32::consts::PI;
    let two_pi = 2.0 * PI;
    let mut y = (a + PI).rem_euclid(two_pi) - PI;
    if y == PI { y = -PI; }
    y
}

/// SE(2) inverse-compose: returns the body pose expressed in the
/// anchor's frame. (x_local, y_local, yaw_local) such that composing
/// `anchor` with that local pose gives back `body_world`.
#[inline]
fn world_to_local(anchor: Pose2, body_world: Pose2) -> Pose2 {
    let (ax, ay, ayaw) = anchor;
    let (wx, wy, wyaw) = body_world;
    let dx = wx - ax;
    let dy = wy - ay;
    let ca = ayaw.cos();
    let sa = ayaw.sin();
    let xl =  ca * dx + sa * dy;
    let yl = -sa * dx + ca * dy;
    let yawl = wrap_pi(wyaw - ayaw);
    (xl, yl, yawl)
}

/// Number of raw scans retained per submap for loop-closure matching.
/// 10 covers a few seconds of capture at 15 Hz; way more than enough
/// to give the matcher signal but cheap (10 × 64 beams × 8 B = 5 KB).
pub const MAX_RAW_SCANS: usize = 10;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Submap {
    grid: OccupancyGrid,
    anchor_pose: Pose2,
    raw_scans: Vec<RawScan>,
}

impl Submap {
    /// Create a submap with its origin (local 0,0,0) at `anchor_pose` in
    /// the world frame. The grid is sized by `grid_cfg`.
    pub fn new_at(anchor_pose: Pose2, grid_cfg: GridConfig) -> Self {
        Self {
            grid: OccupancyGrid::new(grid_cfg),
            anchor_pose,
            raw_scans: Vec::with_capacity(MAX_RAW_SCANS),
        }
    }

    pub fn anchor_pose(&self) -> Pose2 { self.anchor_pose }
    pub fn grid(&self) -> &OccupancyGrid { &self.grid }
    pub fn grid_mut(&mut self) -> &mut OccupancyGrid { &mut self.grid }
    pub fn raw_scans(&self) -> &[RawScan] { &self.raw_scans }

    /// Update the anchor pose. Used by the pose-graph optimizer after
    /// loop closure: the submap's local content (grid + raw scans)
    /// stays untouched, only its anchor changes.
    pub fn set_anchor_pose(&mut self, new_anchor: Pose2) {
        self.anchor_pose = new_anchor;
    }

    /// Integrate one scan. `body_pose_world` is the duck's pose at scan
    /// capture (world frame). `angles_body` and `ranges_horiz` are the
    /// per-beam body-frame azimuths and horizontal-plane ranges (already
    /// floor-filtered upstream); NaN/zero ranges are skipped.
    pub fn integrate_scan(
        &mut self,
        body_pose_world: Pose2,
        angles_body: &[f32],
        ranges_horiz: &[f32],
    ) {
        debug_assert_eq!(angles_body.len(), ranges_horiz.len());
        let pose_local = world_to_local(self.anchor_pose, body_pose_world);
        let (ox, oy, oyaw) = pose_local;
        for (a, &r) in angles_body.iter().zip(ranges_horiz) {
            if !r.is_finite() || r <= 0.0 { continue; }
            let theta = oyaw + a;
            let hx = ox + r * theta.cos();
            let hy = oy + r * theta.sin();
            self.grid.integrate_ray(ox, oy, hx, hy, true);
        }
        // Retain the first `MAX_RAW_SCANS` for loop closure.
        if self.raw_scans.len() < MAX_RAW_SCANS {
            self.raw_scans.push(RawScan {
                pose_in_submap: pose_local,
                angles_body:  angles_body.to_vec(),
                ranges_horiz: ranges_horiz.to_vec(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_to_local_is_inverse_of_compose() {
        // pick a non-trivial anchor and a body pose
        let anchor = (1.0_f32, -0.5, 0.6);
        let body   = (2.5_f32,  0.3, 0.2);
        let local = world_to_local(anchor, body);
        // recompose: world = anchor ⊕ local
        let (ax, ay, ayaw) = anchor;
        let ca = ayaw.cos(); let sa = ayaw.sin();
        let wx = ax + ca * local.0 - sa * local.1;
        let wy = ay + sa * local.0 + ca * local.1;
        let wyaw = wrap_pi(ayaw + local.2);
        assert!((wx - body.0).abs() < 1e-5);
        assert!((wy - body.1).abs() < 1e-5);
        assert!((wyaw - body.2).abs() < 1e-5);
    }

    #[test]
    fn integrating_a_perpendicular_wall_lights_up_the_right_cell() {
        // 4 m square grid, 5 cm cells, anchor at world origin, no rotation.
        let cfg = GridConfig {
            x_range: (-2.0, 2.0),
            y_range: (-2.0, 2.0),
            cell:    0.05,
        };
        let mut s = Submap::new_at((0.0, 0.0, 0.0), cfg);
        // Body at world origin too. One beam pointing +x at 1 m.
        s.integrate_scan((0.0, 0.0, 0.0), &[0.0], &[1.0]);
        // Cell at world (1.0, 0.0) should be occupied (positive log-odds).
        let (i, j) = s.grid().world_to_idx(1.0, 0.0).unwrap();
        assert!(s.grid().log_at(i, j) > 0,
                "wall cell wasn't marked occupied (log_odds={})",
                s.grid().log_at(i, j));
        // Cell halfway along the ray should be free (negative log-odds).
        let (i, j) = s.grid().world_to_idx(0.5, 0.0).unwrap();
        assert!(s.grid().log_at(i, j) < 0,
                "free cell wasn't marked free (log_odds={})",
                s.grid().log_at(i, j));
    }

    #[test]
    fn integrating_at_offset_anchor_uses_local_frame() {
        let cfg = GridConfig {
            x_range: (-2.0, 2.0),
            y_range: (-2.0, 2.0),
            cell:    0.05,
        };
        // Anchor at world (5, 5, 0). Body at world (5, 5, 0) (same as anchor).
        // Scan should mark up the wall at LOCAL (1, 0) = WORLD (6, 5),
        // but the local grid's cell at LOCAL (1, 0) is what gets ink.
        let mut s = Submap::new_at((5.0, 5.0, 0.0), cfg);
        s.integrate_scan((5.0, 5.0, 0.0), &[0.0], &[1.0]);
        let (i, j) = s.grid().world_to_idx(1.0, 0.0).unwrap();
        assert!(s.grid().log_at(i, j) > 0);
    }
}
