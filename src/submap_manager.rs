//! SubmapManager — owns the list of frozen submaps + the active one,
//! and decides when to switch.
//!
//! Switching policy (initial cut, tunable):
//!   * close + open new submap when the active one has been live for
//!     ≥ `max_age_s`,
//!   * OR when the tracked pose has moved ≥ `max_travel_m` from the
//!     submap's anchor.
//!
//! On switch the new submap's anchor pose is the *current tracked
//! pose at switch time*, so there's no positional discontinuity in
//! the rendered global map (frozen submaps and the new one share the
//! same world frame).

use crate::grid::GridConfig;
use crate::submap::{Pose2, Submap};

#[derive(Debug, Clone, Copy)]
pub struct SubmapManagerConfig {
    /// Submap dimensions (centred on the anchor; defines the local grid).
    pub grid: GridConfig,
    /// Max wall-time before forcing a switch.
    pub max_age_s: f32,
    /// Max in-submap travel before forcing a switch.
    pub max_travel_m: f32,
}

impl Default for SubmapManagerConfig {
    fn default() -> Self {
        // 4 m × 4 m local grid at 5 cm cells, centred on the anchor.
        let grid = GridConfig {
            x_range: (-2.0, 2.0),
            y_range: (-2.0, 2.0),
            cell:    0.05,
        };
        Self { grid, max_age_s: 20.0, max_travel_m: 2.0 }
    }
}

pub struct SubmapManager {
    cfg: SubmapManagerConfig,
    frozen: Vec<Submap>,
    current: Option<Submap>,
    /// Wall-time (session-relative seconds) at which the current
    /// submap was created. `None` while there is no current submap.
    current_started_s: Option<f32>,
}

impl SubmapManager {
    pub fn new(cfg: SubmapManagerConfig) -> Self {
        Self { cfg, frozen: Vec::new(), current: None, current_started_s: None }
    }

    pub fn config(&self) -> SubmapManagerConfig { self.cfg }

    /// Number of *frozen* submaps (excluding the active one).
    pub fn n_frozen(&self) -> usize { self.frozen.len() }

    /// Total submaps, including the active one if any.
    pub fn n_total(&self) -> usize {
        self.frozen.len() + if self.current.is_some() { 1 } else { 0 }
    }

    pub fn frozen(&self) -> &[Submap] { &self.frozen }
    pub fn current(&self) -> Option<&Submap> { self.current.as_ref() }
    pub fn current_mut(&mut self) -> Option<&mut Submap> { self.current.as_mut() }

    /// Iterate frozen + current as `&Submap` (cheap to call repeatedly).
    pub fn all(&self) -> impl Iterator<Item = &Submap> {
        self.frozen.iter().chain(self.current.iter())
    }

    /// Update the manager. Call on every tick; `tracked_pose` is the
    /// robot's current world pose (from odom-driven tracking), and
    /// `now_s` is the session-relative wall-time. Returns `true` iff a
    /// new submap was started this tick (useful for downstream loop
    /// closure / pose-graph hooks at submap-close).
    pub fn tick(&mut self, now_s: f32, tracked_pose: Pose2) -> bool {
        // Bootstrap the first submap on the very first tick.
        if self.current.is_none() {
            self.start_new(tracked_pose, now_s);
            return true;
        }
        if self.should_switch(now_s, tracked_pose) {
            // Move current → frozen, start fresh at the new anchor.
            let old = self.current.take().unwrap();
            self.frozen.push(old);
            self.start_new(tracked_pose, now_s);
            return true;
        }
        false
    }

    fn should_switch(&self, now_s: f32, tracked_pose: Pose2) -> bool {
        let cur = match &self.current {
            Some(c) => c,
            None => return false,
        };
        let age = now_s - self.current_started_s.unwrap_or(now_s);
        if age >= self.cfg.max_age_s {
            return true;
        }
        let (ax, ay, _) = cur.anchor_pose();
        let dx = tracked_pose.0 - ax;
        let dy = tracked_pose.1 - ay;
        let travel = (dx * dx + dy * dy).sqrt();
        travel >= self.cfg.max_travel_m
    }

    fn start_new(&mut self, anchor_pose: Pose2, now_s: f32) {
        self.current = Some(Submap::new_at(anchor_pose, self.cfg.grid));
        self.current_started_s = Some(now_s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_tick_creates_a_submap() {
        let mut mgr = SubmapManager::new(SubmapManagerConfig::default());
        assert_eq!(mgr.n_total(), 0);
        let opened = mgr.tick(0.0, (0.0, 0.0, 0.0));
        assert!(opened);
        assert_eq!(mgr.n_total(), 1);
        assert_eq!(mgr.n_frozen(), 0);
    }

    #[test]
    fn travel_triggers_switch() {
        let mut cfg = SubmapManagerConfig::default();
        cfg.max_travel_m = 1.0;
        cfg.max_age_s    = 1000.0;
        let mut mgr = SubmapManager::new(cfg);
        mgr.tick(0.0, (0.0, 0.0, 0.0));
        assert_eq!(mgr.n_frozen(), 0);
        mgr.tick(1.0, (0.5, 0.0, 0.0));
        assert_eq!(mgr.n_frozen(), 0, "0.5 m < threshold should not trigger");
        mgr.tick(2.0, (1.5, 0.0, 0.0));
        assert_eq!(mgr.n_frozen(), 1, "1.5 m ≥ threshold should switch");
        assert_eq!(mgr.n_total(), 2);
    }

    #[test]
    fn age_triggers_switch() {
        let mut cfg = SubmapManagerConfig::default();
        cfg.max_age_s    = 5.0;
        cfg.max_travel_m = 1000.0;
        let mut mgr = SubmapManager::new(cfg);
        mgr.tick(0.0, (0.0, 0.0, 0.0));
        mgr.tick(3.0, (0.0, 0.0, 0.0));
        assert_eq!(mgr.n_frozen(), 0);
        mgr.tick(6.0, (0.0, 0.0, 0.0));
        assert_eq!(mgr.n_frozen(), 1);
    }

    #[test]
    fn new_submap_anchor_equals_tracked_pose_at_switch() {
        let mut cfg = SubmapManagerConfig::default();
        cfg.max_travel_m = 0.5;
        cfg.max_age_s    = 1000.0;
        let mut mgr = SubmapManager::new(cfg);
        mgr.tick(0.0, (0.0, 0.0, 0.0));
        let switch_pose = (1.0, 0.5, 0.7);
        mgr.tick(1.0, switch_pose);
        let cur = mgr.current().unwrap();
        let a = cur.anchor_pose();
        assert!((a.0 - switch_pose.0).abs() < 1e-6);
        assert!((a.1 - switch_pose.1).abs() < 1e-6);
        assert!((a.2 - switch_pose.2).abs() < 1e-6);
    }
}
