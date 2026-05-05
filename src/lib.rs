//! microduck_maploc — v2 (submap-based pose-graph SLAM).
//!
//! See `docs/PLAN.md` and `docs/DESIGN.md` for the intended architecture.
//!
//! Phase status — bullets get checked as the corresponding modules ship:
//!
//!   [x] grid           — 2D log-odds occupancy + distance field
//!   [x] scan_matcher   — Hector-style ICP against a target grid
//!   [ ] submap         — local grid + anchor pose
//!   [ ] submap_manager — open / close submaps based on time + travel
//!   [ ] pose_graph     — SE(2) nodes + edges + sqrt-info
//!   [ ] optimizer      — sparse Gauss-Newton on SE(2)
//!   [ ] loop_closer    — submap-to-submap scan match for loop edges
//!   [ ] global_render  — composite all submaps into a single grid
//!   [ ] replay         — read back .mdlg session files for offline iteration
//!
//! v1's MCL + scan accumulator were band-aids that have been removed; MCL
//! will return at Phase 7 scoped to relocalize-from-uniform on a saved map.

pub mod follower;
pub mod global_render;
pub mod grid;
pub mod loop_closer;
pub mod mount;
pub mod optimizer;
pub mod planner;
pub mod pose_graph;
pub mod replay;
pub mod scan_matcher;
pub mod stream;
pub mod submap;
pub mod submap_manager;
pub mod wire;

pub use follower::{follow_step, FollowCommand, FollowerState};
pub use grid::{GridConfig, OccupancyGrid};
pub use planner::{plan, PlannerConfig};
pub use scan_matcher::{match_scan, ScanMatchConfig, ScanMatchResult};
pub use wire::{Goal, LockState, Message, Path as WirePath, Pose, Scan};
