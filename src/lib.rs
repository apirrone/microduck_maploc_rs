//! 2D ToF mapping + MCL localization for microduck.
//!
//! This crate provides the on-device pieces of the perception stack:
//!
//! * [`OccupancyGrid`] — log-odds 2D occupancy map, fixed-point so the
//!   inner loop is integer math and the whole map of an apartment fits
//!   in a few tens of kB.
//! * (Coming next) Monte Carlo Localization with augmented MCL kidnap
//!   recovery, ported from the Python reference in `microduck_maploc`.
//!
//! The save/load format is intentionally trivial (a tagged binary blob)
//! so a Pi-side runtime can persist the map between boots without
//! pulling in a serialization framework.

pub mod accumulator;
pub mod follower;
pub mod grid;
pub mod mcl;
pub mod planner;
pub mod scan_matcher;
pub mod stream;
pub mod wire;

pub use accumulator::{BufferedBeam, ScanAccumulator, AccumulatorConfig};
pub use follower::{follow_step, FollowCommand, FollowerState};
pub use grid::{GridConfig, OccupancyGrid};
pub use mcl::{FieldRelocConfig, Localizer, MclConfig, Particle};
pub use planner::{plan, PlannerConfig};
pub use scan_matcher::{match_scan, ScanMatchConfig, ScanMatchResult};
pub use wire::{Goal, LockState, Message, Path as WirePath, Pose, Scan};
