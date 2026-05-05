//! PoseGraph — a SE(2) pose graph over submap anchor poses. Phase 5.
//!
//! Nodes: one per submap, holding its current world pose.
//! Edges: relative-pose constraints with a 3×3 information matrix:
//!   * "odometry" edges between consecutive submaps, weight from how
//!     trustworthy the inter-submap odom delta is,
//!   * "loop" edges added by `loop_closer` when a new submap aligns
//!     against an older one via scan matching.

#![allow(dead_code)]
