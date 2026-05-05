//! Composite all submap grids into a single global occupancy grid for
//! visualization and (eventually) navigation. Phase 4.
//!
//! For each submap, transform its local cells into world coordinates
//! using its anchor pose, then merge log-odds into the destination
//! global grid. Naive but fits well within budget at our scales.

#![allow(dead_code)]
