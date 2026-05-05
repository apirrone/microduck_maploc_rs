//! LoopCloser — detect submap-to-submap loop closures. Phase 5.
//!
//! On each submap close, search older submaps within a spatial radius
//! of the new anchor pose. For each candidate, scan-match the new
//! submap's first ~10 scans against the candidate's grid. If the
//! resulting residual is below threshold, emit a loop edge for the
//! pose graph.

#![allow(dead_code)]
