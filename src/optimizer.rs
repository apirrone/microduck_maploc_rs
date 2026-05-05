//! Sparse Gauss-Newton optimizer for the SE(2) pose graph. Phase 5.
//!
//! Anchors node 0 (first submap) and adjusts the rest to minimize the
//! sum of squared edge residuals weighted by their information matrices.
//! For our small graphs (tens of nodes), a direct dense or simple sparse
//! Cholesky is plenty.

#![allow(dead_code)]
