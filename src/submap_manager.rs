//! SubmapManager — owns the list of frozen submaps and the current
//! one. Decides when to close a submap and start a fresh one. Phase 4.
//!
//! Switching policy (initial cut, will be tuned):
//!   * close + open new submap when the duck has been mapping in the
//!     current one for ≥ 20 s,
//!   * OR when it has accumulated ≥ 2 m of in-submap travel.

#![allow(dead_code)]
