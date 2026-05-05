//! LoopCloser — detect submap-to-submap loop closures via scan matching.
//!
//! Triggered after a submap closes. For each older submap whose anchor
//! is within `radius_m` of the new submap's anchor (and not the
//! immediate predecessor — that's already captured by the odom edge),
//! pick a representative raw scan from the new submap, transform its
//! pose into the older submap's local frame, and run `match_scan` on
//! the older submap's grid. If the residual is below threshold and at
//! least `min_beams_used` beams contributed, emit a relative-pose
//! constraint suitable for adding to the pose graph as a loop edge.

use crate::pose_graph::{between, compose, inverse};
use crate::scan_matcher::{match_scan, ScanMatchConfig};
use crate::submap::{Pose2, Submap};

#[derive(Debug, Clone, Copy)]
pub struct LoopCloserConfig {
    /// Spatial proximity gate (metres): only consider older submaps
    /// within this radius of the new anchor.
    pub radius_m: f32,
    /// Don't try matching against this many submaps directly preceding
    /// the new one (their odom edges already constrain things and
    /// short-range loops are noisy).
    pub min_index_gap: usize,
    /// Per-beam RMS residual upper bound (metres) for accepting a
    /// match as a real loop closure.
    pub max_residual_m: f32,
    /// At least this many beams must have contributed to the match
    /// for it to be trustworthy.
    pub min_beams_used: u32,
    /// Scan matcher hyperparameters used for the actual match.
    pub sm: ScanMatchConfig,
    /// Per-axis sigmas for the loop edge's information matrix.
    pub edge_sigma_xy:  f32,
    pub edge_sigma_yaw: f32,
    /// If true, print one line per rejection (residual / beams) so we
    /// can debug "why did no loop close".
    pub verbose: bool,
}

impl Default for LoopCloserConfig {
    fn default() -> Self {
        let mut sm = ScanMatchConfig::default();
        sm.prior_sigma_xy  = 0.30;
        sm.prior_sigma_yaw = 0.20;
        Self {
            radius_m: 1.5,
            min_index_gap: 2,
            max_residual_m: 0.10,   // 10 cm — tighter would reject good closes
            min_beams_used: 16,
            sm,
            edge_sigma_xy:  0.05,
            edge_sigma_yaw: 0.03,
            verbose: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LoopClosure {
    pub from_idx: usize,
    pub to_idx:   usize,
    /// Relative pose from `from_idx`'s anchor to `to_idx`'s anchor as
    /// inferred by the scan match (i.e. the loop edge's measurement).
    pub measurement: Pose2,
    pub residual_m:  f32,
    pub n_beams_used: u32,
}

/// Try to detect loop closures for a freshly-closed submap.
/// `submaps[new_idx]` is the new submap. We scan its first stored raw
/// scan against each candidate older submap's grid.
pub fn detect_loops(
    submaps: &mut [Submap],
    new_idx: usize,
    cfg: &LoopCloserConfig,
) -> Vec<LoopClosure> {
    let n = submaps.len();
    if new_idx == 0 || new_idx >= n { return Vec::new(); }

    // Snapshot what we need from the new submap so the borrow is
    // released before we mutably touch older submaps' grids.
    let new_anchor = submaps[new_idx].anchor_pose();
    let scans = submaps[new_idx].raw_scans().to_vec();
    if scans.is_empty() { return Vec::new(); }

    let mut out = Vec::new();
    for older_idx in 0..n {
        if older_idx == new_idx { continue; }
        if older_idx + cfg.min_index_gap >= new_idx { continue; }

        // Spatial gate.
        let older_anchor = submaps[older_idx].anchor_pose();
        let dx = new_anchor.0 - older_anchor.0;
        let dy = new_anchor.1 - older_anchor.1;
        if (dx * dx + dy * dy).sqrt() > cfg.radius_m { continue; }

        // Pick the representative scan: middle of the buffer (more
        // likely to be away from submap-creation transients than the
        // very first scan).
        let scan = &scans[scans.len() / 2];
        let pose_world = compose(new_anchor, scan.pose_in_submap);
        let pose_in_older = between(older_anchor, pose_world);

        let result = match_scan(
            submaps[older_idx].grid_mut(),
            &scan.angles_body,
            &scan.ranges_horiz,
            pose_in_older,
            Some(pose_in_older),
            &cfg.sm,
        );
        if cfg.verbose {
            eprintln!("[loop-try] {} → {}  resid={:.3}  beams={}  iters={}",
                      older_idx, new_idx, result.residual_m,
                      result.n_beams_used, result.iterations);
        }
        if !result.residual_m.is_finite() { continue; }
        if result.residual_m > cfg.max_residual_m { continue; }
        if result.n_beams_used < cfg.min_beams_used { continue; }

        // Convert match.pose (in `older_idx`'s local frame) into a
        // corrected anchor for the new submap, then a relative pose
        // from older to new (the loop-edge measurement).
        let corrected_world = compose(older_anchor, result.pose);
        let corrected_new_anchor = compose(corrected_world, inverse(scan.pose_in_submap));
        let measurement = between(older_anchor, corrected_new_anchor);

        out.push(LoopClosure {
            from_idx: older_idx,
            to_idx:   new_idx,
            measurement,
            residual_m: result.residual_m,
            n_beams_used: result.n_beams_used,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::GridConfig;

    #[test]
    fn empty_or_single_submap_returns_no_loops() {
        let mut submaps = Vec::<Submap>::new();
        assert!(detect_loops(&mut submaps, 0, &LoopCloserConfig::default()).is_empty());
        let cfg = GridConfig::default();
        let mut s = vec![Submap::new_at((0.0, 0.0, 0.0), cfg)];
        assert!(detect_loops(&mut s, 0, &LoopCloserConfig::default()).is_empty());
    }
}
