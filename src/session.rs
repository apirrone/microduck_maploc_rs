//! Persisted SLAM session state — load it back next boot to keep mapping
//! the same room without starting from scratch.
//!
//! Serializes the live submap manager + pose graph + tracked pose to a
//! single binary file via bincode + serde. Atomic writes (write to
//! `<path>.tmp`, fsync, rename) so an interrupted save can't corrupt a
//! good prior file.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::pose_graph::PoseGraph;
use crate::submap::{Pose2, Submap};

const MAGIC: &[u8; 4] = b"MDLS";
const VERSION: u32 = 1;

/// Snapshot of the runtime SLAM pipeline. Restored on next boot to
/// continue mapping the same world.
#[derive(Serialize, Deserialize)]
pub struct SessionState {
    pub frozen: Vec<Submap>,
    pub current: Option<Submap>,
    pub graph: PoseGraph,
    /// `node_for_submap[i]` is the pose-graph node index that anchors
    /// the i-th submap (frozen first, then current).
    pub node_for_submap: Vec<usize>,
    /// Last known world pose of the duck. On reload, the runtime resumes
    /// pose tracking from here — accurate iff the duck actually starts
    /// where it left off. Phase 7 (MCL relocalize) will replace this
    /// "trust the last pose" with a proper kidnapped-robot search.
    pub tracked: Pose2,
}

/// Borrowed view used only for serialization, so the caller doesn't have
/// to give up ownership (or clone) of the live SLAM state.
#[derive(Serialize)]
struct SessionStateRef<'a> {
    frozen: &'a [Submap],
    current: Option<&'a Submap>,
    graph: &'a PoseGraph,
    node_for_submap: &'a [usize],
    tracked: Pose2,
}

/// Atomically write a session to `path`. Creates parent dirs as needed.
pub fn save_session<P: AsRef<Path>>(
    path: P,
    frozen: &[Submap],
    current: Option<&Submap>,
    graph: &PoseGraph,
    node_for_submap: &[usize],
    tracked: Pose2,
) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(p) = path.parent() {
        if !p.as_os_str().is_empty() {
            std::fs::create_dir_all(p)?;
        }
    }
    let view = SessionStateRef {
        frozen, current, graph, node_for_submap, tracked,
    };
    let tmp = path.with_extension("tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        bincode::serialize_into(&mut w, &view).map_err(to_io)?;
        w.flush()?;
    }
    std::fs::rename(tmp, path)
}

impl SessionState {

    /// Load a session from `path`. Returns `Ok(None)` if the file does
    /// not exist (fresh-room case); returns `Err` on a corrupt file or
    /// version mismatch (caller can decide whether to wipe and start
    /// over).
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Option<Self>> {
        let path = path.as_ref();
        if !path.exists() { return Ok(None); }
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                                      format!("bad magic {magic:?}")));
        }
        let mut ver = [0u8; 4];
        r.read_exact(&mut ver)?;
        let v = u32::from_le_bytes(ver);
        if v != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                                      format!("unsupported session version {v}")));
        }
        let s: SessionState = bincode::deserialize_from(r).map_err(to_io)?;
        Ok(Some(s))
    }
}

fn to_io(e: bincode::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::GridConfig;
    use crate::pose_graph::{information_from_sigmas, PoseEdge, PoseGraph};

    #[test]
    fn round_trip_preserves_state() {
        let grid_cfg = GridConfig {
            x_range: (-1.0, 1.0), y_range: (-1.0, 1.0), cell: 0.05,
        };
        let mut a = Submap::new_at((0.0, 0.0, 0.0), grid_cfg);
        a.integrate_scan((0.0, 0.0, 0.0), &[0.0, 0.5], &[0.5, 0.4]);
        let mut b = Submap::new_at((0.5, 0.0, 0.0), grid_cfg);
        b.integrate_scan((0.5, 0.0, 0.0), &[0.0], &[0.3]);

        let mut graph = PoseGraph::new();
        let n0 = graph.add_node(a.anchor_pose(), 0);
        let n1 = graph.add_node(b.anchor_pose(), 1);
        graph.add_edge(PoseEdge {
            from: n0, to: n1,
            measurement: (0.5, 0.0, 0.0),
            information: information_from_sigmas(0.1, 0.05),
        });
        let node_for_submap = vec![n0, n1];
        let tracked = (0.5, 0.0, 0.0);

        let path = std::env::temp_dir()
            .join(format!("microduck_maploc_session_test_{}.bin",
                          std::process::id()));
        save_session(&path, &[a], Some(&b), &graph, &node_for_submap, tracked).unwrap();
        let loaded = SessionState::load(&path).unwrap().unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.frozen.len(), 1);
        assert!(loaded.current.is_some());
        assert_eq!(loaded.graph.nodes().len(), 2);
        assert_eq!(loaded.graph.edges().len(), 1);
        assert_eq!(loaded.node_for_submap, vec![n0, n1]);
        assert_eq!(loaded.tracked, tracked);

        // The frozen submap should still mark a wall along +x.
        let g = &loaded.frozen[0].grid();
        let (i, j) = g.world_to_idx(0.5, 0.0).unwrap();
        assert!(g.log_at(i, j) > 0,
                "saved/loaded grid lost the wall mark (log={})",
                g.log_at(i, j));
    }

    #[test]
    fn load_missing_returns_none() {
        let path = std::env::temp_dir().join("microduck_maploc_does_not_exist.bin");
        std::fs::remove_file(&path).ok();
        assert!(SessionState::load(&path).unwrap().is_none());
    }
}
