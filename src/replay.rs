//! Offline reader for `.mdlg` session logs.
//!
//! Mirrors `tools/replay_session.py` but skips the TCP indirection — feeds
//! decoded ToF frames and digital-twin packets directly into the v2 dev
//! loop:
//!
//! ```ignore
//! use microduck_maploc::replay::{SessionReplayer, Record};
//! for record in SessionReplayer::open("session.mdlg")? {
//!     match record? {
//!         Record::Tof(t)  => /* t.ranges_m, t.status, t.ts_us */ ,
//!         Record::Twin(d) => /* d.odom_x, d.odom_yaw, d.quat_wxyz, ... */ ,
//!     }
//! }
//! ```
//!
//! Wire format (little-endian, kept stable):
//!
//! ```text
//! header (16 B):
//!   magic         : 4 bytes  "MDLG"
//!   version       : u32      currently 1
//!   epoch_unix_ms : u64      capture start (ms since Unix epoch)
//!
//! record (until EOF):
//!   ts_us         : u64      microseconds since recorder start
//!   stream_id     : u8       0 = ToF, 1 = digital twin
//!   size          : u32      payload size
//!   payload       : u8[size] verbatim TCP wire bytes
//! ```
//!
//! The ToF payload format is the one emitted by `tof_streamer.py`:
//!
//! ```text
//! f64 ts_sender_s | u8 rows | u8 cols | u8[2] reserved
//!                 | f32[rows*cols] ranges_m  (NaN = invalid)
//!                 | u8 [rows*cols] target_status
//! ```
//!
//! The digital-twin payload is the 172 B packet documented at the top of
//! `microduck_runtime/fk/viewer.py`.

use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read};
use std::path::Path;

const MAGIC: &[u8; 4] = b"MDLG";
const VERSION: u32 = 1;

const STREAM_TOF:  u8 = 0;
const STREAM_TWIN: u8 = 1;

const TOF_ROWS: usize = 8;
const TOF_COLS: usize = 8;

const TWIN_PACKET_SIZE: usize = 8 + 41 * 4; // 172 B

#[derive(Debug, Clone)]
pub struct TofRecord {
    /// Recorder-side timestamp (µs since session start).
    pub ts_us: u64,
    /// Sender (Pi monotonic) timestamp from the payload header.
    pub sender_ts_s: f64,
    /// Per-zone slant ranges, metres. NaN means the chip flagged it
    /// invalid (status code outside the valid set).
    pub ranges_m: [[f32; TOF_COLS]; TOF_ROWS],
    /// Raw VL53L5CX target_status per zone. 5 / 6 = valid; everything
    /// else = various failure modes. The streamer already NaNs the
    /// corresponding `ranges_m`, but we surface the byte too in case
    /// downstream tooling wants to slice differently.
    pub status: [[u8; TOF_COLS]; TOF_ROWS],
}

#[derive(Debug, Clone, Copy)]
pub struct TwinRecord {
    /// Recorder-side timestamp (µs since session start).
    pub ts_us: u64,
    /// Sender timestamp from the packet header (runtime monotonic).
    pub sender_ts_s: f64,
    /// IMU quaternion `[w, x, y, z]`, body→world.
    pub quat_wxyz: [f32; 4],
    /// 15 joint positions, runtime motor order.
    pub joints: [f32; 15],
    /// 15 motor currents, mA.
    pub motor_currents_ma: [f32; 15],
    pub odom_x: f32,
    pub odom_y: f32,
    /// Ball world XYZ; any component may be NaN if not detected.
    pub ball_xyz: [f32; 3],
    pub odom_z: f32,
    pub odom_yaw: f32,
}

#[derive(Debug, Clone)]
pub enum Record {
    Tof(TofRecord),
    Twin(TwinRecord),
}

impl Record {
    pub fn ts_us(&self) -> u64 {
        match self {
            Record::Tof(t)  => t.ts_us,
            Record::Twin(t) => t.ts_us,
        }
    }
}

pub struct SessionReplayer {
    reader: BufReader<File>,
    epoch_unix_ms: u64,
}

impl SessionReplayer {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("not an mdlg file (magic = {:?})", magic),
            ));
        }
        let version = read_u32(&mut reader)?;
        if version != VERSION {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("unsupported mdlg version {} (this build expects {})",
                        version, VERSION),
            ));
        }
        let epoch_unix_ms = read_u64(&mut reader)?;
        Ok(Self { reader, epoch_unix_ms })
    }

    pub fn epoch_unix_ms(&self) -> u64 {
        self.epoch_unix_ms
    }
}

impl Iterator for SessionReplayer {
    type Item = io::Result<Record>;

    fn next(&mut self) -> Option<Self::Item> {
        // Try to read the next record header. UnexpectedEof here means
        // end-of-stream, which is the normal way to stop iterating.
        let ts_us = match read_u64(&mut self.reader) {
            Ok(v) => v,
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => return None,
            Err(e) => return Some(Err(e)),
        };
        let stream_id = match read_u8(&mut self.reader) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let size = match read_u32(&mut self.reader) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let mut payload = vec![0u8; size as usize];
        if let Err(e) = self.reader.read_exact(&mut payload) {
            return Some(Err(e));
        }
        match stream_id {
            STREAM_TOF  => Some(decode_tof(ts_us, &payload).map(Record::Tof)),
            STREAM_TWIN => Some(decode_twin(ts_us, &payload).map(Record::Twin)),
            other => Some(Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("unknown stream_id {}", other),
            ))),
        }
    }
}

// ── Decoders ─────────────────────────────────────────────────────────────────

fn decode_tof(ts_us: u64, payload: &[u8]) -> io::Result<TofRecord> {
    if payload.len() < 12 {
        return Err(io::Error::new(ErrorKind::InvalidData,
            "ToF payload too short for header"));
    }
    let sender_ts_s = read_f64_le(&payload[0..8]);
    let rows = payload[8] as usize;
    let cols = payload[9] as usize;
    if rows != TOF_ROWS || cols != TOF_COLS {
        return Err(io::Error::new(ErrorKind::InvalidData,
            format!("expected 8x8 ToF, got {}x{}", rows, cols)));
    }
    let n = rows * cols;
    let need = 12 + n * 4 + n;
    if payload.len() < need {
        return Err(io::Error::new(ErrorKind::InvalidData,
            format!("ToF payload {} < expected {}", payload.len(), need)));
    }
    let mut ranges_m = [[0.0_f32; TOF_COLS]; TOF_ROWS];
    let mut status   = [[0u8;    TOF_COLS]; TOF_ROWS];
    let dist_off = 12;
    let stat_off = dist_off + n * 4;
    for r in 0..TOF_ROWS {
        for c in 0..TOF_COLS {
            let off = dist_off + (r * TOF_COLS + c) * 4;
            ranges_m[r][c] = read_f32_le(&payload[off..off + 4]);
            status[r][c]   = payload[stat_off + r * TOF_COLS + c];
        }
    }
    Ok(TofRecord { ts_us, sender_ts_s, ranges_m, status })
}

fn decode_twin(ts_us: u64, payload: &[u8]) -> io::Result<TwinRecord> {
    if payload.len() != TWIN_PACKET_SIZE {
        return Err(io::Error::new(ErrorKind::InvalidData,
            format!("twin payload {} != expected {}",
                    payload.len(), TWIN_PACKET_SIZE)));
    }
    let sender_ts_s = read_f64_le(&payload[0..8]);
    let f = |idx: usize| -> f32 {
        let off = 8 + idx * 4;
        read_f32_le(&payload[off..off + 4])
    };
    let quat_wxyz = [f(0), f(1), f(2), f(3)];
    let mut joints = [0.0_f32; 15];
    for i in 0..15 { joints[i] = f(4 + i); }
    let mut motor_currents_ma = [0.0_f32; 15];
    for i in 0..15 { motor_currents_ma[i] = f(19 + i); }
    let odom_x  = f(34);
    let odom_y  = f(35);
    let ball_xyz = [f(36), f(37), f(38)];
    let odom_z   = f(39);
    let odom_yaw = f(40);
    Ok(TwinRecord {
        ts_us, sender_ts_s, quat_wxyz, joints, motor_currents_ma,
        odom_x, odom_y, ball_xyz, odom_z, odom_yaw,
    })
}

// ── Byte helpers ─────────────────────────────────────────────────────────────

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f32_le(b: &[u8]) -> f32 {
    f32::from_le_bytes(b.try_into().unwrap())
}

fn read_f64_le(b: &[u8]) -> f64 {
    f64::from_le_bytes(b.try_into().unwrap())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_synthetic(path: &Path) {
        let mut f = std::fs::File::create(path).unwrap();
        // Header.
        f.write_all(MAGIC).unwrap();
        f.write_all(&VERSION.to_le_bytes()).unwrap();
        f.write_all(&123_456_789u64.to_le_bytes()).unwrap();
        // ToF record at ts_us = 1000.
        let mut tof_payload = Vec::new();
        tof_payload.extend(&3.1415_f64.to_le_bytes());
        tof_payload.push(8);  // rows
        tof_payload.push(8);  // cols
        tof_payload.extend(&[0u8, 0u8]); // reserved
        for r in 0..8 {
            for c in 0..8 {
                let v = (r * 8 + c) as f32 / 100.0;
                tof_payload.extend(&v.to_le_bytes());
            }
        }
        for r in 0..8 {
            for c in 0..8 {
                tof_payload.push((r * 8 + c) as u8);
            }
        }
        f.write_all(&1000u64.to_le_bytes()).unwrap();
        f.write_all(&[STREAM_TOF]).unwrap();
        f.write_all(&(tof_payload.len() as u32).to_le_bytes()).unwrap();
        f.write_all(&tof_payload).unwrap();
        // Twin record at ts_us = 2000.
        let mut twin = Vec::with_capacity(TWIN_PACKET_SIZE);
        twin.extend(&2.71_f64.to_le_bytes());
        for i in 0..41 {
            twin.extend(&((i as f32) * 0.1).to_le_bytes());
        }
        f.write_all(&2000u64.to_le_bytes()).unwrap();
        f.write_all(&[STREAM_TWIN]).unwrap();
        f.write_all(&(twin.len() as u32).to_le_bytes()).unwrap();
        f.write_all(&twin).unwrap();
    }

    #[test]
    fn round_trip_decode() {
        let dir = tempdir().unwrap();
        let path = dir.join("test.mdlg");
        write_synthetic(&path);
        let mut r = SessionReplayer::open(&path).unwrap();
        assert_eq!(r.epoch_unix_ms(), 123_456_789);
        let rec1 = r.next().unwrap().unwrap();
        if let Record::Tof(t) = rec1 {
            assert_eq!(t.ts_us, 1000);
            assert!((t.sender_ts_s - 3.1415).abs() < 1e-9);
            assert_eq!(t.status[0][0], 0);
            assert_eq!(t.status[7][7], 63);
            assert!((t.ranges_m[0][0] - 0.0).abs() < 1e-6);
            assert!((t.ranges_m[7][7] - 0.63).abs() < 1e-5);
        } else {
            panic!("expected Tof");
        }
        let rec2 = r.next().unwrap().unwrap();
        if let Record::Twin(d) = rec2 {
            assert_eq!(d.ts_us, 2000);
            assert!((d.sender_ts_s - 2.71).abs() < 1e-9);
            // Quat is the first 4 floats.
            assert!((d.quat_wxyz[0] - 0.0).abs() < 1e-6);
            assert!((d.quat_wxyz[3] - 0.3).abs() < 1e-5);
            // odom_yaw is the last float: index 40 → 4.0.
            assert!((d.odom_yaw - 4.0).abs() < 1e-5);
        } else {
            panic!("expected Twin");
        }
        assert!(r.next().is_none());
    }

    fn tempdir() -> io::Result<std::path::PathBuf> {
        let p = std::env::temp_dir().join(format!(
            "mdlg_test_{}", std::process::id(),
        ));
        std::fs::create_dir_all(&p)?;
        Ok(p)
    }
}
