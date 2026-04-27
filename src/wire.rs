//! Wire protocol between the duck (server) and a laptop viewer (client).
//!
//! Framing: every message is `u32 LE length` ++ `u8 tag` ++ payload.
//! `length` covers the tag + payload, so the on-wire frame is `length + 4`
//! bytes total. All multi-byte fields are little-endian.
//!
//! The protocol is intentionally minimal — no serde, no schemas. Add a
//! field by appending; bump the version constant if you ever break a
//! message's layout. Forward-compat is the caller's responsibility (we
//! return `BadVersion` and let the viewer warn the user to upgrade).

use std::io::{self, Read, Write};

pub const PROTOCOL_VERSION: u32 = 1;

// Message tags. Keep them stable.
pub const TAG_HELLO:   u8 = 0x01;
pub const TAG_POSE:    u8 = 0x02;
pub const TAG_MAP:     u8 = 0x03;
pub const TAG_PATH:    u8 = 0x04;
pub const TAG_SCAN:    u8 = 0x05;
pub const TAG_GOAL:    u8 = 0x80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockState { Searching = 0, Tracking = 1 }

impl LockState {
    fn to_u8(self) -> u8 { self as u8 }
    fn from_u8(b: u8) -> Self {
        match b { 1 => LockState::Tracking, _ => LockState::Searching }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Hello { pub version: u32 }

#[derive(Debug, Clone, Copy)]
pub struct Pose {
    pub x: f32, pub y: f32, pub yaw: f32,
    pub std_xy: f32, pub residual_m: f32,
    pub lock: LockState,
    pub timestamp_ms: u64,
}

/// A path is a list of (x, y) waypoints in world coordinates.
#[derive(Debug, Clone, Default)]
pub struct Path { pub waypoints: Vec<(f32, f32)> }

#[derive(Debug, Clone, Default)]
pub struct Scan {
    pub angles_body: Vec<f32>,
    pub ranges:      Vec<f32>,    // NaN = no return
    pub origin:      (f32, f32),  // sensor origin in world (for visualization)
}

#[derive(Debug, Clone, Copy)]
pub struct Goal { pub x: f32, pub y: f32 }

/// Inbound (server-side) and outbound messages, tagged.
#[derive(Debug)]
pub enum Message {
    Hello(Hello),
    Pose(Pose),
    /// Map payload is the raw bytes of `OccupancyGrid::save` — same
    /// format on disk and on the wire so there's one parser.
    Map(Vec<u8>),
    Path(Path),
    Scan(Scan),
    Goal(Goal),
}

#[derive(Debug)]
pub enum WireError {
    Io(io::Error),
    BadTag(u8),
    BadVersion(u32),
    Truncated,
}

impl From<io::Error> for WireError {
    fn from(e: io::Error) -> Self { WireError::Io(e) }
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireError::Io(e) => write!(f, "io: {e}"),
            WireError::BadTag(t) => write!(f, "unknown message tag 0x{t:02x}"),
            WireError::BadVersion(v) => write!(f, "unsupported protocol version {v}"),
            WireError::Truncated => write!(f, "truncated message"),
        }
    }
}

impl std::error::Error for WireError {}

// ── Frame helpers ─────────────────────────────────────────────────────────

fn write_frame<W: Write>(w: &mut W, tag: u8, body: &[u8]) -> io::Result<()> {
    let len = (body.len() + 1) as u32;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&[tag])?;
    w.write_all(body)?;
    Ok(())
}

fn read_frame<R: Read>(r: &mut R) -> Result<(u8, Vec<u8>), WireError> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len < 1 { return Err(WireError::Truncated); }
    let mut tag_buf = [0u8; 1];
    r.read_exact(&mut tag_buf)?;
    let mut body = vec![0u8; len - 1];
    r.read_exact(&mut body)?;
    Ok((tag_buf[0], body))
}

// ── Per-message encode/decode ────────────────────────────────────────────

fn put_u32(b: &mut Vec<u8>, v: u32) { b.extend_from_slice(&v.to_le_bytes()); }
fn put_u64(b: &mut Vec<u8>, v: u64) { b.extend_from_slice(&v.to_le_bytes()); }
fn put_f32(b: &mut Vec<u8>, v: f32) { b.extend_from_slice(&v.to_le_bytes()); }

fn take_u32(r: &mut &[u8]) -> Result<u32, WireError> {
    if r.len() < 4 { return Err(WireError::Truncated); }
    let v = u32::from_le_bytes(r[..4].try_into().unwrap());
    *r = &r[4..]; Ok(v)
}
fn take_u64(r: &mut &[u8]) -> Result<u64, WireError> {
    if r.len() < 8 { return Err(WireError::Truncated); }
    let v = u64::from_le_bytes(r[..8].try_into().unwrap());
    *r = &r[8..]; Ok(v)
}
fn take_f32(r: &mut &[u8]) -> Result<f32, WireError> {
    if r.len() < 4 { return Err(WireError::Truncated); }
    let v = f32::from_le_bytes(r[..4].try_into().unwrap());
    *r = &r[4..]; Ok(v)
}

pub fn write_hello<W: Write>(w: &mut W, h: Hello) -> io::Result<()> {
    let mut body = Vec::with_capacity(4);
    put_u32(&mut body, h.version);
    write_frame(w, TAG_HELLO, &body)
}

pub fn write_pose<W: Write>(w: &mut W, p: Pose) -> io::Result<()> {
    let mut body = Vec::with_capacity(32);
    put_f32(&mut body, p.x);
    put_f32(&mut body, p.y);
    put_f32(&mut body, p.yaw);
    put_f32(&mut body, p.std_xy);
    put_f32(&mut body, p.residual_m);
    body.push(p.lock.to_u8());
    put_u64(&mut body, p.timestamp_ms);
    write_frame(w, TAG_POSE, &body)
}

pub fn write_map<W: Write>(w: &mut W, map_bytes: &[u8]) -> io::Result<()> {
    write_frame(w, TAG_MAP, map_bytes)
}

pub fn write_path<W: Write>(w: &mut W, p: &Path) -> io::Result<()> {
    let mut body = Vec::with_capacity(4 + p.waypoints.len() * 8);
    put_u32(&mut body, p.waypoints.len() as u32);
    for &(x, y) in &p.waypoints {
        put_f32(&mut body, x);
        put_f32(&mut body, y);
    }
    write_frame(w, TAG_PATH, &body)
}

pub fn write_scan<W: Write>(w: &mut W, s: &Scan) -> io::Result<()> {
    let mut body = Vec::with_capacity(8 + s.angles_body.len() * 8);
    put_u32(&mut body, s.angles_body.len() as u32);
    put_f32(&mut body, s.origin.0);
    put_f32(&mut body, s.origin.1);
    for &a in &s.angles_body { put_f32(&mut body, a); }
    for &r in &s.ranges      { put_f32(&mut body, r); }
    write_frame(w, TAG_SCAN, &body)
}

pub fn write_goal<W: Write>(w: &mut W, g: Goal) -> io::Result<()> {
    let mut body = Vec::with_capacity(8);
    put_f32(&mut body, g.x);
    put_f32(&mut body, g.y);
    write_frame(w, TAG_GOAL, &body)
}

pub fn read_message<R: Read>(r: &mut R) -> Result<Message, WireError> {
    let (tag, body) = read_frame(r)?;
    let mut s = body.as_slice();
    match tag {
        TAG_HELLO => {
            let v = take_u32(&mut s)?;
            if v != PROTOCOL_VERSION { return Err(WireError::BadVersion(v)); }
            Ok(Message::Hello(Hello { version: v }))
        }
        TAG_POSE => {
            let x = take_f32(&mut s)?;
            let y = take_f32(&mut s)?;
            let yaw = take_f32(&mut s)?;
            let std_xy = take_f32(&mut s)?;
            let residual_m = take_f32(&mut s)?;
            if s.is_empty() { return Err(WireError::Truncated); }
            let lock = LockState::from_u8(s[0]); s = &s[1..];
            let timestamp_ms = take_u64(&mut s)?;
            Ok(Message::Pose(Pose { x, y, yaw, std_xy, residual_m, lock, timestamp_ms }))
        }
        TAG_MAP => Ok(Message::Map(body)),
        TAG_PATH => {
            let n = take_u32(&mut s)? as usize;
            let mut wpts = Vec::with_capacity(n);
            for _ in 0..n {
                wpts.push((take_f32(&mut s)?, take_f32(&mut s)?));
            }
            Ok(Message::Path(Path { waypoints: wpts }))
        }
        TAG_SCAN => {
            let n = take_u32(&mut s)? as usize;
            let ox = take_f32(&mut s)?; let oy = take_f32(&mut s)?;
            let mut angles = Vec::with_capacity(n);
            for _ in 0..n { angles.push(take_f32(&mut s)?); }
            let mut ranges = Vec::with_capacity(n);
            for _ in 0..n { ranges.push(take_f32(&mut s)?); }
            Ok(Message::Scan(Scan { angles_body: angles, ranges, origin: (ox, oy) }))
        }
        TAG_GOAL => {
            let x = take_f32(&mut s)?;
            let y = take_f32(&mut s)?;
            Ok(Message::Goal(Goal { x, y }))
        }
        t => Err(WireError::BadTag(t)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(write_one: impl FnOnce(&mut Vec<u8>)) -> Message {
        let mut buf = Vec::new();
        write_one(&mut buf);
        read_message(&mut buf.as_slice()).unwrap()
    }

    #[test]
    fn hello_round_trip() {
        let msg = round_trip(|b| write_hello(b, Hello { version: PROTOCOL_VERSION }).unwrap());
        assert!(matches!(msg, Message::Hello(Hello { version: 1 })));
    }

    #[test]
    fn pose_round_trip() {
        let p = Pose {
            x: 1.5, y: -0.7, yaw: 0.3,
            std_xy: 0.05, residual_m: 0.12,
            lock: LockState::Tracking,
            timestamp_ms: 123_456,
        };
        let msg = round_trip(|b| write_pose(b, p).unwrap());
        match msg {
            Message::Pose(q) => {
                assert!((q.x - p.x).abs() < 1e-6);
                assert!((q.yaw - p.yaw).abs() < 1e-6);
                assert_eq!(q.lock, LockState::Tracking);
                assert_eq!(q.timestamp_ms, p.timestamp_ms);
            }
            _ => panic!("expected pose"),
        }
    }

    #[test]
    fn path_round_trip() {
        let p = Path { waypoints: vec![(0.0, 0.0), (1.0, 0.5), (1.5, 0.5)] };
        let msg = round_trip(|b| write_path(b, &p).unwrap());
        match msg {
            Message::Path(q) => assert_eq!(q.waypoints.len(), 3),
            _ => panic!("expected path"),
        }
    }

    #[test]
    fn goal_round_trip() {
        let msg = round_trip(|b| write_goal(b, Goal { x: -1.2, y: 0.4 }).unwrap());
        match msg {
            Message::Goal(g) => { assert_eq!(g.x, -1.2); assert_eq!(g.y, 0.4); }
            _ => panic!("expected goal"),
        }
    }

    #[test]
    fn unknown_tag_errors() {
        let mut buf = Vec::new();
        // length=1, tag=0x55, no body
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.push(0x55);
        let r = read_message(&mut buf.as_slice());
        assert!(matches!(r, Err(WireError::BadTag(0x55))));
    }
}
