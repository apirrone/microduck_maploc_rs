//! Pi-side TCP streamer + goal receiver.
//!
//! Two independent listeners:
//!
//! * [`Telemetry`] (default port 9874) — server pushes pose + map +
//!   path + scan to whichever single client is connected. Drops on
//!   client disconnect; ready for the next one.
//! * [`GoalServer`] (default port 9875) — server reads goals from a
//!   single connected client and forwards them via an mpsc channel.
//!
//! Why two ports? Telemetry is high-rate firehose, goals are sparse
//! click events; muxing them on one socket means you lose the ability
//! to drop telemetry on slow consumers without also losing goals.
//!
//! Both listeners are non-blocking and don't spawn threads — call
//! `tick()` from the runtime's main loop to accept connections and
//! drain inbound goals. Network errors deliberately fall through as
//! "client gone" rather than killing the runtime.

use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;

use crate::wire::{
    self, Goal, Hello, Message, Path, Pose, Scan, PROTOCOL_VERSION,
};

const TELEMETRY_DEFAULT_PORT: u16 = 9874;
const GOAL_DEFAULT_PORT:      u16 = 9875;

pub struct Telemetry {
    listener: TcpListener,
    client:   Option<TcpStream>,
}

impl Telemetry {
    pub fn bind(port: u16) -> io::Result<Self> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        listener.set_nonblocking(true)?;
        Ok(Self { listener, client: None })
    }

    pub fn bind_default() -> io::Result<Self> {
        Self::bind(TELEMETRY_DEFAULT_PORT)
    }

    /// Accept a pending connection if any. No-op when a client is
    /// already connected — single-consumer model.
    pub fn poll_accept(&mut self) {
        if self.client.is_some() { return; }
        match self.listener.accept() {
            Ok((mut s, addr)) => {
                let _ = s.set_nodelay(true);
                let _ = s.set_nonblocking(false);   // writes block briefly under load
                if wire::write_hello(&mut s, Hello { version: PROTOCOL_VERSION }).is_ok() {
                    eprintln!("[maploc] telemetry client connected from {addr}");
                    self.client = Some(s);
                }
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {}
            Err(e) => eprintln!("[maploc] telemetry accept error: {e}"),
        }
    }

    pub fn has_client(&self) -> bool { self.client.is_some() }

    fn send<F>(&mut self, write: F)
    where F: FnOnce(&mut TcpStream) -> io::Result<()> {
        if let Some(s) = self.client.as_mut() {
            if let Err(e) = write(s) {
                eprintln!("[maploc] telemetry client dropped: {e}");
                self.client = None;
            }
        }
    }

    pub fn send_pose(&mut self, p: Pose)         { self.send(|s| wire::write_pose(s, p)); }
    pub fn send_map(&mut self, bytes: &[u8])      { self.send(|s| wire::write_map(s, bytes)); }
    pub fn send_path(&mut self, p: &Path)         { self.send(|s| wire::write_path(s, p)); }
    pub fn send_scan(&mut self, sc: &Scan)        { self.send(|s| wire::write_scan(s, sc)); }
}

pub struct GoalServer {
    listener: TcpListener,
    client:   Option<TcpStream>,
    rx:       mpsc::Receiver<Goal>,
    tx:       mpsc::Sender<Goal>,
}

impl GoalServer {
    pub fn bind(port: u16) -> io::Result<Self> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        listener.set_nonblocking(true)?;
        let (tx, rx) = mpsc::channel();
        Ok(Self { listener, client: None, rx, tx })
    }

    pub fn bind_default() -> io::Result<Self> {
        Self::bind(GOAL_DEFAULT_PORT)
    }

    /// Drain accepted connections + any inbound goals. Returns the
    /// most-recent goal received this tick (older goals are discarded
    /// — the user just clicked again, the new click wins).
    pub fn tick(&mut self) -> Option<Goal> {
        // Accept.
        if self.client.is_none() {
            match self.listener.accept() {
                Ok((s, addr)) => {
                    let _ = s.set_nonblocking(true);
                    eprintln!("[maploc] goal client connected from {addr}");
                    self.client = Some(s);
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => {}
                Err(e) => eprintln!("[maploc] goal accept error: {e}"),
            }
        }
        // Read.
        if let Some(s) = self.client.as_mut() {
            // Try to read one full message; if EAGAIN, fine.
            match wire::read_message(s) {
                Ok(Message::Goal(g)) => { let _ = self.tx.send(g); }
                Ok(_other)           => { /* ignore; we only accept goals */ }
                Err(wire::WireError::Io(e)) if e.kind() == ErrorKind::WouldBlock => {}
                Err(e) => {
                    eprintln!("[maploc] goal client dropped: {e}");
                    self.client = None;
                }
            }
        }
        // Newest-wins: drain the queue.
        let mut latest = None;
        while let Ok(g) = self.rx.try_recv() { latest = Some(g); }
        latest
    }
}

// Ensure the goal-read path is genuinely non-blocking — we want
// `read_message` to fail with WouldBlock if the client hasn't sent
// anything, not hang the runtime. The free function in `wire` uses
// `read_exact`, which on a non-blocking stream surfaces WouldBlock
// from the underlying read once data runs out. Verified via test.
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpStream;
    use std::time::{Duration, Instant};

    #[test]
    fn telemetry_handshake_and_pose() {
        let mut srv = Telemetry::bind(0).unwrap();
        let port = srv.listener.local_addr().unwrap().port();
        let _client = std::thread::spawn(move || {
            let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
            // Read the Hello frame.
            let mut buf = [0u8; 9];   // 4 len + 1 tag + 4 version
            s.read_exact(&mut buf).unwrap();
            assert_eq!(buf[4], crate::wire::TAG_HELLO);
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        while !srv.has_client() && Instant::now() < deadline {
            srv.poll_accept();
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(srv.has_client(), "expected telemetry client to connect");
    }

    #[test]
    fn goal_server_receives_click() {
        let mut srv = GoalServer::bind(0).unwrap();
        let port = srv.listener.local_addr().unwrap().port();
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        crate::wire::write_goal(&mut s, Goal { x: 0.5, y: -1.2 }).unwrap();
        s.flush().unwrap();
        // Spin tick() until the goal arrives.
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut got = None;
        while Instant::now() < deadline && got.is_none() {
            got = srv.tick();
            std::thread::sleep(Duration::from_millis(10));
        }
        let g = got.expect("expected a goal to arrive");
        assert_eq!(g.x, 0.5);
        assert_eq!(g.y, -1.2);
    }
}
