//! Print summary stats for a `.mdlg` session — sanity check that the
//! Rust replay reader handles real recordings.
//!
//! Usage:  cargo run --example dump_session -- path/to/session.mdlg

use std::env;
use std::process::ExitCode;

use microduck_maploc::replay::{Record, SessionReplayer};

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: dump_session <path/to/session.mdlg>");
            return ExitCode::FAILURE;
        }
    };
    let replayer = match SessionReplayer::open(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {}", path, e);
            return ExitCode::FAILURE;
        }
    };
    println!("session: {}", path);
    println!("  epoch_unix_ms = {}", replayer.epoch_unix_ms());

    let mut n_tof  = 0u64;
    let mut n_twin = 0u64;
    let mut first_ts = u64::MAX;
    let mut last_ts  = 0u64;
    let mut n_valid_zones_total = 0u64;
    let mut n_valid_zones_max   = 0u32;
    let mut last_odom = (0.0_f32, 0.0_f32, 0.0_f32);

    for rec in replayer {
        let rec = match rec {
            Ok(r) => r,
            Err(e) => {
                eprintln!("decode error: {}", e);
                return ExitCode::FAILURE;
            }
        };
        first_ts = first_ts.min(rec.ts_us());
        last_ts  = last_ts.max(rec.ts_us());
        match rec {
            Record::Tof(t) => {
                n_tof += 1;
                let mut n_valid = 0u32;
                for r in 0..8 {
                    for c in 0..8 {
                        if t.ranges_m[r][c].is_finite() {
                            n_valid += 1;
                        }
                    }
                }
                n_valid_zones_total += n_valid as u64;
                n_valid_zones_max   = n_valid_zones_max.max(n_valid);
            }
            Record::Twin(d) => {
                n_twin += 1;
                last_odom = (d.odom_x, d.odom_y, d.odom_yaw);
            }
        }
    }

    let dur_s = (last_ts - first_ts) as f64 / 1e6;
    let mean_valid = if n_tof > 0 {
        n_valid_zones_total as f64 / n_tof as f64
    } else {
        0.0
    };
    println!("  duration: {:.1} s", dur_s);
    println!("  ToF frames: {} ({:.1}/s, mean {:.1} valid zones, max {})",
             n_tof, n_tof as f64 / dur_s.max(1e-3), mean_valid, n_valid_zones_max);
    println!("  twin packets: {} ({:.1}/s)",
             n_twin, n_twin as f64 / dur_s.max(1e-3));
    println!("  final odom: x={:+.3} m  y={:+.3} m  yaw={:+.1}°",
             last_odom.0, last_odom.1, last_odom.2.to_degrees());
    ExitCode::SUCCESS
}
