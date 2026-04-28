//! Quick microbench for the likelihood-field global relocalize.
//! Builds an apartment-sized grid + scattered obstacles, runs the
//! distance transform, then times `global_relocalize_field` against
//! a synthetic wide scan.
//!
//! Run: `cargo run --release --example bench_field`

use std::time::Instant;

use microduck_maploc::accumulator::BufferedBeam;
use microduck_maploc::grid::{GridConfig, OccupancyGrid};
use microduck_maploc::mcl::{FieldRelocConfig, Localizer, MclConfig};

fn main() {
    let mut g = OccupancyGrid::new(GridConfig::default());
    // Apartment-shaped obstacles + free interior. Mirrors the python
    // bench's mapping density (~3000 occupied cells over an 8×6 m grid).
    for t in 0..120 {
        let p = -3.5 + (t as f32) * 0.06;
        for _ in 0..10 {
            g.integrate_ray(0.0, 0.0,  3.5,  p, true);
            g.integrate_ray(0.0, 0.0, -3.5,  p, true);
            g.integrate_ray(0.0, 0.0,   p,  2.5, true);
            g.integrate_ray(0.0, 0.0,   p, -2.5, true);
        }
    }
    let n_occ: usize = g.log_raw().iter().filter(|&&v| v > 150).count();
    let n_free: usize = (0..g.height())
        .flat_map(|i| (0..g.width()).map(move |j| (i, j)))
        .filter(|&(i, j)| g.is_known_free(i, j))
        .count();
    println!("grid: {}×{} cells, {n_occ} occupied, {n_free} known-free",
             g.width(), g.height());

    let t0 = Instant::now();
    let _ = g.distance_field(150);
    println!("distance_field (cold): {:.1} ms",
             t0.elapsed().as_secs_f64() * 1000.0);
    let t0 = Instant::now();
    let _ = g.distance_field(150);
    println!("distance_field (cached): {:.3} ms",
             t0.elapsed().as_secs_f64() * 1000.0);

    let mut loc = Localizer::new(&g, MclConfig { n_particles: 2000, ..MclConfig::default() }, 0);

    // Synthesize a wide scan from the origin.
    let mut beams: Vec<BufferedBeam> = Vec::new();
    for k in 0..32 {
        let a = -std::f32::consts::PI + (k as f32) * (std::f32::consts::PI / 16.0);
        let r = g.cast_ray(0.0, 0.0, a, 4.0);
        beams.push(BufferedBeam {
            angle_body: a, range_m: r,
            est_origin: (0.0, 0.0), est_yaw: 0.0,
        });
    }
    println!("scan: {} beams", beams.len());

    // Warm + bench.
    loc.global_relocalize_field(&mut g, &beams, (0.0, 0.0, 0.0));
    let n_iters = 5;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        loc.global_relocalize_field_with(
            &mut g, &beams, (0.0, 0.0, 0.0),
            FieldRelocConfig::default(),
        );
    }
    let dt_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
    println!("global_relocalize_field: {:.1} ms / call (n={n_iters})", dt_ms);
}
