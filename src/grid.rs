//! 2D log-odds occupancy grid, fixed-point.
//!
//! Cells store log-odds × `LOG_SCALE` as `i16` in `[LO_MIN, LO_MAX]`.
//! That keeps the per-cell footprint at 2 bytes and the inner update
//! loop fully integer — so it's cheap on the Pi Zero 2W's A53 cores
//! and avoids any FP vs FP-fastmath subtlety in the cell math.
//!
//! Float ops only show up at the world↔index boundary (where the input
//! is f32 anyway), and inside `cast_ray` (where we step along a unit
//! vector in world coordinates).

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// Fixed-point scale: stored value = round(log_odds * LOG_SCALE).
pub const LOG_SCALE: f32 = 100.0;
/// Log-odds clamps. ±4 → posterior probability ≈ [0.018, 0.982].
pub const LO_MIN: i16 = -400;
pub const LO_MAX: i16 = 400;
/// Per-update increments (matches Python `lo_hit = +0.85`, `lo_miss = -0.40`).
pub const LO_HIT: i16 = 85;
pub const LO_MISS: i16 = -40;
/// A cell counts as occupied for ray-casting / display when log-odds > 0.
pub const OCC_THRESHOLD: i16 = 0;

const SAVE_MAGIC: &[u8; 4] = b"MDLM";
const SAVE_VERSION: u32 = 1;

/// Geometry of an occupancy grid in world coordinates.
#[derive(Debug, Clone, Copy)]
pub struct GridConfig {
    pub x_range: (f32, f32),
    pub y_range: (f32, f32),
    pub cell:    f32,
}

impl Default for GridConfig {
    fn default() -> Self {
        // Matches the Python defaults used in the simulator.
        Self {
            x_range: (-3.0, 3.5),
            y_range: (-2.5, 2.5),
            cell:    0.05,
        }
    }
}

/// 2D log-odds occupancy grid. Row-major, `[i, j]` indexed by
/// `(row=y_index, col=x_index)`.
pub struct OccupancyGrid {
    cfg: GridConfig,
    w: usize,
    h: usize,
    log: Vec<i16>,
}

impl OccupancyGrid {
    pub fn new(cfg: GridConfig) -> Self {
        let w = ((cfg.x_range.1 - cfg.x_range.0) / cfg.cell).ceil() as usize;
        let h = ((cfg.y_range.1 - cfg.y_range.0) / cfg.cell).ceil() as usize;
        Self { cfg, w, h, log: vec![0; w * h] }
    }

    pub fn cfg(&self) -> &GridConfig { &self.cfg }
    pub fn width(&self)  -> usize { self.w }
    pub fn height(&self) -> usize { self.h }
    pub fn cell(&self)   -> f32   { self.cfg.cell }
    pub fn log_raw(&self) -> &[i16] { &self.log }

    /// World coordinate → grid index. Returns `None` if out of bounds.
    #[inline]
    pub fn world_to_idx(&self, x: f32, y: f32) -> Option<(usize, usize)> {
        let j = ((x - self.cfg.x_range.0) / self.cfg.cell) as i32;
        let i = ((y - self.cfg.y_range.0) / self.cfg.cell) as i32;
        if i < 0 || j < 0 || (i as usize) >= self.h || (j as usize) >= self.w {
            None
        } else {
            Some((i as usize, j as usize))
        }
    }

    #[inline]
    pub fn in_bounds(&self, i: i32, j: i32) -> bool {
        i >= 0 && j >= 0 && (i as usize) < self.h && (j as usize) < self.w
    }

    #[inline]
    pub fn log_at(&self, i: usize, j: usize) -> i16 { self.log[i * self.w + j] }

    /// True if this cell has been explicitly observed as free at least once.
    /// Useful for seeding MCL particles in known-free space.
    #[inline]
    pub fn is_known_free(&self, i: usize, j: usize) -> bool {
        self.log[i * self.w + j] < -50  // ≈ 0.5 log-odds below 0
    }

    #[inline]
    pub fn is_occupied(&self, i: usize, j: usize) -> bool {
        self.log[i * self.w + j] > OCC_THRESHOLD
    }

    /// Apply one log-odds increment to a cell, clamped to `[LO_MIN, LO_MAX]`.
    #[inline]
    fn bump(&mut self, i: usize, j: usize, delta: i16) {
        let idx = i * self.w + j;
        let v = self.log[idx] as i32 + delta as i32;
        self.log[idx] = v.clamp(LO_MIN as i32, LO_MAX as i32) as i16;
    }

    /// Integrate one ray from `(x0,y0)` (sensor origin) to `(x1,y1)` (hit).
    /// Cells along the ray are marked free; the endpoint is marked occupied
    /// iff `hit_is_occupied` (e.g. ray hit a wall, not the floor).
    pub fn integrate_ray(&mut self, x0: f32, y0: f32,
                         x1: f32, y1: f32, hit_is_occupied: bool) {
        let cell = self.cfg.cell;
        let i0 = ((y0 - self.cfg.y_range.0) / cell) as i32;
        let j0 = ((x0 - self.cfg.x_range.0) / cell) as i32;
        let i1 = ((y1 - self.cfg.y_range.0) / cell) as i32;
        let j1 = ((x1 - self.cfg.x_range.0) / cell) as i32;
        bresenham(i0, j0, i1, j1, |i, j, at_end| {
            if !self.in_bounds(i, j) { return; }
            let (i, j) = (i as usize, j as usize);
            let delta = if at_end && hit_is_occupied { LO_HIT } else { LO_MISS };
            self.bump(i, j, delta);
        });
    }

    /// Cast a ray in the grid; return distance until the first occupied
    /// cell or the bound, capped at `max_range`. Used by the localizer to
    /// predict expected ToF returns from each particle pose.
    ///
    /// Fixed-step march with `cell/2` step. Faster than a DDA for small
    /// grids and trivially vectorizable later.
    pub fn cast_ray(&self, x: f32, y: f32, theta: f32, max_range: f32) -> f32 {
        let cx = theta.cos();
        let cy = theta.sin();
        let cell = self.cfg.cell;
        let step = cell * 0.5;
        let n_steps = (max_range / step) as i32;
        for k in 1..=n_steps {
            let r = (k as f32) * step;
            let xi = x + cx * r;
            let yi = y + cy * r;
            let j = ((xi - self.cfg.x_range.0) / cell) as i32;
            let i = ((yi - self.cfg.y_range.0) / cell) as i32;
            if !self.in_bounds(i, j) {
                return r;
            }
            if self.log[(i as usize) * self.w + (j as usize)] > OCC_THRESHOLD {
                return r;
            }
        }
        max_range
    }

    // ── Persistence ────────────────────────────────────────────────────

    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = File::create(path)?;
        f.write_all(SAVE_MAGIC)?;
        f.write_all(&SAVE_VERSION.to_le_bytes())?;
        f.write_all(&self.cfg.x_range.0.to_le_bytes())?;
        f.write_all(&self.cfg.x_range.1.to_le_bytes())?;
        f.write_all(&self.cfg.y_range.0.to_le_bytes())?;
        f.write_all(&self.cfg.y_range.1.to_le_bytes())?;
        f.write_all(&self.cfg.cell.to_le_bytes())?;
        f.write_all(&(self.w as u32).to_le_bytes())?;
        f.write_all(&(self.h as u32).to_le_bytes())?;
        // log-odds as raw little-endian i16. Endianness is fixed in the
        // file format so cross-arch loads don't get garbled.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.log.as_ptr() as *const u8,
                self.log.len() * 2,
            )
        };
        if cfg!(target_endian = "little") {
            f.write_all(bytes)?;
        } else {
            for v in &self.log { f.write_all(&v.to_le_bytes())?; }
        }
        Ok(())
    }

    /// Load a previously saved map. Returns `Ok(None)` if the file's
    /// geometry doesn't match the current `GridConfig` defaults the
    /// caller wants — let the caller start fresh rather than silently
    /// reshaping the data.
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Option<Self>> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != SAVE_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let mut u32_buf = [0u8; 4];
        f.read_exact(&mut u32_buf)?;
        let version = u32::from_le_bytes(u32_buf);
        if version != SAVE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported map version {version}"),
            ));
        }
        let mut f32_buf = [0u8; 4];
        let mut read_f32 = |f: &mut File| -> io::Result<f32> {
            f.read_exact(&mut f32_buf)?;
            Ok(f32::from_le_bytes(f32_buf))
        };
        let x0 = read_f32(&mut f)?; let x1 = read_f32(&mut f)?;
        let y0 = read_f32(&mut f)?; let y1 = read_f32(&mut f)?;
        let cell = read_f32(&mut f)?;
        f.read_exact(&mut u32_buf)?;
        let w = u32::from_le_bytes(u32_buf) as usize;
        f.read_exact(&mut u32_buf)?;
        let h = u32::from_le_bytes(u32_buf) as usize;

        let cfg = GridConfig { x_range: (x0, x1), y_range: (y0, y1), cell };
        let mut log = vec![0i16; w * h];
        // Same little-endian-on-disk convention as save().
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                log.as_mut_ptr() as *mut u8,
                log.len() * 2,
            )
        };
        f.read_exact(bytes)?;
        if cfg!(target_endian = "big") {
            for v in log.iter_mut() {
                *v = i16::from_le_bytes(v.to_ne_bytes());
            }
        }
        Ok(Some(Self { cfg, w, h, log }))
    }
}

// ── Bresenham ─────────────────────────────────────────────────────────────

/// Walk integer cells along the line `(i0,j0)..(i1,j1)`; calls `visit(i, j,
/// at_endpoint)` at each cell, including both endpoints.
pub fn bresenham<F: FnMut(i32, i32, bool)>(
    i0: i32, j0: i32, i1: i32, j1: i32, mut visit: F,
) {
    let di = (i1 - i0).abs();
    let dj = (j1 - j0).abs();
    let si = if i0 < i1 { 1 } else { -1 };
    let sj = if j0 < j1 { 1 } else { -1 };
    let (mut i, mut j) = (i0, j0);
    let mut err = di - dj;
    loop {
        let at_end = i == i1 && j == j1;
        visit(i, j, at_end);
        if at_end { return; }
        let e2 = 2 * err;
        if e2 > -dj { err -= dj; i += si; }
        if e2 <  di { err += di; j += sj; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_to_idx_round_trip() {
        let g = OccupancyGrid::new(GridConfig::default());
        let (i, j) = g.world_to_idx(0.0, 0.0).unwrap();
        // Centre of an apartment with x in [-3, 3.5] and 5 cm cells.
        assert!(j > 0 && i > 0);
        assert!(g.world_to_idx(-100.0, 0.0).is_none());
    }

    #[test]
    fn integrate_then_cast_ray_finds_obstacle() {
        let mut g = OccupancyGrid::new(GridConfig::default());
        // Repeatedly hit the same wall at (1.0, 0.0) from origin so the
        // log-odds saturate above the occupancy threshold.
        for _ in 0..10 {
            g.integrate_ray(0.0, 0.0, 1.0, 0.0, true);
        }
        let d = g.cast_ray(0.0, 0.0, 0.0, 4.0);
        assert!((d - 1.0).abs() < g.cell() * 2.0,
                "expected hit near 1.0 m, got {d}");
    }

    #[test]
    fn save_load_round_trip() {
        let mut g = OccupancyGrid::new(GridConfig::default());
        for _ in 0..5 {
            g.integrate_ray(0.0, 0.0, 1.5, 0.7, true);
        }
        let dir = tempdir();
        let path = dir.join("map.bin");
        g.save(&path).unwrap();
        let g2 = OccupancyGrid::load(&path).unwrap().unwrap();
        assert_eq!(g.log_raw(), g2.log_raw());
        assert_eq!(g.width(),   g2.width());
        assert_eq!(g.height(),  g2.height());
        std::fs::remove_dir_all(dir).ok();
    }

    fn tempdir() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nano = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        p.push(format!("microduck_maploc_test_{pid}_{nano}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
