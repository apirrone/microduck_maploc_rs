//! ToF mount parameters + beam projection.
//!
//! Mirrors the math in `microduck_runtime/src/tof.rs::precompute_zone_lookups`
//! so offline tooling consumes raw 8x8 frames the same way the runtime
//! does. The output of `project_frame` is what every downstream consumer
//! (scan matcher, submap integration, etc.) actually wants:
//!
//!   * `angles_body[k]` — body-frame azimuth of beam k (radians)
//!   * `ranges_horiz[k]` — horizontal-plane range of beam k (metres),
//!                        NaN for floor / out-of-range / status-invalid beams.

const FOV_DEG: f32 = 45.0;
pub const TOF_ROWS: usize = 8;
pub const TOF_COLS: usize = 8;
pub const N_ZONES: usize = TOF_ROWS * TOF_COLS;

/// ToF mount parameters. Numbers come from `tools/calibrate_tof.py`.
#[derive(Debug, Clone, Copy)]
pub struct TofMount {
    /// Mount pitch in radians. Positive = nose-down (rotation about body +y).
    pub pitch_rad: f32,
    /// Mount yaw in radians. Positive = sensor +x rotated toward body +y
    /// (i.e. the sensor is pointing slightly to the duck's left).
    pub yaw_rad: f32,
    /// Sensor height above the floor (metres), as inferred by the
    /// pitch+height calibration. Used by the floor filter.
    pub sensor_height_m: f32,
    /// Multiplier applied to `sensor_height_m` for the floor threshold.
    /// `<1.0` makes the filter more aggressive (drops beams that *almost*
    /// reach the floor — catches ToF noise + small live-pitch deviations).
    /// 0.85 is a sane default.
    pub floor_safety: f32,
    /// Drop any horizontal-plane range below this (metres). Catches
    /// near-range spurious returns (own body in FOV, specular floor
    /// reflections, VL53L5CX cross-talk).
    pub min_range_m: f32,
}

impl Default for TofMount {
    fn default() -> Self {
        Self {
            pitch_rad: 0.0,
            yaw_rad: 0.0,
            sensor_height_m: 0.0,
            floor_safety: 1.0,
            min_range_m: 0.0,
        }
    }
}

/// Precomputed per-zone geometry. Cheap to store, expensive to compute,
/// constant for a given mount → cache it once and reuse.
pub struct ZoneLut {
    pub angles_body: [f32; N_ZONES],
    pub cos_elev:    [f32; N_ZONES],
    pub sin_below:   [f32; N_ZONES],
}

/// Build the per-zone direction lookup matching `microduck_runtime`'s
/// `tof.rs::precompute_zone_lookups`. Sensor convention: +x forward,
/// +y left, +z up. Row 0 is the top of the image (looking up).
pub fn precompute_zone_lookups(mount: &TofMount) -> ZoneLut {
    let half_deg = (FOV_DEG / 2.0) - (FOV_DEG / TOF_COLS as f32) / 2.0;
    let half = half_deg.to_radians();
    let step_az = (2.0 * half) / (TOF_COLS as f32 - 1.0);
    let step_el = (2.0 * half) / (TOF_ROWS as f32 - 1.0);
    let cos_p = mount.pitch_rad.cos();
    let sin_p = mount.pitch_rad.sin();
    let cos_y = mount.yaw_rad.cos();
    let sin_y = mount.yaw_rad.sin();
    let mut angles_body = [0.0_f32; N_ZONES];
    let mut cos_elev    = [0.0_f32; N_ZONES];
    let mut sin_below   = [0.0_f32; N_ZONES];
    let mut k = 0;
    for r in 0..TOF_ROWS {
        let el_s = half - (r as f32) * step_el;
        for c in 0..TOF_COLS {
            let az_s = half - (c as f32) * step_az;
            let cx = el_s.cos() * az_s.cos();
            let cy = el_s.cos() * az_s.sin();
            let cz = el_s.sin();
            // pitch about body +y, then yaw about body +z.
            let dx_p =  cos_p * cx + sin_p * cz;
            let dz_p = -sin_p * cx + cos_p * cz;
            let dy_p = cy;
            let dx_b = cos_y * dx_p - sin_y * dy_p;
            let dy_b = sin_y * dx_p + cos_y * dy_p;
            let dz_b = dz_p;
            angles_body[k] = dy_b.atan2(dx_b);
            cos_elev[k]    = (dx_b * dx_b + dy_b * dy_b).sqrt();
            sin_below[k]   = -dz_b;  // > 0 when beam looks below horizontal
            k += 1;
        }
    }
    ZoneLut { angles_body, cos_elev, sin_below }
}

/// Project a single 8x8 slant-range frame into per-beam
/// (body azimuth, horizontal range). Filters:
///   * NaN-pass-through (chip flagged invalid → NaN slant in input).
///   * Floor cutoff: drop beams whose `r * sin_below` exceeds
///     `mount.floor_safety * mount.sensor_height_m`.
///   * Min-range cutoff: drop beams whose horizontal range is below
///     `mount.min_range_m` (catches spurious near hits).
///
/// Returns a (angles, ranges_horiz) pair of length 64. NaN range means
/// "skip" — downstream consumers (scan matcher, submap integrate) all
/// already short-circuit on NaN/non-finite ranges.
pub fn project_frame(
    raw_slant_m: &[[f32; TOF_COLS]; TOF_ROWS],
    lut: &ZoneLut,
    mount: &TofMount,
) -> ([f32; N_ZONES], [f32; N_ZONES]) {
    let mut angles = [0.0_f32; N_ZONES];
    let mut ranges = [f32::NAN; N_ZONES];
    let height_threshold = mount.sensor_height_m * mount.floor_safety;
    let mut k = 0;
    for r in 0..TOF_ROWS {
        for c in 0..TOF_COLS {
            angles[k] = lut.angles_body[k];
            let r_slant = raw_slant_m[r][c];
            ranges[k] = if !r_slant.is_finite() {
                f32::NAN
            } else if mount.sensor_height_m > 0.0
                && lut.sin_below[k] > 0.0
                && r_slant * lut.sin_below[k] >= height_threshold
            {
                // Floor hit: would terminate within `floor_safety` of the floor.
                f32::NAN
            } else {
                let h = r_slant * lut.cos_elev[k];
                if mount.min_range_m > 0.0 && h < mount.min_range_m {
                    f32::NAN
                } else {
                    h
                }
            };
            k += 1;
        }
    }
    (angles, ranges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_mount_centre_zone_points_forward() {
        let m = TofMount::default();
        let lut = precompute_zone_lookups(&m);
        // The centre-of-image zone (row 3, col 3 OR row 4, col 4 — the four
        // central zones span the centre) should have body azimuth near 0
        // and cos_elev near 1.
        for &k in &[(3, 3), (4, 4)] {
            let (r, c) = k;
            let idx = r * TOF_COLS + c;
            assert!(lut.angles_body[idx].abs() < 0.05);
            assert!(lut.cos_elev[idx] > 0.998);
        }
    }

    #[test]
    fn pitch_down_makes_bottom_rows_look_below() {
        let m = TofMount { pitch_rad: 10.0_f32.to_radians(), ..Default::default() };
        let lut = precompute_zone_lookups(&m);
        // Row 7 (bottom) should have the largest sin_below.
        let s_top    = lut.sin_below[0 * TOF_COLS + 3];   // row 0, mid col
        let s_bottom = lut.sin_below[7 * TOF_COLS + 3];   // row 7, mid col
        assert!(s_bottom > 0.4);
        assert!(s_top    < 0.0);   // top row actually looks slightly UP
    }

    #[test]
    fn floor_filter_drops_steep_short_returns() {
        let m = TofMount {
            pitch_rad: 10.0_f32.to_radians(),
            sensor_height_m: 0.10,
            floor_safety: 0.85,
            ..Default::default()
        };
        let lut = precompute_zone_lookups(&m);
        let mut raw = [[1.5_f32; 8]; 8];
        // Row 7: pretend the chip returns 0.3 m (a floor hit at our height).
        raw[7] = [0.30; 8];
        let (_, ranges) = project_frame(&raw, &lut, &m);
        // Row 7 entries should be NaN (filtered), row 0 entries should be finite.
        for c in 0..8 {
            assert!(ranges[7 * 8 + c].is_nan(),
                    "row 7 col {} expected NaN, got {}", c, ranges[7 * 8 + c]);
            assert!(ranges[0 * 8 + c].is_finite());
        }
    }
}
