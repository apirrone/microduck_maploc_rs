//! Scan accumulator — buffers ToF beams across motion so MCL can run
//! a single wide-FoV update from multiple viewpoints. Matches the
//! Python reference in `microduck_maploc/sim/accumulator.py`.

use std::f32::consts::PI;

/// One buffered beam, tagged with the localizer's belief at capture
/// time. Used by `Localizer::update_accumulated` /
/// `global_relocalize_field` to compute particle pose-at-capture under
/// the constant-offset approximation.
#[derive(Debug, Clone, Copy)]
pub struct BufferedBeam {
    pub angle_body: f32,
    pub range_m:    f32,
    pub est_origin: (f32, f32),
    pub est_yaw:    f32,
}

#[derive(Debug, Clone, Copy)]
pub struct AccumulatorConfig {
    pub max_age_s:                 f32,
    pub min_azimuth_coverage_frac: f32,
    pub max_beams:                 usize,
    pub n_azimuth_bins:            usize,
}

impl Default for AccumulatorConfig {
    fn default() -> Self {
        Self {
            max_age_s:                  2.5,
            min_azimuth_coverage_frac:  0.80,
            max_beams:                  800,
            n_azimuth_bins:             36,
        }
    }
}

pub struct ScanAccumulator {
    cfg:      AccumulatorConfig,
    buffer:   Vec<BufferedBeam>,
    az_bins:  Vec<bool>,
    start_t:  Option<f64>,
}

impl Default for ScanAccumulator {
    fn default() -> Self { Self::new(AccumulatorConfig::default()) }
}

impl ScanAccumulator {
    pub fn new(cfg: AccumulatorConfig) -> Self {
        let bins = vec![false; cfg.n_azimuth_bins];
        Self { cfg, buffer: Vec::new(), az_bins: bins, start_t: None }
    }

    pub fn add_scan(
        &mut self,
        t: f64,
        est_pose: (f32, f32, f32),
        angles_body: &[f32],
        ranges: &[f32],
    ) {
        let (ex, ey, eyaw) = est_pose;
        if self.start_t.is_none() { self.start_t = Some(t); }
        for (a, r) in angles_body.iter().zip(ranges.iter()) {
            if !r.is_finite() { continue; }
            self.buffer.push(BufferedBeam {
                angle_body: *a, range_m: *r,
                est_origin: (ex, ey), est_yaw: eyaw,
            });
            let world_az = ((eyaw + a).rem_euclid(2.0 * PI)) as f64;
            let n = self.cfg.n_azimuth_bins;
            let bin = ((world_az / (2.0 * std::f64::consts::PI)) * n as f64) as usize;
            if bin < n { self.az_bins[bin] = true; }
        }
    }

    pub fn coverage_frac(&self) -> f32 {
        let n = self.cfg.n_azimuth_bins as f32;
        let on = self.az_bins.iter().filter(|b| **b).count() as f32;
        on / n
    }

    pub fn n_beams(&self) -> usize { self.buffer.len() }

    pub fn ready(&self, t: f64) -> bool {
        let Some(start) = self.start_t else { return false; };
        if self.buffer.is_empty() { return false; }
        if (t - start) >= self.cfg.max_age_s as f64 { return true; }
        if self.buffer.len() >= self.cfg.max_beams { return true; }
        if self.coverage_frac() >= self.cfg.min_azimuth_coverage_frac { return true; }
        false
    }

    pub fn drain(&mut self) -> Vec<BufferedBeam> {
        let out = std::mem::take(&mut self.buffer);
        for b in self.az_bins.iter_mut() { *b = false; }
        self.start_t = None;
        out
    }
}
