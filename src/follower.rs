//! Waypoint follower: turn-then-go controller on top of a planned path.
//!
//! Mirrors `microduck_maploc/sim/streaming.py::follow_step`. Inputs are
//! the **estimated** pose (from the localizer) — that's all the duck has
//! on hardware. The caller applies the returned `forward` distance using
//! the *true* body heading: motors act in the actual body frame, not
//! the estimated one. If the localizer is wrong, the duck physically
//! drifts off the planned path — that's the honest behaviour we want.

#[derive(Debug, Clone, Default)]
pub struct FollowerState {
    waypoints: Vec<(f32, f32)>,
    idx:       usize,
}

impl FollowerState {
    pub fn new(waypoints: Vec<(f32, f32)>) -> Self {
        Self { waypoints, idx: 0 }
    }
    pub fn empty() -> Self { Self::default() }
    pub fn done(&self) -> bool { self.idx >= self.waypoints.len() }
    pub fn current(&self) -> Option<(f32, f32)> {
        self.waypoints.get(self.idx).copied()
    }
    pub fn waypoints(&self) -> &[(f32, f32)] { &self.waypoints }
}

/// Body-frame command produced by one follower tick.
#[derive(Debug, Clone, Copy, Default)]
pub struct FollowCommand {
    /// Forward distance to translate this tick (≥ 0). Apply along the
    /// **true** body yaw on the actuator side.
    pub forward: f32,
    /// Yaw delta to apply this tick (signed). Applied directly.
    pub dyaw:    f32,
}

/// Advance one tick toward the current waypoint. `arrive_radius`
/// determines when a waypoint is "reached" and the follower advances.
pub fn follow_step(
    state: &mut FollowerState,
    est_pos: (f32, f32), est_yaw: f32,
    lin_speed: f32, yaw_speed: f32, dt: f32,
    arrive_radius: f32,
) -> FollowCommand {
    let Some((tx, ty)) = state.current() else {
        return FollowCommand::default();
    };
    let dx = tx - est_pos.0;
    let dy = ty - est_pos.1;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist < arrive_radius {
        state.idx += 1;
        return FollowCommand::default();
    }
    let target_yaw = dy.atan2(dx);
    let mut yaw_err = target_yaw - est_yaw;
    while yaw_err >  std::f32::consts::PI { yaw_err -= 2.0 * std::f32::consts::PI; }
    while yaw_err < -std::f32::consts::PI { yaw_err += 2.0 * std::f32::consts::PI; }
    let dyaw = yaw_err.clamp(-yaw_speed * dt, yaw_speed * dt);
    // Turn-then-go: only translate when roughly aimed at the target.
    if yaw_err.abs() > 0.30 {
        return FollowCommand { forward: 0.0, dyaw };
    }
    let forward = (lin_speed * dt).min(dist);
    FollowCommand { forward, dyaw }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turns_first_then_drives() {
        // Waypoint due north of (0, 0); duck initially facing east (yaw=0).
        let mut s = FollowerState::new(vec![(0.0, 1.0)]);
        let cmd = follow_step(&mut s, (0.0, 0.0), 0.0,
                              0.5, 1.0, 0.1, 0.05);
        // First tick: large yaw error → only turn.
        assert!(cmd.forward.abs() < 1e-6, "should not translate yet, got {}", cmd.forward);
        assert!(cmd.dyaw > 0.0, "should turn CCW toward +y, got {}", cmd.dyaw);
    }

    #[test]
    fn translates_when_aimed() {
        let mut s = FollowerState::new(vec![(1.0, 0.0)]);
        let cmd = follow_step(&mut s, (0.0, 0.0), 0.0,
                              0.5, 1.0, 0.1, 0.05);
        assert!(cmd.forward > 0.0, "should translate forward; got {}", cmd.forward);
    }

    #[test]
    fn arrives_and_advances() {
        let mut s = FollowerState::new(vec![(0.01, 0.0), (1.0, 0.0)]);
        let cmd = follow_step(&mut s, (0.0, 0.0), 0.0,
                              0.5, 1.0, 0.1, 0.05);
        // Inside arrive_radius (0.05) on first waypoint, idx advances.
        assert!(cmd.forward.abs() < 1e-6 && cmd.dyaw.abs() < 1e-6);
        assert_eq!(s.idx, 1);
        assert!(!s.done());
    }
}
