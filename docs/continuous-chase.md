# Hybrid PTZ Escape-Chase Controller

Normal tracking uses Dahua/Amcrest `moveDirectly` positional moves. Continuous pan/tilt is reserved for a bounded escape mode when a fast target is genuinely in danger of leaving the frame.

Current controls:

```env
TRACKER_HYBRID_CHASE_ENABLED=true
TRACKER_HYBRID_CHASE_PAN_SIGN=-1
TRACKER_HYBRID_CHASE_ENTRY_ERROR=0.82
TRACKER_HYBRID_CHASE_EXIT_ERROR=0.50
TRACKER_HYBRID_CHASE_MOTION_ERROR=0.55
TRACKER_HYBRID_CHASE_MOTION_SPEED_NORM=0.03
TRACKER_HYBRID_CHASE_MISS_GRACE=0.15
TRACKER_HYBRID_MISSING_GRACE=0.50
TRACKER_HYBRID_CHASE_MIN_SPEED=1
TRACKER_HYBRID_CHASE_MAX_SPEED=6
TRACKER_HYBRID_CHASE_FULL_SPEED_ERROR=0.90
TRACKER_HYBRID_CHASE_COMMAND_INTERVAL=0.18
TRACKER_HYBRID_CHASE_KEEPALIVE=0.45
TRACKER_HYBRID_CHASE_CAMERA_TIMEOUT=1
TRACKER_HYBRID_CHASE_MAX_SECONDS=2.50  # native-discrete rescue only
TRACKER_SERVO_BRAKE_HORIZON=0.30
TRACKER_SERVO_BRAKE_VELOCITY_EXTENSION=0.08
TRACKER_SERVO_LARGE_ERROR_START=0.45
TRACKER_SERVO_LARGE_ERROR_FULL=0.75
TRACKER_SERVO_LARGE_ERROR_GAIN=1.45
TRACKER_SERVO_OUTWARD_CATCHUP_START=0.60
TRACKER_SERVO_OUTWARD_CATCHUP_FULL=0.85
TRACKER_SERVO_OUTWARD_CATCHUP_GAIN=1.30
TRACKER_SERVO_HOLD_SECONDS=0.30
TRACKER_SERVO_HOLD_MIN_FRAMES=3
TRACKER_SERVO_HOLD_RESUME_GROWTH=0.06
TRACKER_CONFIDENCE_COAST_START_SCALE=0.70
TRACKER_CONFIDENCE_COAST_END_SCALE=0.20
TRACKER_HYBRID_CHASE_COOLDOWN=0.35
TRACKER_HYBRID_CHASE_SETTLE_FRAMES=2
```

Edge rescue remains a positional `moveDirectly` correction and is configured separately with `TRACKER_EDGE_RESCUE_ENABLED`, `TRACKER_EDGE_RESCUE_ERROR`, and `TRACKER_EDGE_RESCUE_GAIN`.

Chase can start from a hard edge/error condition or from sufficiently fast outward target motion. The calibrated fractional ONVIF actuator is a closed-loop feedback servo and is not stopped merely because 2.5 seconds elapsed; the duration watchdog remains for coarse native-discrete rescue. An established chase uses `TRACKER_HOLD_CONF` for confidence gating, while `TRACKER_REACQUIRE_CONF` remains reserved for reacquisition. The fractional servo progressively increases only its proportional term for large errors (default 1.0× through 0.45 error, reaching 1.45× at 0.75), while derivative damping, feed-forward, braking, calibration mapping, and the ONVIF velocity ceiling remain unchanged. Chase still stops before center, on genuine low confidence, target loss beyond the blur grace period, stale/unavailable frames, inference errors, divergence, tracker stop/shutdown, or home commands. Camera-side timeout is an additional fail-safe.

Useful diagnostics are `control_mode=hybrid`, `hybrid_chase_active`, `hybrid_chase_speed`, and the `hybrid_chase_*` history events.


## Rev 6.3 refinements

Rev 6.3 retains the Rev 6.2 proportional boost but starts braking sooner as current fractional ONVIF velocity rises. The default brake horizon is 0.30 seconds plus up to 0.08 seconds of velocity-dependent extension. Active chase detector misses receive 0.50 seconds before the continuous actuator is stopped, and a lost target remains eligible for reacquisition for 5.0 seconds before release/home.

Association failures are attached to `target_missing` / `target_loss_paused` history and surfaced as `last_association_diagnostic` in tracker status. Diagnostics include detection count, same-class count, the nearest same-class candidate's normalized distance/IoU/size similarity/confidence, whether it passed the distance gate, and confidence-gate rejection details.


## Rev 6.4 refinements

Rev 6.4 preserves the Rev 6.3 braking and loss windows while improving pursuit continuity. Large errors that are still growing receive a progressive outward-only desired-rate multiplier up to 1.30x. Marginal-confidence detections and complete detector misses inside their existing grace windows use a decaying snapshot of the last verified fractional velocity, so they cannot accelerate or reverse the camera. A transient predicted stop now enters a 0.30-second servo hold for at least three observations; error growth of 0.06 immediately resumes the chase, otherwise the hold settles into a normal stop.
