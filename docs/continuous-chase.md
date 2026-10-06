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
TRACKER_HYBRID_CHASE_MIN_SPEED=1
TRACKER_HYBRID_CHASE_MAX_SPEED=6
TRACKER_HYBRID_CHASE_FULL_SPEED_ERROR=0.90
TRACKER_HYBRID_CHASE_COMMAND_INTERVAL=0.18
TRACKER_HYBRID_CHASE_KEEPALIVE=0.45
TRACKER_HYBRID_CHASE_CAMERA_TIMEOUT=1
TRACKER_HYBRID_CHASE_MAX_SECONDS=2.50  # native-discrete rescue only
TRACKER_SERVO_LARGE_ERROR_START=0.45
TRACKER_SERVO_LARGE_ERROR_FULL=0.75
TRACKER_SERVO_LARGE_ERROR_GAIN=1.45
TRACKER_HYBRID_CHASE_COOLDOWN=0.35
TRACKER_HYBRID_CHASE_SETTLE_FRAMES=2
```

Edge rescue remains a positional `moveDirectly` correction and is configured separately with `TRACKER_EDGE_RESCUE_ENABLED`, `TRACKER_EDGE_RESCUE_ERROR`, and `TRACKER_EDGE_RESCUE_GAIN`.

Chase can start from a hard edge/error condition or from sufficiently fast outward target motion. The calibrated fractional ONVIF actuator is a closed-loop feedback servo and is not stopped merely because 2.5 seconds elapsed; the duration watchdog remains for coarse native-discrete rescue. An established chase uses `TRACKER_HOLD_CONF` for confidence gating, while `TRACKER_REACQUIRE_CONF` remains reserved for reacquisition. The fractional servo progressively increases only its proportional term for large errors (default 1.0× through 0.45 error, reaching 1.45× at 0.75), while derivative damping, feed-forward, braking, calibration mapping, and the ONVIF velocity ceiling remain unchanged. Chase still stops before center, on genuine low confidence, target loss beyond the blur grace period, stale/unavailable frames, inference errors, divergence, tracker stop/shutdown, or home commands. Camera-side timeout is an additional fail-safe.

Useful diagnostics are `control_mode=hybrid`, `hybrid_chase_active`, `hybrid_chase_speed`, and the `hybrid_chase_*` history events.
