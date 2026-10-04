# Continuous PTZ Chase Controller

The PTZ tracker now uses Dahua/Amcrest `Continuously` pan/tilt motor control for active pursuit instead of `moveDirectly` positional moves.

This mode is intended for targets that can change direction quickly, such as dogs. The target's normalized image error is converted directly into signed pan/tilt motor speed. Pan and tilt are controlled independently.

Recommended starting values:

```env
TRACKER_CONTINUOUS_CHASE_ENABLED=true
TRACKER_FPS=15
TRACKER_MOVE_DEADZONE_X=0.14
TRACKER_MOVE_DEADZONE_Y=0.18
TRACKER_CONTINUOUS_MIN_SPEED=5
TRACKER_CONTINUOUS_MAX_SPEED=8
TRACKER_CONTINUOUS_FULL_SPEED_ERROR=0.60
TRACKER_CONTINUOUS_COMMAND_INTERVAL=0.15
TRACKER_CONTINUOUS_KEEPALIVE=0.45
TRACKER_CONTINUOUS_CAMERA_TIMEOUT=1
```

The controller stops continuous motion when the target enters the deadzone, when the target is lost, when frames become stale/unavailable, on inference errors, and when tracking is stopped, shut down, or sent home. The camera-side timeout is an additional fail-safe.

Home preset and bounded optical auto-zoom remain discrete/status-driven operations. Auto-zoom runs only while continuous pan/tilt is stopped and the target is centered.

Target acquisition now requires every confirmation frame to meet `TRACKER_ACQUIRE_CONF`; the lower hold threshold applies only after a target is fully acquired.

Useful diagnostics:

- `control_mode` should report `continuous`.
- `continuous_active` shows whether the motors are currently being driven.
- `continuous_speed` reports current signed `[pan, tilt]` speeds.
- History records `continuous_move` and `continuous_stop` events with errors, speeds, confidence, and CGI latency.
