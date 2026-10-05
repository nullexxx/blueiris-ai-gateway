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
TRACKER_HYBRID_CHASE_MAX_SECONDS=2.50
TRACKER_HYBRID_CHASE_COOLDOWN=0.35
TRACKER_HYBRID_CHASE_SETTLE_FRAMES=2
```

Edge rescue remains a positional `moveDirectly` correction and is configured separately with `TRACKER_EDGE_RESCUE_ENABLED`, `TRACKER_EDGE_RESCUE_ERROR`, and `TRACKER_EDGE_RESCUE_GAIN`.

Chase can start from a hard edge/error condition or from sufficiently fast outward target motion. It stops before center, on direction reversal, low confidence, target loss beyond the blur grace period, stale/unavailable frames, inference errors, maximum chase duration, tracker stop/shutdown, or home commands. Camera-side timeout is an additional fail-safe.

Useful diagnostics are `control_mode=hybrid`, `hybrid_chase_active`, `hybrid_chase_speed`, and the `hybrid_chase_*` history events.
