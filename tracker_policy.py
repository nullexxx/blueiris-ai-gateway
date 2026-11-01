from __future__ import annotations

import math
from typing import Dict, Tuple


def _servo_axis_decision(
    *,
    error: float,
    error_rate: float,
    feedforward_rate: float,
    exit_error: float,
    kp: float,
    kd: float,
    feedforward_gain: float,
    brake_horizon_s: float,
) -> Dict[str, float | bool | str]:
    """Rev 6 proportional+damped image-space servo decision for one axis.

    error_rate is measured from successive tracked bbox centers while the
    camera is moving. That captures both subject motion and the camera's real
    mechanical response, so braking follows what the image is actually doing.
    """
    error = float(error)
    error_rate = float(error_rate)
    exit_error = max(0.01, abs(float(exit_error)))
    predicted_error = error + error_rate * max(0.0, float(brake_horizon_s))
    # Positive means the tracked bbox is moving toward center. Rev 5 used
    # copysign(error_rate, error), which discarded the sign of error_rate and
    # could report an outward-moving negative error as "closing".
    closing_rate = (
        0.0
        if abs(error) < 1e-6
        else -error_rate * math.copysign(1.0, error)
    )
    p_term = float(kp) * error
    d_term = float(kd) * error_rate
    ff_term = float(feedforward_gain) * float(feedforward_rate)
    raw_rate = p_term + d_term + ff_term

    phase = "track"
    brake = False
    if abs(error) <= exit_error:
        brake = True
        phase = "deadband"
    elif error * predicted_error <= 0.0 or abs(predicted_error) <= exit_error * 0.90:
        brake = True
        phase = "predicted_stop"
    elif error * raw_rate <= 0.0:
        brake = True
        phase = "closing_fast"
    elif closing_rate > 0.0 and abs(predicted_error) < abs(error) * 0.55:
        phase = "brake"

    desired_rate = 0.0 if brake else raw_rate
    return {
        "desired_rate": desired_rate,
        "raw_rate": raw_rate,
        "predicted_error": predicted_error,
        "closing_rate": closing_rate,
        "p_term": p_term,
        "d_term": d_term,
        "ff_term": ff_term,
        "brake": brake,
        "phase": phase,
    }


def _servo_large_error_gain(
    error: float,
    *,
    start_error: float,
    full_error: float,
    max_gain: float,
) -> float:
    """Progressively add servo authority only when the target is far off-center."""
    magnitude = abs(float(error))
    start = max(0.0, float(start_error))
    full = max(start + 1e-6, float(full_error))
    ceiling = max(1.0, float(max_gain))
    if magnitude <= start:
        return 1.0
    if magnitude >= full:
        return ceiling
    fraction = (magnitude - start) / (full - start)
    return 1.0 + fraction * (ceiling - 1.0)


def _servo_dynamic_brake_horizon(
    current_velocity: float,
    *,
    max_velocity: float,
    base_horizon_s: float,
    velocity_extension_s: float,
) -> float:
    """Extend braking look-ahead as the fractional actuator moves faster."""
    base = max(0.0, float(base_horizon_s))
    extension = max(0.0, float(velocity_extension_s))
    maximum = max(1e-6, abs(float(max_velocity)))
    fraction = min(1.0, abs(float(current_velocity)) / maximum)
    return base + extension * fraction


def _servo_outward_catchup_gain(
    error: float,
    error_rate: float,
    *,
    start_error: float,
    full_error: float,
    max_gain: float,
) -> float:
    """Boost only a large axis error whose magnitude is still growing."""
    if float(error) * float(error_rate) <= 0.0:
        return 1.0
    return _servo_large_error_gain(
        error,
        start_error=start_error,
        full_error=full_error,
        max_gain=max_gain,
    )


def _servo_coast_scale(
    elapsed_s: float,
    duration_s: float,
    *,
    start_scale: float,
    end_scale: float,
) -> float:
    """Linearly decay a previously verified velocity without changing direction."""
    start = max(0.0, min(1.0, float(start_scale)))
    end = max(0.0, min(start, float(end_scale)))
    duration = max(1e-6, float(duration_s))
    fraction = max(0.0, min(1.0, float(elapsed_s) / duration))
    return start + (end - start) * fraction


def _servo_axis_divergence_count(
    previous_error: float | None,
    current_error: float,
    desired_rate: float,
    previous_count: int,
    growth: float,
    *,
    camera_shift: float | None = None,
    min_camera_shift: float = 0.75,
) -> int:
    """Count only proven wrong-direction camera response while error worsens.

    A fast subject can legitimately make image error grow even while the camera
    is correcting in the right direction. Global background motion is masked
    around detections and gives the missing physical evidence: calibration
    defines image-space correction as -scene_shift, so a correct camera response
    has scene shift opposite the controller's desired image-space rate.
    """
    if previous_error is None:
        return 0
    prev = float(previous_error)
    cur = float(current_error)
    desired = float(desired_rate)
    if abs(desired) <= 1e-6:
        return 0
    if prev == 0.0 or cur == 0.0 or ((prev > 0.0) != (cur > 0.0)):
        return 0
    if cur * desired <= 0.0:
        return 0
    if camera_shift is None:
        return 0
    shift = float(camera_shift)
    if not math.isfinite(shift) or abs(shift) < max(0.0, float(min_camera_shift)):
        return 0
    if shift * desired <= 0.0:
        return 0
    if abs(cur) > abs(prev) + max(0.0, float(growth)):
        return max(0, int(previous_count)) + 1
    if abs(cur) < abs(prev):
        return 0
    return max(0, int(previous_count))


def _servo_hold_action(
    held_error_x: float,
    held_error_y: float,
    current_error_x: float,
    current_error_y: float,
    *,
    exit_error: float,
    resume_growth: float,
    elapsed_s: float,
    hold_seconds: float,
    frames: int,
    min_frames: int,
    projected_error_x: float | None = None,
    projected_error_y: float | None = None,
    max_hold_seconds: float | None = None,
) -> str:
    """Keep chase identity unless the target is genuinely settled near center.

    Rev 6.6.1 adds a motion-aware hold guard. A target that is momentarily
    centered but projected to leave the continuous-exit region remains in the
    paused chase instead of paying a full stop/settle/velocity-relearn cycle.
    """
    hx, hy = float(held_error_x), float(held_error_y)
    cx, cy = float(current_error_x), float(current_error_y)
    exit_mag = max(0.0, float(exit_error))
    crossed = (
        (hx * cx < 0.0 and abs(cx) > exit_mag)
        or (hy * cy < 0.0 and abs(cy) > exit_mag)
    )
    if crossed:
        return "resume_center_cross"
    held_dom = max(abs(hx), abs(hy))
    current_dom = max(abs(cx), abs(cy))
    if current_dom >= held_dom + max(0.0, float(resume_growth)):
        return "resume_growth"
    if float(elapsed_s) < float(hold_seconds) or int(frames) < int(min_frames):
        return "wait"

    px = cx if projected_error_x is None else float(projected_error_x)
    py = cy if projected_error_y is None else float(projected_error_y)
    projected_dom = max(abs(px), abs(py))
    hold_limit = (
        float(hold_seconds)
        if max_hold_seconds is None
        else max(float(hold_seconds), float(max_hold_seconds))
    )
    if (
        current_dom <= exit_mag
        and projected_dom > exit_mag
        and float(elapsed_s) < hold_limit
    ):
        return "wait_motion"
    if current_dom <= exit_mag:
        return "settled"
    return "resume_outside_exit"


def _native_edge_rescue_axis_requested(
    error: float,
    error_rate: float,
    predicted_error: float,
    onvif_velocity: float | None,
    *,
    max_velocity: float,
    start_error: float,
    saturation: float,
    full_error: float,
) -> bool:
    """Request native rescue only when a near-edge axis is still escaping."""
    err = float(error)
    rate = float(error_rate)
    if abs(err) < float(start_error) or err * rate <= 0.0:
        return False
    velocity = 0.0 if onvif_velocity is None else abs(float(onvif_velocity))
    saturated = velocity >= max(1e-6, abs(float(max_velocity))) * float(saturation)
    projected_edge = abs(float(predicted_error)) >= float(full_error)
    return saturated or projected_edge


def _native_edge_rescue_speed(
    error: float,
    *,
    start_error: float,
    full_error: float,
    min_speed: int,
    max_speed: int,
) -> int:
    """Use calibrated mid-speed rescue until the target reaches the extreme edge."""
    magnitude = abs(float(error))
    if magnitude < float(start_error):
        return 0
    speed = int(max_speed) if magnitude >= float(full_error) else int(min_speed)
    return speed if float(error) > 0.0 else -speed


def _motion_control_decision(
    *,
    err_x,
    err_y,
    vx,
    vy,
    velocity_valid,
    velocity_sample_ms,
    frame_shape,
    move_eta_s,
    hybrid_enabled,
    hybrid_disabled,
    cooldown_ready,
    edge_clipped,
    exit_error,
    entry_error,
    motion_error,
    motion_speed_norm,
    min_sample_ms,
    deadline_travel_norm,
    stationary_speed_norm,
    moving_error,
    fast_follow_speed_norm=0.10,
    fast_follow_horizon_s=0.45,
    fast_follow_error=0.22,
):
    """Choose continuous chase vs slow/stationary precision positioning.

    Rev 6 keeps moveDirectly as a precision controller, not a predictor. A
    moving target either receives continuous PTZ (when sufficiently displaced)
    or is intentionally held until its motion settles. This avoids committing
    the camera to a 1.3-1.8 second move based on a target that will be somewhere
    else when the move completes.
    """
    h, w = frame_shape[:2]
    half_w = max(1.0, float(w) / 2.0)
    half_h = max(1.0, float(h) / 2.0)
    frame_diag = max(1.0, math.hypot(w, h))
    dominant_error = max(abs(err_x), abs(err_y))
    target_speed_norm = math.hypot(vx, vy) / frame_diag if velocity_valid else 0.0
    velocity_mature = bool(velocity_valid and velocity_sample_ms >= min_sample_ms)
    moving_target = bool(
        velocity_mature and target_speed_norm > stationary_speed_norm
    )

    if abs(err_x) >= abs(err_y):
        dominant_axis_error = err_x
        dominant_axis_velocity = vx
    else:
        dominant_axis_error = err_y
        dominant_axis_velocity = vy

    moving_inward_dominant = velocity_mature and (
        abs(dominant_axis_error) >= moving_error
        and dominant_axis_error * dominant_axis_velocity < 0.0
        and abs(dominant_axis_velocity) / frame_diag >= stationary_speed_norm
    )
    moving_outward = velocity_mature and (
        (abs(err_x) >= moving_error and err_x * vx > 0.0)
        or (abs(err_y) >= moving_error and err_y * vy > 0.0)
    )

    projected_travel_norm = 0.0
    future_error_x = float(err_x)
    future_error_y = float(err_y)
    if velocity_mature:
        projected_travel_norm = max(
            abs(vx) * max(0.0, move_eta_s) / half_w,
            abs(vy) * max(0.0, move_eta_s) / half_h,
        )
        future_error_x = float(err_x) + (float(vx) / half_w) * max(0.0, fast_follow_horizon_s)
        future_error_y = float(err_y) + (float(vy) / half_h) * max(0.0, fast_follow_horizon_s)

    fast_x = bool(
        velocity_mature
        and target_speed_norm >= fast_follow_speed_norm
        and (
            (err_x * future_error_x <= 0.0 and abs(future_error_x) >= fast_follow_error)
            or (abs(err_x) >= fast_follow_error and err_x * vx > 0.0)
        )
    )
    fast_y = bool(
        velocity_mature
        and target_speed_norm >= fast_follow_speed_norm
        and (
            (err_y * future_error_y <= 0.0 and abs(future_error_y) >= fast_follow_error)
            or (abs(err_y) >= fast_follow_error and err_y * vy > 0.0)
        )
    )
    fast_predictive_follow = bool(moving_target and (fast_x or fast_y))

    motion_escape = (
        moving_target
        and dominant_error >= moving_error
        and moving_outward
        and target_speed_norm >= min(motion_speed_norm, stationary_speed_norm * 1.25)
    )
    deadline_escape = (
        moving_target
        and dominant_error >= moving_error
        and projected_travel_norm >= deadline_travel_norm
        and not moving_inward_dominant
    )
    hard_escape_error = max(0.90, entry_error + 0.08)
    # A clipped target deserves continuous recovery sooner than the historical
    # emergency-only entry threshold.
    edge_escape = (
        edge_clipped
        and dominant_error >= max(moving_error, min(exit_error, entry_error))
        and not moving_inward_dominant
    )
    hard_escape = dominant_error >= hard_escape_error and not moving_inward_dominant

    use_continuous = bool(
        hybrid_enabled
        and not hybrid_disabled
        and cooldown_ready
        and (fast_predictive_follow or edge_escape or hard_escape or motion_escape or deadline_escape)
    )
    reason = None
    if use_continuous:
        if fast_predictive_follow:
            reason = "fast_predictive_follow"
        elif motion_escape:
            reason = "motion_escape"
        elif deadline_escape:
            reason = "deadline_motion"
        elif edge_escape:
            reason = "edge_clipped"
        else:
            reason = "hard_escape"

    # Rev 6 never falls back to a 1.3-1.8 s positional move just because the
    # continuous path is cooling down or circuit-broken. moveDirectly remains
    # a stationary-target precision tool in every controller state.
    precision_move_allowed = True
    precision_hold_reason = None
    if not velocity_mature:
        precision_move_allowed = False
        precision_hold_reason = "velocity_sample"
    elif moving_target:
        precision_move_allowed = False
        precision_hold_reason = "moving_target"

    return {
        "use_continuous": use_continuous,
        "reason": reason,
        "velocity_mature": velocity_mature,
        "moving_target": moving_target,
        "target_speed_norm": target_speed_norm,
        "projected_travel_norm": projected_travel_norm,
        "future_error_x": future_error_x,
        "future_error_y": future_error_y,
        "fast_predictive_follow": fast_predictive_follow,
        "moving_outward": moving_outward,
        "moving_inward_dominant": moving_inward_dominant,
        "motion_escape": motion_escape,
        "deadline_escape": deadline_escape,
        "precision_move_allowed": precision_move_allowed,
        "precision_hold_reason": precision_hold_reason,
    }

