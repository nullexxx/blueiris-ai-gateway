from pathlib import Path

p = Path("tracker.py")
s = p.read_text()


def r(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"expected one match, found {n}: {old[:120]!r}")
    s = s.replace(old, new, 1)


# Make fast chase actually fast enough to catch an escaping target while still
# stopping early in the inner region to avoid the old continuous-mode oscillation.
r("    hybrid_chase_motion_error: float = 0.50\n", "    hybrid_chase_motion_error: float = 0.45\n")
r("    hybrid_chase_motion_speed_norm: float = 0.04\n", "    hybrid_chase_motion_speed_norm: float = 0.03\n")
r("    hybrid_chase_miss_grace: float = 0.25\n", "    hybrid_chase_miss_grace: float = 0.60\n")
r("    hybrid_chase_max_speed: int = 6\n", "    hybrid_chase_max_speed: int = 8\n")
r("    hybrid_chase_full_speed_error: float = 0.95\n", "    hybrid_chase_full_speed_error: float = 0.82\n")

r('_env_float("TRACKER_HYBRID_CHASE_MOTION_ERROR", 0.50)', '_env_float("TRACKER_HYBRID_CHASE_MOTION_ERROR", 0.45)')
r('_env_float("TRACKER_HYBRID_CHASE_MOTION_SPEED_NORM", 0.04)', '_env_float("TRACKER_HYBRID_CHASE_MOTION_SPEED_NORM", 0.03)')
r('_env_float("TRACKER_HYBRID_CHASE_MISS_GRACE", 0.25)', '_env_float("TRACKER_HYBRID_CHASE_MISS_GRACE", 0.60)')
r('_env_int("TRACKER_HYBRID_CHASE_MAX_SPEED", 6)', '_env_int("TRACKER_HYBRID_CHASE_MAX_SPEED", 8)')
r('_env_float("TRACKER_HYBRID_CHASE_FULL_SPEED_ERROR", 0.95)', '_env_float("TRACKER_HYBRID_CHASE_FULL_SPEED_ERROR", 0.82)')

old = '''            moving_outward = self.target.velocity_valid and (
                (abs(err_x) >= self.cfg.hybrid_chase_motion_error and err_x * self.target.vx > 0.0)
                or (abs(err_y) >= self.cfg.hybrid_chase_motion_error and err_y * self.target.vy > 0.0)
            )
            motion_escape = (
                dominant_error >= self.cfg.hybrid_chase_motion_error
                and target_speed_norm >= self.cfg.hybrid_chase_motion_speed_norm
                and moving_outward
            )
            hybrid_entry = (
                self.cfg.hybrid_chase_enabled
                and (now - self._hybrid_last_stopped_at) >= self.cfg.hybrid_chase_cooldown
                and (
                    (edge_clipped_now and dominant_error >= self.cfg.hybrid_chase_entry_error)
                    or dominant_error >= hard_escape_error
                    or motion_escape
                )
            )
'''
new = '''            moving_outward = self.target.velocity_valid and (
                (abs(err_x) >= self.cfg.hybrid_chase_motion_error and err_x * self.target.vx > 0.0)
                or (abs(err_y) >= self.cfg.hybrid_chase_motion_error and err_y * self.target.vy > 0.0)
            )
            if abs(err_x) >= abs(err_y):
                dominant_axis_error = err_x
                dominant_axis_velocity = self.target.vx
            else:
                dominant_axis_error = err_y
                dominant_axis_velocity = self.target.vy
            moving_inward_dominant = self.target.velocity_valid and (
                abs(dominant_axis_error) >= self.cfg.hybrid_chase_exit_error
                and dominant_axis_error * dominant_axis_velocity < 0.0
                and abs(dominant_axis_velocity) / frame_diag >= self.cfg.hybrid_chase_motion_speed_norm
            )
            motion_escape = (
                dominant_error >= self.cfg.hybrid_chase_motion_error
                and target_speed_norm >= self.cfg.hybrid_chase_motion_speed_norm
                and moving_outward
            )
            hybrid_entry = (
                self.cfg.hybrid_chase_enabled
                and (now - self._hybrid_last_stopped_at) >= self.cfg.hybrid_chase_cooldown
                and (
                    (edge_clipped_now and dominant_error >= self.cfg.hybrid_chase_entry_error and not moving_inward_dominant)
                    or (dominant_error >= hard_escape_error and not moving_inward_dominant)
                    or motion_escape
                )
            )
'''
r(old, new)

r(
    '''                    target_speed_norm=round(target_speed_norm, 4),
                    moving_outward=bool(moving_outward),
                    edge_clipped=edge_clipped_now,
''',
    '''                    target_speed_norm=round(target_speed_norm, 4),
                    moving_outward=bool(moving_outward),
                    moving_inward_dominant=bool(moving_inward_dominant),
                    edge_clipped=edge_clipped_now,
''',
)

old = '''            edge_rescue_active = self.cfg.edge_rescue_enabled and (
                edge_clipped
                or abs(err_x) >= self.cfg.edge_rescue_error
                or abs(err_y) >= self.cfg.edge_rescue_error
            )
            edge_rescue_reason: Optional[str] = None
            if edge_rescue_active:
                edge_rescue_reason = "edge_clipped" if edge_clipped else "extreme_error"
'''
new = '''            edge_rescue_requested = self.cfg.edge_rescue_enabled and (
                edge_clipped
                or abs(err_x) >= self.cfg.edge_rescue_error
                or abs(err_y) >= self.cfg.edge_rescue_error
            )
            edge_rescue_active = edge_rescue_requested and not moving_inward_dominant
            edge_rescue_reason: Optional[str] = None
            if edge_rescue_active:
                edge_rescue_reason = "edge_clipped" if edge_clipped else "extreme_error"
            elif edge_rescue_requested and moving_inward_dominant:
                edge_rescue_reason = "inward_motion_suppressed"
'''
r(old, new)

p.write_text(s)
