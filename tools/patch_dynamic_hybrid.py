from pathlib import Path

p = Path("tracker.py")
s = p.read_text()


def replace_once(old: str, new: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one match, found {count}: {old[:120]!r}")
    s = s.replace(old, new, 1)


replace_once(
    "    hybrid_chase_entry_error: float = 0.82\n    hybrid_chase_exit_error: float = 0.42\n    hybrid_chase_min_speed: int = 1\n",
    "    hybrid_chase_entry_error: float = 0.82\n    hybrid_chase_exit_error: float = 0.42\n    # Fast-moving targets may enter chase before the hard edge threshold.\n    hybrid_chase_motion_error: float = 0.50\n    hybrid_chase_motion_speed_norm: float = 0.04\n    # Coast through very short detector dropouts caused by PTZ motion blur.\n    hybrid_chase_miss_grace: float = 0.25\n    hybrid_chase_min_speed: int = 1\n",
)

replace_once(
    '            hybrid_chase_entry_error=_env_float("TRACKER_HYBRID_CHASE_ENTRY_ERROR", 0.82),\n            hybrid_chase_exit_error=_env_float("TRACKER_HYBRID_CHASE_EXIT_ERROR", 0.42),\n            hybrid_chase_min_speed=_env_int("TRACKER_HYBRID_CHASE_MIN_SPEED", 1),\n',
    '            hybrid_chase_entry_error=_env_float("TRACKER_HYBRID_CHASE_ENTRY_ERROR", 0.82),\n            hybrid_chase_exit_error=_env_float("TRACKER_HYBRID_CHASE_EXIT_ERROR", 0.42),\n            hybrid_chase_motion_error=_env_float("TRACKER_HYBRID_CHASE_MOTION_ERROR", 0.50),\n            hybrid_chase_motion_speed_norm=_env_float("TRACKER_HYBRID_CHASE_MOTION_SPEED_NORM", 0.04),\n            hybrid_chase_miss_grace=_env_float("TRACKER_HYBRID_CHASE_MISS_GRACE", 0.25),\n            hybrid_chase_min_speed=_env_int("TRACKER_HYBRID_CHASE_MIN_SPEED", 1),\n',
)

replace_once(
    "        cfg.hybrid_chase_exit_error = max(0.20, min(0.65, cfg.hybrid_chase_exit_error))\n        cfg.hybrid_chase_entry_error = max(cfg.hybrid_chase_exit_error + 0.10, min(0.98, cfg.hybrid_chase_entry_error))\n        cfg.hybrid_chase_min_speed = max(1, min(8, cfg.hybrid_chase_min_speed))\n",
    "        cfg.hybrid_chase_exit_error = max(0.20, min(0.65, cfg.hybrid_chase_exit_error))\n        cfg.hybrid_chase_entry_error = max(cfg.hybrid_chase_exit_error + 0.10, min(0.98, cfg.hybrid_chase_entry_error))\n        cfg.hybrid_chase_motion_error = max(cfg.hybrid_chase_exit_error, min(cfg.hybrid_chase_entry_error, cfg.hybrid_chase_motion_error))\n        cfg.hybrid_chase_motion_speed_norm = max(0.005, min(1.0, cfg.hybrid_chase_motion_speed_norm))\n        cfg.hybrid_chase_miss_grace = max(0.0, min(0.75, cfg.hybrid_chase_miss_grace))\n        cfg.hybrid_chase_min_speed = max(1, min(8, cfg.hybrid_chase_min_speed))\n",
)

replace_once(
    '            "hybrid_chase_entry_error": self.hybrid_chase_entry_error,\n            "hybrid_chase_exit_error": self.hybrid_chase_exit_error,\n            "hybrid_chase_min_speed": self.hybrid_chase_min_speed,\n',
    '            "hybrid_chase_entry_error": self.hybrid_chase_entry_error,\n            "hybrid_chase_exit_error": self.hybrid_chase_exit_error,\n            "hybrid_chase_motion_error": self.hybrid_chase_motion_error,\n            "hybrid_chase_motion_speed_norm": self.hybrid_chase_motion_speed_norm,\n            "hybrid_chase_miss_grace": self.hybrid_chase_miss_grace,\n            "hybrid_chase_min_speed": self.hybrid_chase_min_speed,\n',
)

replace_once(
    '        if self._hybrid_chase_active:\n            await self._stop_hybrid_chase("target_missing", seq=seq)\n\n        # A temporary detector miss while the camera is moving is not evidence that\n',
    '        if self._hybrid_chase_active:\n            # Keep the bounded chase alive across one or two blurred YOLO misses.\n            hybrid_missing_for = max(0.0, now - self.target.last_seen)\n            if hybrid_missing_for <= self.cfg.hybrid_chase_miss_grace:\n                self.state = "ESCAPE_CHASE"\n                return\n            await self._stop_hybrid_chase("target_missing", seq=seq)\n\n        # A temporary detector miss while the camera is moving is not evidence that\n',
)

old = '''            dominant_error = max(abs(err_x), abs(err_y))
            hard_escape_error = max(0.90, self.cfg.hybrid_chase_entry_error + 0.08)
            hybrid_entry = (
                self.cfg.hybrid_chase_enabled
                and (now - self._hybrid_last_stopped_at) >= self.cfg.hybrid_chase_cooldown
                and (
                    (edge_clipped_now and dominant_error >= self.cfg.hybrid_chase_entry_error)
                    or dominant_error >= hard_escape_error
                )
            )
            if hybrid_entry:
'''
new = '''            dominant_error = max(abs(err_x), abs(err_y))
            hard_escape_error = max(0.90, self.cfg.hybrid_chase_entry_error + 0.08)

            # Dynamic handoff: keep moveDirectly for slow/stable motion, but a
            # fast target that is already well outside center and still moving
            # outward should not be committed to another ~1.5s positional move.
            frame_diag = max(1.0, math.hypot(w, h))
            target_speed_norm = (
                math.hypot(self.target.vx, self.target.vy) / frame_diag
                if self.target.velocity_valid
                else 0.0
            )
            moving_outward = self.target.velocity_valid and (
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
            if hybrid_entry:
'''
replace_once(old, new)

old = '''                    "hybrid_chase_enter",
                    label=self.target.label,
                    error_x=round(err_x, 3),
                    error_y=round(err_y, 3),
                    edge_clipped=edge_clipped_now,
                    pan_speed=pan_speed,
'''
new = '''                    "hybrid_chase_enter",
                    label=self.target.label,
                    entry_reason=(
                        "motion_escape" if motion_escape else
                        ("edge_clipped" if edge_clipped_now else "hard_escape")
                    ),
                    error_x=round(err_x, 3),
                    error_y=round(err_y, 3),
                    target_speed_norm=round(target_speed_norm, 4),
                    moving_outward=bool(moving_outward),
                    edge_clipped=edge_clipped_now,
                    pan_speed=pan_speed,
'''
replace_once(old, new)

p.write_text(s)
