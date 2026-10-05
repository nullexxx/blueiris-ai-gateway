from pathlib import Path


def scoped(text: str, start_marker: str, end_marker: str):
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return start, end, text[start:end]


def one(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one scoped match, found {count}")
    return text.replace(old, new, 1)


path = Path("tracker.py")
text = path.read_text()

s, e, block = scoped(text, "    async def _calibration_move_directly_sample", "    async def _calibration_continuous_sample")
block = one(
    block,
    '''        requested = math.hypot(err_x, err_y)\n        elapsed = time.monotonic() - started\n        row = {\n''',
    '''        requested = math.hypot(err_x, err_y)\n        command_s = command_done - started\n        settled_s = command_s + float(wait["elapsed_s"])\n        row = {\n''',
    "move settled duration",
)
block = one(
    block,
    '''            "motion_start_ms": None if wait["motion_start_s"] is None else int(wait["motion_start_s"] * 1000),\n            "settled_ms": int(elapsed * 1000),\n''',
    '''            "motion_start_ms": None if wait["motion_start_s"] is None else int((command_s + float(wait["motion_start_s"])) * 1000),\n            "settled_ms": int(settled_s * 1000),\n''',
    "move telemetry duration",
)
block = one(
    block,
    '''            self._update_move_timing_model(requested, elapsed)\n''',
    '''            self._update_move_timing_model(requested, settled_s)\n''',
    "move timing learner duration",
)
text = text[:s] + block + text[e:]

s, e, block = scoped(text, "    async def _calibration_continuous_sample", "    async def _calibration_onvif_benchmark_run")
block = one(
    block,
    '''        desired_sign = raw_sign if correction >= 0 else -raw_sign\n        row = {\n''',
    '''        desired_sign = raw_sign if correction >= 0 else -raw_sign\n        settled_s = (command_done - started) + self._calibration_continuous_duration + float(wait["elapsed_s"])\n        row = {\n''',
    "continuous settled duration",
)
block = one(
    block,
    '''            "settled_ms": int((time.monotonic() - started) * 1000),\n''',
    '''            "settled_ms": int(settled_s * 1000),\n''',
    "continuous timing telemetry",
)
text = text[:s] + block + text[e:]

s, e, block = scoped(text, "    def _startup_calibration_requirements", "    async def _startup_calibration_sequence")
block = one(
    block,
    '''        motion_missing = not self._active_calibration.has_motion_calibration()\n''',
    '''        saved_move = self._active_calibration.move_directly()\n        try:\n            motion_revision = int(saved_move.get("controller_revision", 0) or 0)\n        except (TypeError, ValueError):\n            motion_revision = 0\n        motion_missing = (\n            not self._active_calibration.has_motion_calibration()\n            or motion_revision < 2\n        )\n''',
    "controller calibration revision",
)
text = text[:s] + block + text[e:]

s, e, block = scoped(text, "    async def _calibrate_motion", "    async def calibrate")
block = one(
    block,
    '''        safe_to_track = True\n        try:\n''',
    '''        safe_to_track = True\n        result_payload: Optional[dict] = None\n        try:\n''',
    "motion result payload",
)
block = one(
    block,
    '''            if run_move:\n                for zoom_index, factor in enumerate(self._calibration_zoom_levels):\n''',
    '''            if run_move:\n                self._move_timing_samples.clear()\n                self._move_eta_intercept = max(\n                    self.cfg.move_eta_min,\n                    min(self.cfg.move_eta_max, self.cfg.lead_time),\n                )\n                self._move_eta_slope = 0.0\n                self._move_eta_model_ready = False\n                self._last_predicted_move_eta = self._move_eta_intercept\n                for zoom_index, factor in enumerate(self._calibration_zoom_levels):\n''',
    "reset timing before active calibration",
)
block = one(
    block,
    '''                    pan_scale = max(0.75, min(1.25, 1.0 / max(0.20, pan_ratio)))\n                    tilt_scale = max(0.75, min(1.25, 1.0 / max(0.20, tilt_ratio)))\n''',
    '''                    # Runtime moveDirectly applies cfg.move_gain first. Seed the\n                    # learned multiplier so a stationary target receives about an\n                    # 80% one-step correction. Live moving targets never tune this.\n                    target_fraction = 0.80\n                    pan_scale = max(0.75, min(1.35, target_fraction / max(0.20, self.cfg.move_gain * pan_ratio)))\n                    tilt_scale = max(0.75, min(1.35, target_fraction / max(0.20, self.cfg.move_gain * tilt_ratio)))\n''',
    "active spatial target",
)
block = one(
    block,
    '''                move_result = {\n                    "samples": move_samples,\n''',
    '''                move_result = {\n                    "controller_revision": 2,\n                    "samples": move_samples,\n''',
    "motion controller revision tag",
)
block = one(
    block,
    '''            onvif_ok = (not run_onvif) or isinstance(onvif_result, dict)\n''',
    '''            # ONVIF remains optional for native motion/all calibration. An\n            # explicit mode=onvif request succeeds only if the service is reachable.\n            onvif_ok = mode != "onvif" or bool(isinstance(onvif_result, dict) and onvif_result.get("available"))\n''',
    "onvif benchmark semantics",
)
old_return = '''            return {\n                "success": success,\n                "mode": mode,\n                "move_directly_samples": len(move_samples),\n                "continuous_samples": len(continuous_samples),\n                "move_timing_model": {\n                    "samples": len(self._move_timing_samples),\n                    "ready": self._move_eta_model_ready,\n                    "intercept_s": round(self._move_eta_intercept, 4),\n                    "slope_s_per_norm": round(self._move_eta_slope, 4),\n                },\n                "active_calibration": self._active_calibration.public_dict(),\n                "onvif_benchmark": onvif_result,\n                "safe_to_track": safe_to_track,\n            }\n'''
new_return = '''            result_payload = {\n                "success": success,\n                "mode": mode,\n                "move_directly_samples": len(move_samples),\n                "continuous_samples": len(continuous_samples),\n                "move_timing_model": {\n                    "samples": len(self._move_timing_samples),\n                    "ready": self._move_eta_model_ready,\n                    "intercept_s": round(self._move_eta_intercept, 4),\n                    "slope_s_per_norm": round(self._move_eta_slope, 4),\n                },\n                "active_calibration": self._active_calibration.public_dict(),\n                "onvif_benchmark": onvif_result,\n                "safe_to_track": True,\n            }\n            return result_payload\n'''
block = one(block, old_return, new_return, "motion success payload")
block = one(
    block,
    '''        except Exception as exc:\n            self.logger.error("Active PTZ calibration failed: %s", exc, exc_info=True)\n            return {"success": False, "mode": mode, "error": str(exc), "safe_to_track": False}\n        finally:\n            await asyncio.to_thread(self.ptz.continuous_stop)\n            home_ok = await self._calibration_home()\n            if not home_ok:\n                safe_to_track = False\n            self.state = "OFF"\n            self._calibrating = False\n''',
    '''        except Exception as exc:\n            self.logger.error("Active PTZ calibration failed: %s", exc, exc_info=True)\n            result_payload = {"success": False, "mode": mode, "error": str(exc), "safe_to_track": False}\n            return result_payload\n        finally:\n            await asyncio.to_thread(self.ptz.continuous_stop)\n            home_ok = await self._calibration_home()\n            safe_to_track = bool(home_ok)\n            if result_payload is not None:\n                result_payload["safe_to_track"] = safe_to_track\n                if not home_ok:\n                    result_payload["success"] = False\n                    result_payload["home_error"] = "Camera did not safely return to the configured home preset."\n            self.state = "OFF"\n            self._calibrating = False\n''',
    "home safety propagation",
)
text = text[:s] + block + text[e:]

motion_policy = r'''


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
):
    # Choose continuous PTZ only when a positional move is likely to become stale.
    h, w = frame_shape[:2]
    half_w = max(1.0, float(w) / 2.0)
    half_h = max(1.0, float(h) / 2.0)
    frame_diag = max(1.0, math.hypot(w, h))
    dominant_error = max(abs(err_x), abs(err_y))
    target_speed_norm = math.hypot(vx, vy) / frame_diag if velocity_valid else 0.0
    velocity_mature = bool(velocity_valid and velocity_sample_ms >= min_sample_ms)

    if abs(err_x) >= abs(err_y):
        dominant_axis_error = err_x
        dominant_axis_velocity = vx
    else:
        dominant_axis_error = err_y
        dominant_axis_velocity = vy

    moving_inward_dominant = velocity_mature and (
        abs(dominant_axis_error) >= exit_error
        and dominant_axis_error * dominant_axis_velocity < 0.0
        and abs(dominant_axis_velocity) / frame_diag >= motion_speed_norm
    )
    moving_outward = velocity_mature and (
        (abs(err_x) >= motion_error and err_x * vx > 0.0)
        or (abs(err_y) >= motion_error and err_y * vy > 0.0)
    )

    projected_travel_norm = 0.0
    if velocity_mature:
        projected_travel_norm = max(
            abs(vx) * max(0.0, move_eta_s) / half_w,
            abs(vy) * max(0.0, move_eta_s) / half_h,
        )

    motion_escape = (
        velocity_mature
        and dominant_error >= motion_error
        and target_speed_norm >= motion_speed_norm
        and moving_outward
    )
    deadline_escape = (
        velocity_mature
        and dominant_error >= motion_error
        and projected_travel_norm >= deadline_travel_norm
        and not moving_inward_dominant
    )
    hard_escape_error = max(0.90, entry_error + 0.08)
    edge_escape = edge_clipped and dominant_error >= entry_error and not moving_inward_dominant
    hard_escape = dominant_error >= hard_escape_error and not moving_inward_dominant

    use_continuous = bool(
        hybrid_enabled
        and not hybrid_disabled
        and cooldown_ready
        and (edge_escape or hard_escape or motion_escape or deadline_escape)
    )
    reason = None
    if use_continuous:
        if motion_escape:
            reason = "motion_escape"
        elif deadline_escape:
            reason = "deadline_motion"
        elif edge_escape:
            reason = "edge_clipped"
        else:
            reason = "hard_escape"

    return {
        "use_continuous": use_continuous,
        "reason": reason,
        "velocity_mature": velocity_mature,
        "target_speed_norm": target_speed_norm,
        "projected_travel_norm": projected_travel_norm,
        "moving_outward": moving_outward,
        "moving_inward_dominant": moving_inward_dominant,
        "motion_escape": motion_escape,
        "deadline_escape": deadline_escape,
    }
'''
if "def _motion_control_decision(" not in text:
    text = one(text, "\n\nclass DogTracker:", motion_policy + "\n\nclass DogTracker:", "insert motion policy")

s, e, block = scoped(text, "    def __init__(self, cfg: TrackerConfig", "    def _record_event")
block = one(
    block,
    '''        self._quality_undershoots = 0\n\n        self.active = False\n''',
    '''        self._quality_undershoots = 0\n        self._motion_control_min_sample_ms = max(\n            200, min(1000, _env_int("TRACKER_MOTION_CONTROL_MIN_SAMPLE_MS", 250))\n        )\n        self._motion_control_deadline_travel = max(\n            0.05, min(0.75, _env_float("TRACKER_MOTION_CONTROL_DEADLINE_TRAVEL", 0.18))\n        )\n        self._move_direct_lead_horizon_max = max(\n            0.25, min(1.50, _env_float("TRACKER_MOVE_DIRECT_LEAD_HORIZON_MAX", 0.80))\n        )\n        self._move_direct_lead_max_fraction = max(\n            0.05, min(0.25, _env_float("TRACKER_MOVE_DIRECT_LEAD_MAX_FRACTION", 0.12))\n        )\n\n        self.active = False\n''',
    "motion policy runtime config",
)
text = text[:s] + block + text[e:]

text = one(
    text,
    '''                min_velocity_sample_s=self.cfg.velocity_min_sample_ms / 1000.0,\n''',
    '''                min_velocity_sample_s=max(\n                    self.cfg.velocity_min_sample_ms,\n                    self._motion_control_min_sample_ms,\n                ) / 1000.0,\n''',
    "velocity maturity update",
)
text = one(
    text,
    '''                    < (self.cfg.velocity_min_sample_ms / 1000.0)\n''',
    '''                    < (\n                        max(self.cfg.velocity_min_sample_ms, self._motion_control_min_sample_ms)\n                        / 1000.0\n                    )\n''',
    "velocity maturity wait",
)
text = one(
    text,
    '''        if self.target is None or not self.target.velocity_valid:\n            return False, "velocity_sample"\n\n        h, w = frame_shape[:2]\n''',
    '''        if self.target is None or not self.target.velocity_valid:\n            return False, "velocity_sample"\n        if self.target.velocity_sample_ms < self._motion_control_min_sample_ms:\n            return False, "velocity_sample"\n\n        h, w = frame_shape[:2]\n''',
    "lead velocity maturity guard",
)

s, e, block = scoped(text, "    def _complete_move_quality", "    def _hybrid_axis_speed")
learn_start = block.index("        learned = False\n")
learn_end = block.index("        self._record_event(", learn_start)
block = (
    block[:learn_start]
    + '''        # Live-target telemetry is observational only. Active calibration\n        # uses a static scene and is authoritative for spatial camera response;\n        # a walking target must never rewrite those camera-specific scales.\n        learned = False\n'''
    + block[learn_end:]
)
text = text[:s] + block + text[e:]

text = one(
    text,
    '''            "control_mode": "hybrid",\n            "primary_control_mode": "moveDirectly",\n''',
    '''            "control_mode": "adaptive_hybrid",\n            "primary_control_mode": "continuous_when_moving",\n''',
    "status controller mode",
)
text = one(
    text,
    '''                "move_quality_samples": len(self._quality_improvements),\n                "mean_error_improvement": (None if not self._quality_improvements else round(sum(self._quality_improvements) / len(self._quality_improvements), 3)),\n''',
    '''                "move_quality_samples": len(self._quality_improvements),\n                "mean_error_improvement": (None if not self._quality_improvements else round(sum(self._quality_improvements) / len(self._quality_improvements), 3)),\n                "motion_control": {\n                    "controller_revision": 2,\n                    "min_velocity_sample_ms": self._motion_control_min_sample_ms,\n                    "deadline_travel_norm": round(self._motion_control_deadline_travel, 3),\n                    "move_direct_lead_horizon_max_s": round(self._move_direct_lead_horizon_max, 3),\n                    "move_direct_lead_max_fraction": round(self._move_direct_lead_max_fraction, 3),\n                    "live_spatial_learning": False,\n                },\n''',
    "status motion policy",
)

s, e, drive = scoped(text, "    async def _drive_to_target", "    def _render_debug_frame")
decision_insert_marker = '''            moving_inward_dominant = self.target.velocity_valid and (\n                abs(dominant_axis_error) >= self.cfg.hybrid_chase_exit_error\n                and dominant_axis_error * dominant_axis_velocity < 0.0\n                and abs(dominant_axis_velocity) / frame_diag >= self.cfg.hybrid_chase_motion_speed_norm\n            )\n'''
decision_insert = decision_insert_marker + '''            rough_move_distance = min(\n                2.0,\n                (abs(err_x) + abs(err_y)) * self.cfg.move_gain,\n            )\n            handoff_eta_s = self._predict_move_eta(rough_move_distance)\n            control_decision = _motion_control_decision(\n                err_x=err_x,\n                err_y=err_y,\n                vx=self.target.vx,\n                vy=self.target.vy,\n                velocity_valid=self.target.velocity_valid,\n                velocity_sample_ms=self.target.velocity_sample_ms,\n                frame_shape=frame_shape,\n                move_eta_s=handoff_eta_s,\n                hybrid_enabled=self.cfg.hybrid_chase_enabled,\n                hybrid_disabled=self._hybrid_disabled_for_session,\n                cooldown_ready=(now - self._hybrid_last_stopped_at) >= self.cfg.hybrid_chase_cooldown,\n                edge_clipped=edge_clipped_now,\n                exit_error=self.cfg.hybrid_chase_exit_error,\n                entry_error=self.cfg.hybrid_chase_entry_error,\n                motion_error=self.cfg.hybrid_chase_motion_error,\n                motion_speed_norm=self.cfg.hybrid_chase_motion_speed_norm,\n                min_sample_ms=self._motion_control_min_sample_ms,\n                deadline_travel_norm=self._motion_control_deadline_travel,\n            )\n            target_speed_norm = float(control_decision["target_speed_norm"])\n            moving_outward = bool(control_decision["moving_outward"])\n            moving_inward_dominant = bool(control_decision["moving_inward_dominant"])\n            motion_escape = bool(control_decision["motion_escape"])\n            deadline_escape = bool(control_decision["deadline_escape"])\n'''
drive = one(drive, decision_insert_marker, decision_insert, "deadline policy insertion")

start_marker = "            motion_escape = (\n"
end_marker = "            if hybrid_entry:\n"
start = drive.index(start_marker)
end = drive.index(end_marker, start)
drive = drive[:start] + '''            hybrid_entry = bool(control_decision["use_continuous"])\n''' + drive[end:]

drive = one(
    drive,
    '''                pan_speed = self.cfg.hybrid_chase_pan_sign * self._hybrid_axis_speed(err_x)\n                tilt_speed = -self._hybrid_axis_speed(err_y)\n''',
    '''                pan_sign = self._active_calibration.continuous_sign("pan", self.cfg.hybrid_chase_pan_sign)\n                tilt_sign = self._active_calibration.continuous_sign("tilt", -1)\n                pan_speed = pan_sign * self._hybrid_axis_speed(err_x, "pan")\n                tilt_speed = tilt_sign * self._hybrid_axis_speed(err_y, "tilt")\n''',
    "hybrid chase entry axis/sign",
)
drive = one(
    drive,
    '''                    entry_reason=(\n                        "motion_escape" if motion_escape else\n                        ("edge_clipped" if edge_clipped_now else "hard_escape")\n                    ),\n''',
    '''                    entry_reason=control_decision["reason"],\n''',
    "hybrid entry reason",
)
drive = one(
    drive,
    '''                    target_speed_norm=round(target_speed_norm, 4),\n                    moving_outward=bool(moving_outward),\n''',
    '''                    target_speed_norm=round(target_speed_norm, 4),\n                    velocity_mature=bool(control_decision["velocity_mature"]),\n                    projected_travel_norm=round(float(control_decision["projected_travel_norm"]), 4),\n                    deadline_escape=deadline_escape,\n                    moving_outward=bool(moving_outward),\n''',
    "hybrid decision telemetry",
)
drive = one(
    drive,
    '''            edge_rescue_active = edge_rescue_requested and not moving_inward_dominant\n''',
    '''            edge_rescue_active = (\n                edge_rescue_requested\n                and not moving_inward_dominant\n                and (not self.cfg.hybrid_chase_enabled or self._hybrid_disabled_for_session)\n            )\n''',
    "edge rescue fallback only",
)
drive = one(
    drive,
    '''            lead_horizon_s = self._predict_move_eta(base_move_distance)\n''',
    '''            predicted_move_eta_s = self._predict_move_eta(base_move_distance)\n            lead_horizon_s = min(predicted_move_eta_s, self._move_direct_lead_horizon_max)\n''',
    "bounded moveDirectly lead horizon",
)
drive = one(
    drive,
    '''                max_lead_x = w * 0.20\n                max_lead_y = h * 0.20\n''',
    '''                max_lead_x = w * self._move_direct_lead_max_fraction\n                max_lead_y = h * self._move_direct_lead_max_fraction\n''',
    "bounded moveDirectly lead distance",
)
drive = one(
    drive,
    '''                    predicted_move_eta_ms=int(lead_horizon_s * 1000),\n''',
    '''                    predicted_move_eta_ms=int(predicted_move_eta_s * 1000),\n''',
    "separate ETA from lead horizon",
)
text = text[:s] + drive + text[e:]

if "_hybrid_axis_speed(err_x)" in text or "_hybrid_axis_speed(err_y)" in text:
    raise RuntimeError("hybrid chase arity regression: one-argument call remains")

path.write_text(text)

path = Path("ptz_enhancements.py")
enh = path.read_text()
replacements = {
    '_bounded(float(row.get("pan_scale", 1.0)), 0.75, 1.25)': '_bounded(float(row.get("pan_scale", 1.0)), 0.75, 1.35)',
    '_bounded(float(row.get("tilt_scale", 1.0)), 0.75, 1.25)': '_bounded(float(row.get("tilt_scale", 1.0)), 0.75, 1.35)',
    '_bounded(pan_scale, 0.75, 1.25)': '_bounded(pan_scale, 0.75, 1.35)',
    '_bounded(tilt_scale, 0.75, 1.25)': '_bounded(tilt_scale, 0.75, 1.35)',
}
for old, new in replacements.items():
    if old not in enh:
        raise RuntimeError(f"missing CalibrationStore bound: {old}")
    enh = enh.replace(old, new)
path.write_text(enh)

print("active PTZ calibration hardening + controller revision 2 applied")
