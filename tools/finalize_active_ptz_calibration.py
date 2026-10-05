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

# moveDirectly sample: learn camera settle duration, not later image-capture overhead.
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

# Continuous sample telemetry should similarly stop its timing clock at settled idle.
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

# Active motion calibration: seed runtime gains for ~80% one-step correction and
# make the returned safety result reflect the final return-home operation.
s, e, block = scoped(text, "    async def _calibrate_motion", "    async def calibrate")
block = one(
    block,
    '''        safe_to_track = True\n        try:\n''',
    '''        safe_to_track = True\n        result_payload: Optional[dict] = None\n        try:\n''',
    "motion result payload",
)
block = one(
    block,
    '''                    pan_scale = max(0.75, min(1.25, 1.0 / max(0.20, pan_ratio)))\n                    tilt_scale = max(0.75, min(1.25, 1.0 / max(0.20, tilt_ratio)))\n''',
    '''                    # The calibration move is a full normalized correction, while\n                    # runtime moveDirectly applies cfg.move_gain first. Seed the learned\n                    # multiplier so one settled runtime move corrects about 80% of a\n                    # stationary target error; passive telemetry can refine from there.\n                    target_fraction = 0.80\n                    pan_scale = max(0.75, min(1.35, target_fraction / max(0.20, self.cfg.move_gain * pan_ratio)))\n                    tilt_scale = max(0.75, min(1.35, target_fraction / max(0.20, self.cfg.move_gain * tilt_ratio)))\n''',
    "active spatial target",
)
block = one(
    block,
    '''            onvif_ok = (not run_onvif) or isinstance(onvif_result, dict)\n''',
    '''            # ONVIF remains optional for native motion/all calibration. An\n            # explicit mode=onvif request only succeeds if the camera actually\n            # exposes a reachable ONVIF PTZ service.\n            onvif_ok = mode != "onvif" or bool(isinstance(onvif_result, dict) and onvif_result.get("available"))\n''',
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

# Hybrid chase entry must pass the new axis argument and use the calibrated
# direction signs. The update-loop path already does this; this guard keeps the
# initial entry path in lock-step so a target near the frame edge cannot crash
# the tracker on first escape-chase activation.
text = one(
    text,
    '''                pan_speed = self.cfg.hybrid_chase_pan_sign * self._hybrid_axis_speed(err_x)\n                tilt_speed = -self._hybrid_axis_speed(err_y)\n''',
    '''                pan_sign = self._active_calibration.continuous_sign("pan", self.cfg.hybrid_chase_pan_sign)\n                tilt_sign = self._active_calibration.continuous_sign("tilt", -1)\n                pan_speed = pan_sign * self._hybrid_axis_speed(err_x, "pan")\n                tilt_speed = tilt_sign * self._hybrid_axis_speed(err_y, "tilt")\n''',
    "hybrid chase entry axis/sign",
)
if '_hybrid_axis_speed(err_x)' in text or '_hybrid_axis_speed(err_y)' in text:
    raise RuntimeError("hybrid chase arity regression: one-argument call remains")

# Let passive refinement preserve the slightly wider active-calibration range.
old = '                    return min(1.25, scale * (1.0 + rate))\n'
if text.count(old) != 1:
    raise RuntimeError(f"passive scale ceiling: expected 1 match, found {text.count(old)}")
text = text.replace(old, '                    return min(1.35, scale * (1.0 + rate))\n', 1)
path.write_text(text)

# CalibrationStore bounds must match the controller's calibrated range.
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

print("final active PTZ calibration hardening applied")
