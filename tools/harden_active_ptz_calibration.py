from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


tracker_path = Path("tracker.py")
tracker = tracker_path.read_text()

tracker = replace_once(
    tracker,
    '''        requested = math.hypot(err_x, err_y)\n        elapsed = time.monotonic() - started\n        row = {\n''',
    '''        requested = math.hypot(err_x, err_y)\n        command_s = command_done - started\n        settled_s = command_s + float(wait["elapsed_s"])\n        row = {\n''',
    "move timing basis",
)
tracker = replace_once(
    tracker,
    '''            "motion_start_ms": None if wait["motion_start_s"] is None else int(wait["motion_start_s"] * 1000),\n            "settled_ms": int(elapsed * 1000),\n''',
    '''            "motion_start_ms": None if wait["motion_start_s"] is None else int((command_s + float(wait["motion_start_s"])) * 1000),\n            "settled_ms": int(settled_s * 1000),\n''',
    "move timing telemetry",
)
tracker = replace_once(
    tracker,
    '''            self._update_move_timing_model(requested, elapsed)\n''',
    '''            self._update_move_timing_model(requested, settled_s)\n''',
    "move timing learner",
)
tracker = replace_once(
    tracker,
    '''        row = {\n            "zoom_factor": round(factor, 3),\n            "axis": axis,\n            "speed": speed,\n            "raw_sign": raw_sign,\n            "correction_sign": desired_sign,\n            "normalized_rate_per_s": round(rate, 4),\n            "command_http_ms": int((command_done - started) * 1000),\n            "settled_ms": int((time.monotonic() - started) * 1000),\n''',
    '''        settled_s = (command_done - started) + self._calibration_continuous_duration + float(wait["elapsed_s"])\n        row = {\n            "zoom_factor": round(factor, 3),\n            "axis": axis,\n            "speed": speed,\n            "raw_sign": raw_sign,\n            "correction_sign": desired_sign,\n            "normalized_rate_per_s": round(rate, 4),\n            "command_http_ms": int((command_done - started) * 1000),\n            "settled_ms": int(settled_s * 1000),\n''',
    "continuous timing telemetry",
)

tracker = replace_once(
    tracker,
    '''                    pan_scale = max(0.75, min(1.25, 1.0 / max(0.20, pan_ratio)))\n                    tilt_scale = max(0.75, min(1.25, 1.0 / max(0.20, tilt_ratio)))\n''',
    '''                    # Target ~80% correction in one settled move. This is\n                    # materially quicker than the old fixed 0.60 gain while still\n                    # leaving margin for target motion and bbox noise.\n                    target_fraction = 0.80\n                    pan_scale = max(0.75, min(1.35, target_fraction / max(0.20, self.cfg.move_gain * pan_ratio)))\n                    tilt_scale = max(0.75, min(1.35, target_fraction / max(0.20, self.cfg.move_gain * tilt_ratio)))\n''',
    "calibrated controller target",
)
tracker = replace_once(
    tracker,
    '''            onvif_ok = (not run_onvif) or isinstance(onvif_result, dict)\n''',
    '''            # ONVIF is optional during combined/native calibration. An explicit\n            # mode=onvif request succeeds only when the camera actually responded.\n            onvif_ok = mode != "onvif" or bool(isinstance(onvif_result, dict) and onvif_result.get("available"))\n''',
    "onvif benchmark success semantics",
)
tracker = replace_once(
    tracker,
    '''        safe_to_track = True\n        try:\n''',
    '''        safe_to_track = True\n        result_payload: Optional[dict] = None\n        try:\n''',
    "motion result payload",
)
tracker = replace_once(
    tracker,
    '''            return {\n                "success": success,\n                "mode": mode,\n                "move_directly_samples": len(move_samples),\n                "continuous_samples": len(continuous_samples),\n                "move_timing_model": {\n                    "samples": len(self._move_timing_samples),\n                    "ready": self._move_eta_model_ready,\n                    "intercept_s": round(self._move_eta_intercept, 4),\n                    "slope_s_per_norm": round(self._move_eta_slope, 4),\n                },\n                "active_calibration": self._active_calibration.public_dict(),\n                "onvif_benchmark": onvif_result,\n                "safe_to_track": safe_to_track,\n            }\n''',
    '''            result_payload = {\n                "success": success,\n                "mode": mode,\n                "move_directly_samples": len(move_samples),\n                "continuous_samples": len(continuous_samples),\n                "move_timing_model": {\n                    "samples": len(self._move_timing_samples),\n                    "ready": self._move_eta_model_ready,\n                    "intercept_s": round(self._move_eta_intercept, 4),\n                    "slope_s_per_norm": round(self._move_eta_slope, 4),\n                },\n                "active_calibration": self._active_calibration.public_dict(),\n                "onvif_benchmark": onvif_result,\n                "safe_to_track": True,\n            }\n            return result_payload\n''',
    "motion success return",
)
tracker = replace_once(
    tracker,
    '''        except Exception as exc:\n            self.logger.error("Active PTZ calibration failed: %s", exc, exc_info=True)\n            return {"success": False, "mode": mode, "error": str(exc), "safe_to_track": False}\n        finally:\n            await asyncio.to_thread(self.ptz.continuous_stop)\n            home_ok = await self._calibration_home()\n            if not home_ok:\n                safe_to_track = False\n            self.state = "OFF"\n            self._calibrating = False\n''',
    '''        except Exception as exc:\n            self.logger.error("Active PTZ calibration failed: %s", exc, exc_info=True)\n            result_payload = {"success": False, "mode": mode, "error": str(exc), "safe_to_track": False}\n            return result_payload\n        finally:\n            await asyncio.to_thread(self.ptz.continuous_stop)\n            home_ok = await self._calibration_home()\n            safe_to_track = bool(home_ok)\n            if result_payload is not None:\n                result_payload["safe_to_track"] = safe_to_track\n                if not home_ok:\n                    result_payload["success"] = False\n                    result_payload["home_error"] = "Camera did not safely return to the configured home preset."\n            self.state = "OFF"\n            self._calibrating = False\n''',
    "home safety return propagation",
)

# Passive refinement must not clamp a trustworthy active-calibration scale back to
# the old 1.25 ceiling on the first undershoot observation.
tracker = replace_once(
    tracker,
    '''                    return min(1.25, scale * (1.0 + rate))\n''',
    '''                    return min(1.35, scale * (1.0 + rate))\n''',
    "passive scale upper bound",
)
tracker_path.write_text(tracker)

enh_path = Path("ptz_enhancements.py")
enh = enh_path.read_text()
enh = enh.replace('_bounded(float(row.get("pan_scale", 1.0)), 0.75, 1.25)', '_bounded(float(row.get("pan_scale", 1.0)), 0.75, 1.35)')
enh = enh.replace('_bounded(float(row.get("tilt_scale", 1.0)), 0.75, 1.25)', '_bounded(float(row.get("tilt_scale", 1.0)), 0.75, 1.35)')
enh = enh.replace('_bounded(pan_scale, 0.75, 1.25)', '_bounded(pan_scale, 0.75, 1.35)')
enh = enh.replace('_bounded(tilt_scale, 0.75, 1.25)', '_bounded(tilt_scale, 0.75, 1.35)')
if '0.75, 1.25' in enh[enh.index('def spatial_scales'):enh.index('def public_dict', enh.index('def spatial_scales'))]:
    raise RuntimeError("spatial scale ceiling was not fully updated")
enh_path.write_text(enh)

print("active PTZ calibration hardening applied")
