from pathlib import Path

path = Path("tools/apply_active_ptz_calibration.py")
text = path.read_text()
start_marker = '''tracker = replace_once(\n    tracker,\n    ''' + "'''" + '''                "history_size": self.cfg.history_size,\\n            },\\n''' + "'''" + ''',\n'''
start = text.find(start_marker)
if start < 0:
    raise RuntimeError("old calibration config status patch block not found")
end_marker = '''    "calibration config status",\n)\n'''
end = text.find(end_marker, start)
if end < 0:
    raise RuntimeError("calibration config status patch block end not found")
end += len(end_marker)
replacement = '''tracker = replace_once(\n    tracker,\n    ''' + "'''" + '''            "config": self.cfg.public_dict(),\\n''' + "'''" + ''',\n    ''' + "'''" + '''            "config": {\n                **self.cfg.public_dict(),\n                "calibrate_on_start": self._startup_calibration_policy,\n                "calibration_scope": self._calibration_scope,\n                "calibration_max_age_days": self._calibration_max_age_days,\n                "calibration_zoom_levels": self._calibration_zoom_levels,\n                "calibration_offsets": self._calibration_offsets,\n                "calibration_continuous_speeds": self._calibration_continuous_speeds,\n                "calibration_continuous_duration": self._calibration_continuous_duration,\n                "calibration_onvif_benchmark": self._calibration_onvif_benchmark,\n            },\\n''' + "'''" + ''',\n    "calibration config status",\n)\n'''
path.write_text(text[:start] + replacement + text[end:])
print("fixed calibration config status patch marker")
