# !!!WARNING!!! THIS IS AI CODED SLOP !!!WARNING!!!
# Blue Iris AI Gateway

A containerized AI gateway built for **Blue Iris**, combining **YOLO object detection**, **face recognition**, and an optional **multi-class PTZ autotracker** in one FastAPI service.

The gateway is designed for NVIDIA/CUDA homelab deployments and keeps all GPU inference behind a shared execution gate so normal Blue Iris inference and the PTZ tracker do not collide on the GPU.

## Features

- **Blue Iris-oriented object detection** using Ultralytics YOLO.
- **TensorRT / CUDA acceleration** with automatic `.pt` -> `.engine` compilation when needed.
- **Built-in face recognition** using `facenet-pytorch` (`MTCNN` + `InceptionResnetV1`).
- **GPU watchdog / tainted-pipeline protection** so a timed-out CUDA worker cannot immediately be followed by another GPU job.
- **Atomic face database persistence** under `/app/models/faces_db.pt`.
- **Optional PTZ autotracking** using the same YOLO model already loaded by the gateway.
- **Latest-frame-only RTSP capture** for tracking, avoiding a backlog of stale frames.
- **Native Dahua/Amcrest 3D positioning** using `moveDirectly`, with status-driven PTZ settling rather than fixed movement sleeps.
- **Bounded predictive lead** based only on stationary-camera target motion.
- **Adaptive `moveDirectly` lead** learned from completed camera moves, with bounded one-shot edge rescue.
- **Hybrid escape chase** that uses continuous PTZ only when a fast target is in genuine danger of leaving the frame.
- **Evidence-gated conservative auto-zoom** that requires a small, confident, centered target across multiple observations before zooming in.
- **GitHub Actions -> GHCR** builds on pushes to `main`.

## Container Image

```text
ghcr.io/nullexxx/blueiris-ai-gateway:latest
```

The container listens on port `32168`. The example Compose file maps that to host port `32169`.

## Quick Start

A complete example is included as [`docker-compose.example.yml`](docker-compose.example.yml), with secrets kept in `.env` rather than committed directly into the Compose file.

```bash
cp .env.example .env
nano .env

docker compose -f docker-compose.example.yml pull
docker compose -f docker-compose.example.yml up -d
```

At minimum, set the PTZ camera IP, username, and password in `.env` if tracking is enabled.

> **Do not commit your real camera password.** `.env` is ignored by the included `.gitignore`.

## Models

Mount the persistent models directory to:

```text
/app/models
```

For the default configuration, place a compatible YOLO model such as:

```text
yolo11m.pt
```

in that directory. On startup, the gateway prefers models in this order:

```text
.engine -> .onnx -> .pt
```

If a `.pt` model exists without a matching TensorRT engine, the gateway can build the `.engine` automatically. A per-model manifest records the source-weight SHA-256, TensorRT/CUDA versions, GPU identity/compute capability, image size, and precision. A source/runtime change invalidates the cached engine, and an engine load failure triggers one rebuild attempt before falling back to the `.pt` model.

## API Endpoints

### Object Detection

- `POST /v1/vision/detection` — Run object detection with the default model.
- `POST /v1/vision/custom/{model_name}` — Run detection with a specific model.
- `GET /v1/vision/custom/list` — List discovered models.

### Face Recognition

- `GET /v1/vision/face/list`
- `POST /v1/vision/face/list`
- `POST /v1/vision/face/register`
- `POST /v1/vision/face/delete`
- `POST /v1/vision/face/recognize`

### System / Health

- `GET /`
- `GET /status`

The health endpoint returns `503` if the GPU pipeline is degraded or tainted.

### PTZ Tracker

- `GET /v1/tracker/status` — Tracker state, current target, PTZ operation, counters, and effective configuration.
- `GET /v1/tracker/camera-status` — Raw camera PTZ status from the Dahua/Amcrest CGI.
- `POST /v1/tracker/calibrate` — Manual bounded ONVIF zoom calibration; requires tracking to be stopped.
- `GET /v1/tracker/debug.jpg` — Latest annotated tracking frame.
- `GET /v1/tracker/history?limit=300` — Recent tracker event history.
- `POST /v1/tracker/history/clear` — Clear tracker history.
- `POST /v1/tracker/start` — Start tracking.
- `POST /v1/tracker/stop` — Stop tracking.
- `POST /v1/tracker/home` — Send the camera to the configured home preset.

Example:

```bash
curl -X POST http://HOST:32169/v1/tracker/start
curl -s http://HOST:32169/v1/tracker/status | python3 -m json.tool
curl -s 'http://HOST:32169/v1/tracker/history?limit=300' | python3 -m json.tool
```

## PTZ Tracker Design

The tracker continuously drains the camera RTSP substream and retains only the newest decoded frame. Tracker inference is opportunistic: regular Blue Iris and FaceNet work has priority, so the tracker drops a frame instead of waiting behind queued gateway inference.

Normal pan/tilt tracking uses Dahua/Amcrest `moveDirectly` 3D positioning. The tracker polls camera status and position until the camera reports idle and stable rather than sleeping for a guessed movement duration. Completed moves feed a small timing model that estimates future `moveDirectly` duration for bounded predictive lead.

Predictive target velocity is learned only while the camera is stationary. It is suppressed for immature/noisy velocity samples, low-confidence or tiny detections, and edge-clipped detections. A one-shot **edge rescue** can use a stronger positional gain when the target is already near escape.

For moving targets, Rev 6 prefers a **fractional ONVIF feedback servo** when the camera proves during active calibration that its ONVIF ContinuousMove velocity is genuinely proportional below native speed 1. Rev 6.2 lets that closed-loop fractional servo run continuously instead of forcing a 2.5-second brake/restart cycle, uses the established-target hold-confidence threshold while actively chasing, and progressively adds proportional authority only when the target is far off-center. Rev 6.3 adds velocity-aware earlier braking for fast fractional moves, a 0.50-second chase miss grace, a 5-second target-release window before returning home, and association-miss diagnostics so rejected detections can be distinguished from true detector dropouts. Rev 6.4 adds an outward-only catch-up multiplier for large errors that are still growing, decays the last verified velocity during marginal-confidence or detector-miss grace, and introduces a short servo-hold state so transient predicted stops do not tear down the entire chase. Rev 6.5 makes divergence per-axis and sign-aware, prevents servo hold from settling while the target remains outside the continuous-exit region, records rejected association candidates, and adds a brief native edge-rescue burst only when the fractional servo is near saturation and the target is still escaping. The controller maps desired image-space correction rate into the measured fractional velocity, brakes on genuine confidence loss, and waits for both a minimum physical quiet interval and optical-flow stability before learning subject velocity again. If fractional ONVIF is unavailable or quantized, native Dahua continuous movement is reserved for clipped/hard-edge rescue rather than used for fine servo tracking. `moveDirectly` is a stationary-target precision tool and no longer uses predictive lead.

PTZ completion is fail-closed. At the operation deadline the tracker performs a final status read. If status remains unavailable, or the camera still cannot be established as safely idle, tracking stops with `state=PTZ_ERROR` rather than issuing another movement command. A later `/v1/tracker/start` can resume tracking; if the tracker task itself crashed, `start` recreates it.

Target-loss timing is paused while the camera is moving so global image motion is not mistaken for subject loss. After movement completes, the velocity estimator is rebased from a fresh stationary-camera frame.

The tracker has been developed against a Dahua/Amcrest-style PTZ CGI camera, including an Amcrest IP2M-863EW-AI. Other cameras may require changes to the PTZ transport or coordinate behavior.

## Main Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `DEFAULT_MODEL` | `yolo11m` | Primary YOLO model stem. |
| `PRELOAD_MODELS` | blank | Comma-separated model stems to preload, or `all`. If blank, the default model is loaded. |
| `ALLOW_LAZY_LOAD` | `false` | Allow requested models to load on demand. |
| `HALF_PRECISION` | `true` | Use FP16 where supported. |
| `INFERENCE_TIMEOUT` | `8.0` | Runtime CUDA job timeout in seconds. Applies to YOLO and FaceNet. |
| `MODEL_LOAD_TIMEOUT` | `180.0` | Timeout for an optional lazy model load/warmup. |
| `RECOVERY_GRACE_PERIOD` | `4.0` | Grace period for an orphaned CUDA worker before the container self-terminates. |
| `MAX_FACE_EMBEDDINGS_PER_USER` | `20` | Maximum saved FaceNet embeddings per enrolled identity; oldest samples are discarded first. |
| `ALERT_WEBHOOK_URL` | blank | Optional crash-alert webhook. |

## Tracker Environment Variables

### Camera / Stream

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKER_ENABLED` | `false` | Enable tracker service. |
| `TRACKER_AUTOSTART` | `false` | Start tracking automatically when the app starts. |
| `TRACKER_CALIBRATE_ON_START` | `if_missing` | Startup calibration policy: `off`, `if_missing`, `if_stale`, or `always`. Full calibration moves the camera before autotracking begins. |
| `TRACKER_CALIBRATION_SCOPE` | `all` | Startup/manual active calibration scope: `zoom`, `movedirectly`, `continuous`, `motion`, `onvif`, or `all`. |
| `TRACKER_ACTIVE_CALIBRATION_PATH` | `/app/models/tracker_ptz_active_calibration.json` | Persistent native/ONVIF motion-response calibration. |
| `TRACKER_CALIBRATION_MAX_AGE_DAYS` | `30` | Age threshold used by the `if_stale` startup policy. |
| `TRACKER_CALIBRATION_ZOOM_LEVELS` | `1.0,1.75,2.25,3.0` | Optical zoom factors sampled by active pan/tilt calibration. |
| `TRACKER_CALIBRATION_OFFSETS` | `0.18,0.35` | Normalized moveDirectly offsets used to learn response and timing. |
| `TRACKER_CALIBRATION_CONTINUOUS_SPEEDS` | `1,3,6` | Native continuous PTZ speeds sampled during calibration. |
| `TRACKER_CALIBRATION_CONTINUOUS_DURATION` | `0.22` | Seconds each bounded continuous test pulse runs. |
| `TRACKER_CALIBRATION_ONVIF_BENCHMARK` | `true` | Probe ONVIF PTZ spaces, benchmark a tiny FOV-relative move, and characterize fractional ContinuousMove response when supported. |
| `TRACKER_SERVO_ACTUATOR` | `auto` | Rev 6 servo actuator: `auto`, `onvif`, or `native`. `auto` uses ONVIF only after a usable fractional response has been calibrated. |
| `TRACKER_ONVIF_SERVO_MAX_VELOCITY` | `0.35` | Maximum normalized ONVIF velocity the feedback servo may request. |
| `TRACKER_SERVO_KP` | `0.70` | Base proportional gain used near center. |
| `TRACKER_SERVO_BRAKE_HORIZON` | `0.30` | Base look-ahead used to brake the fractional servo before crossing center. |
| `TRACKER_SERVO_BRAKE_VELOCITY_EXTENSION` | `0.08` | Additional brake look-ahead added proportionally as current fractional ONVIF velocity approaches its configured ceiling. |
| `TRACKER_SERVO_LARGE_ERROR_START` | `0.45` | Absolute normalized axis error where Rev 6.2 begins adding proportional authority. |
| `TRACKER_SERVO_LARGE_ERROR_FULL` | `0.75` | Axis error where the full large-error gain is reached. |
| `TRACKER_SERVO_LARGE_ERROR_GAIN` | `1.45` | Maximum multiplier applied to the proportional term for large off-center errors. |
| `TRACKER_SERVO_OUTWARD_CATCHUP_START` | `0.60` | Error magnitude where Rev 6.4 may add outward-only catch-up authority if error is still growing. |
| `TRACKER_SERVO_OUTWARD_CATCHUP_FULL` | `0.85` | Error magnitude where full catch-up is reached. |
| `TRACKER_SERVO_OUTWARD_CATCHUP_GAIN` | `1.30` | Maximum extra desired-rate multiplier while a large error is moving farther from center. |
| `TRACKER_SERVO_HOLD_SECONDS` | `0.30` | Duration a predicted stop preserves chase identity before settling out. |
| `TRACKER_SERVO_HOLD_MIN_FRAMES` | `3` | Minimum stopped observations before servo hold becomes a full stop. |
| `TRACKER_SERVO_HOLD_RESUME_GROWTH` | `0.06` | Dominant-error growth that immediately resumes a held chase. |
| `TRACKER_FAST_FOLLOW_SPEED_NORM` | `0.10` | Minimum mature normalized target speed for Rev 6.6 predictive follow. |
| `TRACKER_FAST_FOLLOW_HORIZON` | `0.45` | Seconds ahead used to predict a fast center crossing. |
| `TRACKER_FAST_FOLLOW_ERROR` | `0.22` | Future/current per-axis error that makes a fast target eligible for early continuous follow. |
| `TRACKER_CLASS_CONTINUITY_LABELS` | `dog,cat,bird,bear` | Animal labels that may remain one physical track while class evidence votes on the label. |
| `TRACKER_STATIC_HOTSPOT_GUARD_ENABLED` | `true` | Learn recurring home-view acquisition fingerprints and require extra evidence before they can drive PTZ. |
| `TRACKER_STATIC_HOTSPOT_BURSTS` | `3` | Distinct separated bursts before a home-view box is considered suspicious. |
| `TRACKER_STATIC_HOTSPOT_EXTRA_FRAMES` | `3` | Extra acquisition hits required at a suspicious hotspot. |
| `TRACKER_STATIC_HOTSPOT_OVERRIDE_CONF` | `0.70` | Strong confidence that overrides the hotspot motion requirement. |
| `TRACKER_EVENT_TIMEZONE` | `America/New_York` | Primary history/status timezone; every event also retains `time_utc`. |
| `TRACKER_NATIVE_EDGE_RESCUE_ENABLED` | `true` | Permit a brief native PTZ burst when fractional ONVIF is near saturation and the target is still escaping near an edge. |
| `TRACKER_NATIVE_EDGE_RESCUE_ERROR` | `0.75` | Per-axis error required before native rescue can trigger. |
| `TRACKER_NATIVE_EDGE_RESCUE_EXIT_ERROR` | `0.65` | Exit native rescue once the target is pulled back inside this dominant-error region. |
| `TRACKER_NATIVE_EDGE_RESCUE_FULL_ERROR` | `0.92` | Error at which rescue uses the configured maximum native speed. |
| `TRACKER_NATIVE_EDGE_RESCUE_SATURATION` | `0.80` | Fraction of the ONVIF velocity ceiling that counts as fractional saturation. |
| `TRACKER_NATIVE_EDGE_RESCUE_SECONDS` | `0.35` | Hard duration of one native rescue burst. |
| `TRACKER_NATIVE_EDGE_RESCUE_COOLDOWN` | `0.25` | Minimum fractional-control interval before another native rescue burst may start. |
| `TRACKER_NATIVE_EDGE_RESCUE_MIN_SPEED` | `3` | Native speed used before the full-error threshold. |
| `TRACKER_NATIVE_EDGE_RESCUE_MAX_SPEED` | `6` | Native speed used at the full-error threshold. |
| `TRACKER_CONFIDENCE_COAST_START_SCALE` | `0.70` | Initial fraction of last verified fractional velocity during confidence/miss grace. |
| `TRACKER_CONFIDENCE_COAST_END_SCALE` | `0.20` | Final fraction of snapshot velocity at grace expiry. |
| `TRACKER_ONVIF_SERVO_CALIBRATION_VELOCITIES` | `0.04,0.08,0.16` | Fractional ONVIF ContinuousMove velocities sampled by manual/active continuous calibration. |
| `TRACKER_ONVIF_SERVO_CALIBRATION_DURATION` | `0.22` | Seconds each fractional ONVIF calibration pulse runs. |
| `TRACKER_CAMERA_IP` | blank | PTZ camera IP or hostname. |
| `TRACKER_CAMERA_USER` | `admin` | Camera username. |
| `TRACKER_CAMERA_PASSWORD` | blank | Camera password. Prefer `.env`; do not commit it. |
| `TRACKER_CAMERA_CHANNEL` | `0` | Dahua CGI channel index. |
| `TRACKER_RTSP_CHANNEL` | `1` | RTSP channel number. |
| `TRACKER_RTSP_SUBTYPE` | `1` | RTSP subtype; `1` is normally the substream. |
| `TRACKER_RTSP_URL` | blank | Optional full RTSP URL override. |
| `TRACKER_MODEL` | `DEFAULT_MODEL` | Model used for tracker inference. |

### Detection / Association

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKER_TARGET_CLASSES` | `person,dog,cat,bird,bear` | Classes eligible for tracking. |
| `TRACKER_TARGET_PRIORITY` | `dog,person,bear,cat,bird` | Acquisition priority. |
| `TRACKER_FPS` | `10` | Maximum tracker inference rate. The example Compose uses `15`. |
| `TRACKER_ACQUIRE_CONF` | `0.40` | Confidence required for a new target. |
| `TRACKER_HOLD_CONF` | `0.22` | Confidence allowed for an already-locked target, including an active Rev 6.2 continuous chase. |
| `TRACKER_REACQUIRE_CONF` | `0.35` | Confidence required while actually reacquiring a lost target; it is not used to interrupt an established chase. |
| `TRACKER_ACQUIRE_FRAMES` | `2` | Matching frames required before target lock. |
| `TRACKER_ASSOCIATION_IDLE_DISTANCE` | `0.22` | Same-class association distance while camera is stationary. |
| `TRACKER_ASSOCIATION_MOVING_DISTANCE` | `0.40` | Wider association gate around camera movement. |

### Pan / Tilt Prediction

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKER_MOVE_DIRECTLY_ENABLED` | `true` | Use Dahua/Amcrest 3D `moveDirectly` for normal tracking. |
| `TRACKER_MOVE_DEADZONE_X` | `0.14` | Horizontal normalized deadzone. |
| `TRACKER_MOVE_DEADZONE_Y` | `0.18` | Vertical normalized deadzone. |
| `TRACKER_MOVE_GAIN` | `0.60` | Fraction of projected frame error applied to a normal positional move. |
| `TRACKER_LEAD_TIME` | `0.75` | Fallback lead horizon before enough move timing samples exist. |
| `TRACKER_VELOCITY_MIN_SAMPLE_MS` | `100` | Minimum stationary-camera sample window before velocity is trusted. |
| `TRACKER_LEAD_MIN_CONF` | `0.50` | Suppress predictive lead below this confidence. |
| `TRACKER_LEAD_MIN_SPAN` | `0.08` | Suppress predictive lead for very small targets. |
| `TRACKER_LEAD_EDGE_MARGIN` | `0.02` | Suppress predictive lead when the bbox touches this frame margin. |
| `TRACKER_ADAPTIVE_LEAD` | `true` | Learn move duration from completed `moveDirectly` operations. |
| `TRACKER_MOVE_ETA_MIN_SAMPLES` | `3` | Completed moves required before the learned timing model is trusted. |
| `TRACKER_MOVE_ETA_HISTORY` | `24` | Maximum completed move samples retained. |
| `TRACKER_MOVE_ETA_MIN` | `0.50` | Minimum learned/fallback lead horizon. |
| `TRACKER_MOVE_ETA_MAX` | `2.00` | Maximum learned lead horizon. |
| `TRACKER_VELOCITY_CONSISTENCY_COSINE` | `0.25` | Reject abrupt direction changes from predictive lead. |
| `TRACKER_VELOCITY_JUMP_RATIO` | `4.0` | Reject implausible velocity magnitude jumps. |

Predictive lead is hard-clamped to 20% of frame width/height per axis in code.

### Edge Rescue / Hybrid Escape Chase

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKER_EDGE_RESCUE_ENABLED` | `true` | Permit a stronger one-shot positional correction near an edge. |
| `TRACKER_EDGE_RESCUE_ERROR` | `0.75` | Normalized error that can request edge rescue. |
| `TRACKER_EDGE_RESCUE_GAIN` | `0.85` | Positional gain used by edge rescue. |
| `TRACKER_HYBRID_CHASE_ENABLED` | `true` | Allow continuous PTZ only for genuine escape conditions. |
| `TRACKER_HYBRID_CHASE_PAN_SIGN` | `-1` | Camera-specific horizontal continuous-move direction sign. |
| `TRACKER_HYBRID_CHASE_ENTRY_ERROR` | `0.82` | Hard normalized error threshold for chase entry. |
| `TRACKER_HYBRID_CHASE_EXIT_ERROR` | `0.50` | Stop chase after the target returns inside this error. |
| `TRACKER_HYBRID_CHASE_MOTION_ERROR` | `0.55` | Lower error threshold used for fast outward-motion entry. |
| `TRACKER_HYBRID_CHASE_MOTION_SPEED_NORM` | `0.03` | Minimum normalized target speed for motion-based entry. |
| `TRACKER_HYBRID_CHASE_MISS_GRACE` | `0.15` | Brief detector-miss grace during motion blur. |
| `TRACKER_HYBRID_MISSING_GRACE` | `0.50` | Effective Rev 6.3 active-chase detector-miss grace. |
| `TRACKER_HYBRID_CHASE_MIN_SPEED` | `1` | Minimum continuous PTZ speed. |
| `TRACKER_HYBRID_CHASE_MAX_SPEED` | `6` | Maximum continuous PTZ speed. |
| `TRACKER_HYBRID_CHASE_FULL_SPEED_ERROR` | `0.90` | Error at which maximum continuous speed is reached. |
| `TRACKER_HYBRID_CHASE_COMMAND_INTERVAL` | `0.18` | Minimum interval between changed chase commands. |
| `TRACKER_HYBRID_CHASE_KEEPALIVE` | `0.45` | Keepalive interval when chase speed is unchanged. |
| `TRACKER_HYBRID_CHASE_CAMERA_TIMEOUT` | `1` | Camera-side continuous movement timeout. |
| `TRACKER_HYBRID_CHASE_MAX_SECONDS` | `2.50` | Maximum duration of coarse native-discrete escape rescue. Rev 6.2 does not duration-cap the calibrated fractional ONVIF feedback servo. |
| `TRACKER_HYBRID_CHASE_COOLDOWN` | `0.35` | Cooldown before re-entering chase. |
| `TRACKER_HYBRID_CHASE_SETTLE_FRAMES` | `2` | Fresh frames required after chase stops. |
| `TRACKER_HYBRID_DIVERGENCE_FRAMES` | `3` | Consecutive materially-worsening chase frames before the chase is aborted. |
| `TRACKER_HYBRID_DIVERGENCE_GROWTH` | `0.05` | Minimum normalized error growth that counts toward divergence. |
| `TRACKER_SERVO_POST_STOP_SETTLE` | `0.35` | Minimum physical quiet time after continuous PTZ stops before velocity can be rebased. |
| `TRACKER_SERVO_DIVERGENCE_TRIP_LIMIT` | `3` | Divergence strikes required within the rolling window before continuous tracking is disabled for the session. |
| `TRACKER_SERVO_DIVERGENCE_WINDOW` | `30` | Seconds in the rolling divergence strike window. |
| `TRACKER_SERVO_DIVERGENCE_COOLDOWN` | `0.75` | Recovery cooldown after a single aborted divergent chase. |

### PTZ Operation Settling

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKER_PTZ_STATUS_POLL_INTERVAL` | `0.12` | Seconds between camera PTZ status polls. |
| `TRACKER_PTZ_OPERATION_TIMEOUT` | `4.0` | Hard timeout for the latest physical PTZ destination. |
| `TRACKER_POST_MOVE_FRAMES` | `1` | Fresh frames required after PTZ completion before another action. |
| `TRACKER_PTZ_HTTP_TIMEOUT` | `0.75` | HTTP timeout for camera CGI requests. |

### Auto-Zoom / Home

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKER_AUTOZOOM` | `true` | Enable bounded auto-zoom. |
| `TRACKER_ZOOM_WIDE_POSITION` | `5.12` | Camera-reported full-wide zoom position used by the current calibration. |
| `TRACKER_ZOOM_MIN_FACTOR` | `1.0` | Minimum tracking zoom factor. |
| `TRACKER_ZOOM_MAX_FACTOR` | `3.0` | Maximum tracking zoom factor; deliberately capped to preserve context. |
| `TRACKER_ZOOM_TARGET_MIN` | `0.12` | A target must be smaller than this span before zoom-in can even qualify. |
| `TRACKER_ZOOM_TARGET_MAX` | `0.42` | Zoom out when the target is larger than this span. |
| `TRACKER_ZOOM_IN_MIN_CONF` | `0.65` | Minimum confidence required for zoom-in. |
| `TRACKER_ZOOM_IN_CONFIRM_FRAMES` | `8` | Consecutive qualifying observations required before zoom-in. |
| `TRACKER_ZOOM_IN_MAX_ERROR` | `0.18` | Maximum normalized center error per axis allowed for zoom-in. |
| `TRACKER_ZOOM_IN_STEP_MS` | `90` | Timed zoom-in pulse length. |
| `TRACKER_ZOOM_OUT_STEP_MS` | `140` | Timed zoom-out pulse length. |
| `TRACKER_ZOOM_COOLDOWN` | `3.0` | Minimum seconds between zoom-in commands. |
| `TRACKER_ZOOM_OUT_COOLDOWN` | `0.75` | Faster cooldown for zoom-out/context recovery. |
| `TRACKER_ZOOM_NOOP_BACKOFF` | `5.0` | Suppress further zoom-in after a completed zoom pulse that changed no reported zoom position. |
| `TRACKER_ZOOM_NOOP_EPSILON` | `0.03` | Maximum reported zoom-position delta treated as a no-op. |
| `TRACKER_COAST_TIME` | `0.20` | Short detector-miss coast interval. |
| `TRACKER_REACQUIRE_TIME` | `1.25` | Reacquisition interval before target is considered lost. |
| `TRACKER_HOME_TIMEOUT` | `3.0` | Lost duration before releasing target / returning home. |
| `TRACKER_TARGET_RELEASE_TIMEOUT` | `5.0` | Effective Rev 6.3 lost-target window before release/home; never shorter than `TRACKER_HOME_TIMEOUT`. |
| `TRACKER_HOME_PRESET` | `5` | Camera preset used as home. |
| `TRACKER_GOTO_HOME_ON_START` | `true` | Return to home preset when tracking starts. |
| `TRACKER_RETURN_HOME_ON_LOST` | `true` | Return to home after the target is lost. |

## Recommended Tracker Tuning

The included example Compose reflects the current Rev 6 controller. After upgrading from an older controller, run one manual continuous calibration with tracking stopped so Rev 6 can determine whether the camera really supports sub-speed ONVIF motion:

```bash
curl -s -X POST 'http://<gateway>:32168/v1/tracker/stop'
curl -s -X POST 'http://<gateway>:32168/v1/tracker/calibrate?mode=continuous'
curl -s -X POST 'http://<gateway>:32168/v1/tracker/start'
```

A successful calibration stores `onvif_benchmark.fractional_continuous`. If `usable=true`, the runtime uses calibrated fractional ONVIF velocities for normal moving-target servo tracking. If it is false, the controller safely holds rather than repeatedly quantizing fine corrections up to native speed 1; coarse native continuous motion remains available for genuine edge rescue.

When tuning, use `/v1/tracker/history`. `move_directly` events include target/predicted/command centers, velocity quality, lead suppression, effective gain, move distance, predicted ETA, confidence, and CGI latency. Hybrid chase history records entry reason, signed speed, frame error, outward-motion state, stops, and failures. `ptz_operation_complete`, `ptz_operation_timeout`, and `tracking_stopped_ptz_error` show whether camera motion settled safely.

`/v1/tracker/status` also exposes `stop_reason`, `tracker_task_running`, and `tracker_task_error`. If the tracker loop itself crashes, a subsequent `/v1/tracker/start` recreates the task instead of requiring a container restart.

## Updating

```bash
docker compose pull blueiris-ai
docker compose up -d --force-recreate blueiris-ai
```

GitHub Actions builds `latest` and a commit-SHA-tagged image on every push to `main`.


### Additional tracker reliability notes

The RTSP reader uses OpenCV/FFmpeg open and read timeouts (`TRACKER_RTSP_OPEN_TIMEOUT` / `TRACKER_RTSP_READ_TIMEOUT`) so a stalled stream can reconnect instead of blocking forever. Debug JPEGs are rendered only when `/v1/tracker/debug.jpg` is requested. Tracker sessions use a generation token so results from inference or PTZ calls that complete after `/stop` or `/home` are discarded. Lost-target home commands are retried a bounded number of times (`TRACKER_HOME_RETRY_ATTEMPTS`, `TRACKER_HOME_RETRY_DELAY`). Hybrid chase disables itself for the remainder of the session if tracking error grows materially for several consecutive frames, then falls back to normal `moveDirectly`.


### PTZ active calibration

`POST /v1/tracker/calibrate?mode=all` now calibrates the optical zoom map, native Dahua `moveDirectly` response/timing, native continuous pan/tilt speed response, and ONVIF pan/tilt capabilities. The motion calibration always runs with tracking stopped, repeatedly returns to the configured home preset, uses settled video frames to measure actual scene displacement, and returns home before releasing the camera.

With `TRACKER_AUTOSTART=true` and the default `TRACKER_CALIBRATE_ON_START=if_missing`, the API and health endpoint come up normally while a guarded startup calibration runs. Autotracking starts only after calibration completes. Persisted calibration prevents that full camera exercise from repeating on ordinary restarts. Use `if_stale` to refresh it after `TRACKER_CALIBRATION_MAX_AGE_DAYS`, `always` to recalibrate every startup, or `off` to disable startup calibration.


### Rev 6.6 tracking refinements

Rev 6.6 is based on correlated Rev 6.5 history/BVR evidence.

- **Predictive fast follow:** a mature fast target can enter fractional continuous tracking before the old 0.35 moving-error threshold when a 0.45 s projection crosses center and exits the 0.22 safe region.
- **Animal semantic continuity:** the default continuity group is now `dog,cat,bird,bear`; `person` remains intentionally excluded.
- **Static hotspot guard:** repeated home-view acquisition boxes are learned within the tracker session. After three separated bursts, that fingerprint needs extra confirmation plus either meaningful motion or stronger confidence. It is not a permanent ignore zone.
- **Local timestamps:** history `time`, status `status_time`, and `session_started` use a DST-aware local timezone. UTC companions remain available for precise BVR/tool correlation.
