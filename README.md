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
- **Bounded in-flight retargeting** so a slow native `moveDirectly` slew can have its destination replaced when the target reverses or is escaping toward a frame edge.
- **Escape-zone gain** for stronger corrections near frame edges while retaining the normal conservative gain near center.
- **Conservative auto-zoom** with configurable minimum/maximum optical zoom bounds.
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

If a `.pt` model exists without a matching TensorRT engine, the gateway can build the `.engine` automatically for the current TensorRT/CUDA environment.

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

The tracker uses the camera RTSP substream continuously and retains only the newest decoded frame. Tracker inference is opportunistic: regular gateway work keeps priority, and the tracker skips a frame rather than waiting behind normal Blue Iris inference.

For pan/tilt, the tracker uses Dahua/Amcrest `moveDirectly` 3D positioning. The camera's native positional move can take substantially longer than a manual joystick/continuous movement on some models, so the tracker does not assume a fixed travel time. It polls camera status and position until the camera reports idle and stable.

While a normal positional move is underway, detections continue. The tracker can issue a **bounded in-flight retarget** when the target clearly reverses direction, is escaping near a frame edge, or the tracking error grows sharply. This replaces the current destination without waiting for the old destination to finish. Retargeting is rate-limited and capped per movement chain so the camera is not flooded with commands.

Predictive velocity is **never learned from moving-camera frames**. In-flight retargets therefore use the target's current observed center only, not predictive lead. After the camera becomes stationary, one clean velocity baseline is taken and a minimum stationary-camera sample window must mature before predictive lead is trusted again.

Normal stationary-camera predictive lead is deliberately conservative. It is suppressed when the velocity sample is not mature, confidence is too low, the target is too small, or the detection touches the configured frame-edge margin. In those cases the camera still follows the target's current position; only the predictive lead is disabled.

An **escape zone** is entered when the target approaches a frame edge or the normalized error is very large. In that state, pan/tilt uses a stronger configurable gain (`TRACKER_ESCAPE_GAIN`) to reduce the chance that a fast target leaves the frame.

Target loss is paused while the camera is moving so global image motion does not look like the subject disappeared. When PTZ movement completes, the target-loss clock is restarted before normal coast/reacquisition timing resumes.

The tracker has been developed against a Dahua/Amcrest-style PTZ CGI camera, including an Amcrest IP2M-863EW-AI. Other cameras may require changes to the PTZ transport or coordinate behavior.

## Main Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `DEFAULT_MODEL` | `yolo11m` | Primary YOLO model stem. |
| `PRELOAD_MODELS` | blank | Comma-separated model stems to preload, or `all`. If blank, the default model is loaded. |
| `ALLOW_LAZY_LOAD` | `false` | Allow requested models to load on demand. |
| `HALF_PRECISION` | `true` | Use FP16 where supported. |
| `INFERENCE_TIMEOUT` | `8.0` | GPU inference timeout in seconds. |
| `RECOVERY_GRACE_PERIOD` | `4.0` | Grace period for an orphaned CUDA worker before the container self-terminates. |
| `ALERT_WEBHOOK_URL` | blank | Optional crash-alert webhook. |

## Tracker Environment Variables

### Camera / Stream

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKER_ENABLED` | `false` | Enable tracker service. |
| `TRACKER_AUTOSTART` | `false` | Start tracking automatically when the app starts. |
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
| `TRACKER_HOLD_CONF` | `0.22` | Confidence allowed for an already-locked target. |
| `TRACKER_REACQUIRE_CONF` | `0.35` | Confidence required while reacquiring. |
| `TRACKER_ACQUIRE_FRAMES` | `2` | Matching frames required before target lock. |
| `TRACKER_ASSOCIATION_IDLE_DISTANCE` | `0.22` | Same-class association distance while camera is stationary. |
| `TRACKER_ASSOCIATION_MOVING_DISTANCE` | `0.40` | Wider association gate around camera movement. |

### Pan / Tilt Prediction

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKER_MOVE_DIRECTLY_ENABLED` | `true` | Use Dahua/Amcrest 3D `moveDirectly`. |
| `TRACKER_MOVE_DEADZONE_X` | `0.14` | Horizontal normalized deadzone. |
| `TRACKER_MOVE_DEADZONE_Y` | `0.18` | Vertical normalized deadzone. |
| `TRACKER_MOVE_GAIN` | `0.60` | Fraction of projected frame error applied to a normal 3D move. |
| `TRACKER_LEAD_TIME` | `0.75` | Seconds of target-velocity lead before clamping. |
| `TRACKER_VELOCITY_MIN_SAMPLE_MS` | `100` | Minimum stationary-camera observation window before velocity is trusted. |
| `TRACKER_LEAD_MIN_CONF` | `0.50` | Suppress predictive lead below this confidence. |
| `TRACKER_LEAD_MIN_SPAN` | `0.08` | Suppress predictive lead for very small targets. |
| `TRACKER_LEAD_EDGE_MARGIN` | `0.02` | Suppress predictive lead when the bbox touches this fraction of a frame edge. |

Predictive lead is additionally hard-clamped to 20% of frame width/height per axis in code.

### In-Flight Retargeting / Escape Zone

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKER_INFLIGHT_RETARGET_ENABLED` | `true` | Allow a small number of destination replacements while `moveDirectly` is still moving. |
| `TRACKER_INFLIGHT_RETARGET_INTERVAL` | `0.30` | Minimum seconds between in-flight retarget commands. |
| `TRACKER_INFLIGHT_RETARGET_MAX` | `2` | Maximum retargets during one physical move chain. |
| `TRACKER_INFLIGHT_REVERSAL_ERROR` | `0.22` | Minimum normalized opposite-side error required to treat a crossing as a reversal. |
| `TRACKER_ESCAPE_ERROR` | `0.72` | Normalized error that enters escape mode even if the bbox is not yet near the edge. |
| `TRACKER_ESCAPE_GAIN` | `0.85` | Stronger pan/tilt gain used in escape mode. |

In-flight retargets deliberately use the current target center with **no predictive lead**, because bbox motion while the PTZ is slewing contains global camera motion. The tracker may retarget for `reversal`, `escape`, or `error_growth`, and history records the reason.

### PTZ Operation Settling

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKER_PTZ_STATUS_POLL_INTERVAL` | `0.12` | Seconds between camera PTZ status polls. Example Compose uses `0.06`. |
| `TRACKER_PTZ_OPERATION_TIMEOUT` | `4.0` | Hard timeout for the latest physical PTZ destination. |
| `TRACKER_POST_MOVE_FRAMES` | `1` | Fresh frames required after PTZ completion before another action. |
| `TRACKER_PTZ_HTTP_TIMEOUT` | `0.75` | HTTP timeout for camera CGI requests. |

### Auto-Zoom / Home

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKER_AUTOZOOM` | `true` | Enable bounded auto-zoom. |
| `TRACKER_ZOOM_WIDE_POSITION` | `5.12` | Camera-reported full-wide zoom position used by the current calibration. |
| `TRACKER_ZOOM_MIN_FACTOR` | `1.0` | Minimum tracking zoom factor. |
| `TRACKER_ZOOM_MAX_FACTOR` | `6.0` | Maximum tracking zoom factor. |
| `TRACKER_ZOOM_TARGET_MIN` | `0.18` | Zoom in when a centered target is smaller than this span. |
| `TRACKER_ZOOM_TARGET_MAX` | `0.42` | Zoom out when a centered target is larger than this span. |
| `TRACKER_ZOOM_IN_STEP_MS` | `90` | Timed zoom-in pulse length. |
| `TRACKER_ZOOM_OUT_STEP_MS` | `140` | Timed zoom-out pulse length. |
| `TRACKER_ZOOM_COOLDOWN` | `1.0` | Minimum seconds between zoom commands. |
| `TRACKER_COAST_TIME` | `0.20` | Short detector-miss coast interval. |
| `TRACKER_REACQUIRE_TIME` | `1.25` | Reacquisition interval before target is considered lost. |
| `TRACKER_HOME_TIMEOUT` | `3.0` | Lost duration before releasing target / returning home. |
| `TRACKER_HOME_PRESET` | `5` | Camera preset used as home. |
| `TRACKER_GOTO_HOME_ON_START` | `true` | Return to home preset when tracking starts. |
| `TRACKER_RETURN_HOME_ON_LOST` | `true` | Return to home after the target is lost. |

## Recommended Tracker Tuning

The included example Compose reflects the current tested tuning:

```text
TRACKER_FPS=15
TRACKER_MOVE_GAIN=0.60
TRACKER_LEAD_TIME=0.75
TRACKER_VELOCITY_MIN_SAMPLE_MS=100
TRACKER_LEAD_MIN_CONF=0.50
TRACKER_LEAD_MIN_SPAN=0.08
TRACKER_LEAD_EDGE_MARGIN=0.02
TRACKER_INFLIGHT_RETARGET_ENABLED=true
TRACKER_INFLIGHT_RETARGET_INTERVAL=0.30
TRACKER_INFLIGHT_RETARGET_MAX=2
TRACKER_INFLIGHT_REVERSAL_ERROR=0.22
TRACKER_ESCAPE_ERROR=0.72
TRACKER_ESCAPE_GAIN=0.85
TRACKER_PTZ_STATUS_POLL_INTERVAL=0.06
TRACKER_PTZ_OPERATION_TIMEOUT=4.0
TRACKER_POST_MOVE_FRAMES=1
TRACKER_ZOOM_MAX_FACTOR=6.0
```

When tuning, use `/v1/tracker/history` rather than guessing from visual behavior alone. Normal `move_directly` history events include the target center, predicted center, command center, velocity sample duration, predictive-lead validity/suppression reason, lead pixels, effective gain, HTTP duration, and target confidence.

In-flight destination changes are logged as `move_directly_retarget`, including the retarget reason, index, current target center, command center, gain, current normalized error, total move-chain elapsed time, and HTTP duration. The final `ptz_operation_complete` event includes both the time since the latest destination and the total `chain_elapsed_ms`, plus the number of retargets used.

## Updating

```bash
docker compose pull blueiris-ai
docker compose up -d --force-recreate blueiris-ai
```

GitHub Actions builds `latest` and a commit-SHA-tagged image on every push to `main`.
