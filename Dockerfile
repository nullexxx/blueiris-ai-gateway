FROM ultralytics/ultralytics:latest

WORKDIR /app

# onnxruntime-gpu[cuda] pulls the pip CUDA libraries ONNX Runtime needs that the PyTorch base image does not
# ship (notably cuRAND). Without them the CUDA execution provider fails to load and ONNX models silently run on CPU.
RUN pip install --no-cache-dir fastapi uvicorn python-multipart tqdm tensorrt onvif-zeep-async "onnx>=1.23.2" "onnxruntime-gpu[cuda]>=1.31.0"

# Fail the build if any CUDA library ONNX Runtime requires is missing (no GPU is needed for this check).
RUN python3 -c "import os, sys, nvidia, onnxruntime as ort; \
required = [p for p in ort._get_nvidia_dll_paths(False, True, False) if not any(k in p[-1] for k in ('cufft', 'nvrtc'))]; \
missing = [p[-1] for p in required if not any(os.path.isfile(os.path.join(base, *p[1:])) for base in nvidia.__path__)]; \
print('onnxruntime', ort.__version__, 'CUDA libs:', 'MISSING ' + ', '.join(missing) if missing else 'ok'); \
sys.exit(1 if missing else 0)"
RUN pip install --no-cache-dir --no-deps facenet-pytorch

# Pre-cache the VGGFace2 model weights into the container image
RUN python3 -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2')"

# Runtime modules: core API/tracker plus the smart-PTZ calibration and filtering layers.
COPY app.py /app/app.py
COPY tracker.py /app/tracker.py
COPY tracker_config.py /app/tracker_config.py
COPY tracker_models.py /app/tracker_models.py
COPY tracker_policy.py /app/tracker_policy.py
COPY tracker_calibration.py /app/tracker_calibration.py
COPY tracker_servo.py /app/tracker_servo.py
COPY ptz_enhancements.py /app/ptz_enhancements.py
COPY ptz_phase2.py /app/ptz_phase2.py
COPY ptz_active_calibration.py /app/ptz_active_calibration.py

EXPOSE 32168

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "32168"]
