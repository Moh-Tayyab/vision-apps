# App 1 — Carton Counter

FastAPI service: YOLO-based carton detection and pallet carton counting with
multi-angle fusion. API-only.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8001   # from this directory
# or
docker build -t carton-counter . && docker run -p 8001:8000 carton-counter
```

Weights: `MODEL_PATH` defaults to `models/yolo26n.pt` (COCO pre-trained YOLO26 —
senior's recommendation; "start here" = `models/yolo26m.pt`). The Docker image
bundles yolo26n + yolo26m; standard YOLO weight names auto-download if missing.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health + stream/ingest status |
| GET | `/model/info` | Model/backend info |
| POST | `/detect?confidence=` | Single image detection |
| POST | `/detect/visualize` | Detection with boxes drawn (JPEG) |
| POST | `/count/pan` | **Per-layer counting (MVP)** — vertical pan video or image sequence |
| POST | `/count` | 3-angle fusion count — fields `front`, `side`, `top` |
| POST | `/count/video` | Video processing (supports `method=per_layer_pan` or `multi_frame_voting`) |
| POST | `/pallet/angle` | 3D pallet orientation (pitch/roll/yaw) from one view |
| POST | `/pallet/correct` | Perspective-corrected pallet image |
| POST | `/ingest/frame` | Push one JPEG frame (mobile camera) |
| GET | `/stream` | MJPEG live view (ingest feed or local camera) |
| GET | `/stream/detect` | MJPEG with live detection boxes |

## Per-Layer Carton Counting (Phase 1 MVP)

In mixed-pallet environments where box dimensions vary within or across layers, standard `rows × layers` geometric multiplication fails. The Phase 1 MVP directly clusters cartons into physical layers along a normalized vertical axis, de-duplicates within each layer, and sums the layers:

$$\text{total\_count} = \sum_{l=0}^{L-1} \text{layer\_count}_l \quad (\text{never multiply})$$

### 6-Step Algorithm

1. **Frame Extraction**: Sample frames at regular temporal intervals (default: 0.5–0.8 s or ~8–15 frames total) preserving strict top-to-bottom temporal order ($t=0$ is the top of the stack).
2. **Carton Detection**: Run the YOLO / Roboflow carton detector on every sampled frame.
3. **Vertical Normalization**:
   - Camera motion between frame $t$ and $t+1$ is estimated from the median vertical displacement of high-IoU matched bounding boxes ($\text{IoU} > 0.4$ within search window).
   - If insufficient matched boxes exist, fallback to whole-image vertical phase correlation (`cv2.phaseCorrelate` with Hanning window).
   - Camera offsets are accumulated:
     $$\text{camera\_offset}[0] = 0, \quad \text{camera\_offset}[t+1] = \text{camera\_offset}[t] + \text{median\_shift}(t \to t+1)$$
   - Each detection is mapped to a shared vertical axis:
     $$\text{normalized\_y} = \text{camera\_offset}[\text{frame\_idx}] + (\text{pixel\_y\_center} - \text{frame\_height} / 2)$$
4. **Layer Clustering**:
   - Collect and sort all $\text{normalized\_y}$ coordinates across all frames.
   - Compute gaps between consecutive sorted values: $\text{gaps} = \Delta \text{normalized\_y}$.
   - Compute the hybrid gap threshold:
     $$\text{threshold} = \max(\text{gap\_multiplier} \times \text{median\_gap}, 0.6 \times \text{median\_box\_height})$$
     where $\text{gap\_multiplier}$ defaults to $1.7$ (configurable via query param).
   - Split into clusters wherever $\text{gap} > \text{threshold}$. Each cluster is one physical layer ordered from top to bottom.
5. **Intra-Layer De-duplication**:
   - Inside each layer cluster independently, de-duplicate bounding boxes seen across temporally adjacent/overlapping frames using shared-coordinate IoU ($\text{IoU} \ge 0.45$) and horizontal span alignment.
   - Never de-duplicate across different layers or faces.
6. **Final Count & Breakdown**:
   - Sum the de-duplicated counts of every layer.
   - Generate base64 annotated frames with horizontal layer boundary lines and layer index tags.

### Phase 1 Scope & Known Limitations
- **In Scope**: Single continuous vertical pan/tilt video of a single pallet face (top to bottom).
- **Out of Scope (Phase 1)**:
  - Multi-face fusion across multiple sides/angles of the pallet.
  - Multi-camera spatial calibration.
  - Fully occluded / internal hidden cartons behind the visible face.
  - Real-time streaming pan (batch video upload is targeted).

### Validation & Testing
- Run the full test suite:
  ```bash
  pytest tests/test_layer_counter.py -v
  ```
- **Held-Out Data Confirmation**: The 4 attached validation images (`media_*.jpg` showing Abbott cartons on a pallet) are strictly held-out validation data and were NEVER used for training or fine-tuning.


## Multi-angle fusion

Each view is de-duplicated independently (IoU clustering within the view);
the final total is the **median vote** across per-view counts
(`per_view_counts` in the response).

## Env vars

`MODEL_BACKEND` (local|roboflow|roboflow_workflow), `MODEL_PATH`, `CONF_THRESHOLD`,
`ROBOFLOW_MODEL_URL`, `ROBOFLOW_API_KEY`, `ROBOFLOW_API_URL`, `ROBOFLOW_WORKSPACE`,
`ROBOFLOW_WORKFLOW`, `ROBOFLOW_WORKFLOW_OUTPUT`, `VIDEO_SOURCE`, `STREAM_FPS`, `PORT`.

## Backends

| Backend | What | When to use |
|---------|------|-------------|
| `roboflow` | **RECOMMENDED** — hosted RF-DETR medium v7, full-image inference | Default. mAP@50=97.9, count error 8.1% on GT set |
| `local` | Ultralytics YOLO on-device (COCO pre-trained) | Offline fallback; generic boxes only |
| `roboflow_workflow` | SAHI sliced inference via saved Roboflow Workflow | Legacy — very dense scenes where full-frame recall drops |

### Model history

| Version | Training data | mAP@50 | Count error (14-image GT set) |
|---------|--------------|--------|-------------------------------|
| v5 | 14 pallet images | 12.3% | ~50% (full-image), SAHI needed for photos |
| v7 | 1051 images = 14 pallet + 1037 public cardboard-box dataset (class remapped `box`→`carton`) | **97.9%** | **8.1% @ fixed conf 0.36** |

v7 confidence is well calibrated (~0.95 on true cartons): a single global
threshold works across sources, and 8/14 GT images count EXACTLY at
conf ≥ 0.36 with per-image tuning reaching near-exact on all.

The old bottleneck was data, not architecture: merging one public
single-class box dataset lifted every metric dramatically. Next accuracy
lever remains in-domain labeled frames from the real production camera.

### Validation on full GT dataset (14 images)

With v7 the recommended setup is `roboflow` full-image @ conf 0.36:

- 8/14 images count EXACTLY; video frames within ±1
- Remaining misses are extreme scenes (very tall occluded stack over-counted,
  one dense pallet under-counted) — in-domain labeled data will close this
- SAHI workflow remains available (`roboflow_workflow` backend) but is no
  longer needed: with a well-calibrated model it adds duplicates instead of
  recall

Production guidance unchanged: calibrate `CONF_THRESHOLD` once against a few
hand-counted scenes from the real camera, then keep it fixed.
