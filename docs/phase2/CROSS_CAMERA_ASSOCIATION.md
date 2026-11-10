# Phase 2 — Cross-Camera Physical Carton Association

## 1. Executive Summary & Core Requirements

In a multi-camera truck loading doorway system, the same physical carton is simultaneously or sequentially visible across three distinct perspectives:
- Overhead view (`cam_top`)
- Side elevation view (`cam_side`)
- Interior flank view (`cam_flank`)

### Core Distinctions
The system strictly enforces clear boundaries between four fundamental entities:
1. **Camera Detection:** An instantaneous 2D bounding box and segmentation mask in a single camera frame (e.g., `(x1, y1, x2, y2)` at frame 102 in `cam_top`).
2. **Camera-Specific Tracklet (`CartonTracklet`):** A temporal sequence of detections bound by a single camera's ByteTrack/BoT-SORT tracker (e.g., `cam_top:14`).
3. **Physical Carton (`PhysicalCarton`):** The persistent real-world object (`phys_carton_00003`) moving through 3D space, which may bind multiple camera tracklets (`cam_top:14`, `cam_side:8`, `cam_flank:11`) across its doorway transit.
4. **Logical Carton Crossing Event (`CommittedCrossingEvent`):** The atomic accounting event committing $+1$ or $-1$ count once the physical carton crosses the threshold boundary.

**Strict Constraint:** A local camera tracker ID is NEVER assumed to be globally unique or 1-to-1 with a physical carton.

---

## 2. Association Architecture & Mathematical Formulation

### 2.1 Multi-Stage Association Pipeline
Cross-camera association operates through two complementary layers:
1. **Frame-Level Spatial Clustering (`CrossCameraAssociator`):**
   - Operates on instantaneous active tracklets across cameras at timestamp $t_{\text{global}}$.
   - Projects camera centroid coordinates $(u, v)$ to the calibrated dock floor plane $(X_{\text{dock}}, Y_{\text{dock}})$ using homography matrix $H$:
     $$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} \sim H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
   - Computes pairwise Euclidean dock distance:
     $$D(T_i, T_j) = \sqrt{(X_i - X_j)^2 + (Y_i - Y_j)^2}$$
   - Solves 1-to-1 association with maximum dock distance threshold $D_{\max} = 0.60\text{ m}$.
2. **Corridor Transit Coalescence (`CrossingWindow` & `CartonCrossingStateMachine`):**
   - Because a worker walking at $1.0\text{ m/s}$ takes $400\text{--}800\text{ ms}$ to cross the visual fields of the top, side, and flank cameras, observations do not occur at the exact same microsecond.
   - Crossing candidates within temporal transit tolerance ($0.75\text{--}1.5\text{ s}$) are associated to the existing `PhysicalCarton` record.

### 2.2 Mathematical Association Criteria
An observation $(c_{\text{cam}}, \text{loc\_id}, \mathbf{x}, t)$ matches an existing `PhysicalCarton` $P$ if any of the following mutually verified conditions hold:
1. **Direct Tracklet Match:**
   $$\text{loc\_id} \in P.\text{associated\_tracks}[c_{\text{cam}}]$$
2. **Calibrated Dock Plane Continuity:**
   $$\Delta t = t - P.t_{\text{last}} \le 2.2\text{ s} \quad \text{and} \quad \|\mathbf{x}_{\text{dock}} - P.\mathbf{x}_{\text{dock}}\| \le 0.60\text{ m}$$
3. **Camera Spatial Continuity:**
   $$\Delta t \le 2.2\text{ s} \quad \text{and} \quad \|\mathbf{x} - P.\mathbf{x}_{c_{\text{cam}}}\| \le 140\text{ px}$$
4. **Active Committed Corridor Protection:**
   $$0.0\le (t - P.t_{\text{commit}}) \le 1.5\text{ s} \quad \text{and} \quad \text{direction} = P.\text{last\_committed\_direction}$$
   *(Guard: never absorbs distinct tracklets on the same camera or different vertical lanes).*

---

## 3. Handling Complex Warehouse Edge Cases

| Scenario | System Handling & Guarantee | Decision State |
|---|---|---|
| **Same carton visible in all 3 cameras** | Associated into 1 `PhysicalCarton` via calibrated dock projection and transit window. Evaluated by `ThreeCameraConsensusPolicy` as `THREE_CAMERA_UNANIMOUS`. | `COUNT_COMMITTED` (+1) |
| **Same carton visible in only 2 cameras** (e.g. flank blocked by pallet stack) | Associated into 1 `PhysicalCarton`. Evaluated as `TWO_CAMERA_AGREEMENT`. Validated direction and displacement. | `COUNT_COMMITTED` (+1) |
| **Carton visible in only 1 camera** (severe occlusion) | Associated as single-camera observation. Fails multi-view corroboration. | `REVIEW_REQUIRED` (Excluded from confirmed count) |
| **Two workers carrying cartons close together** (parallel lanes) | Centroids separated along $Y$ axis ($\Delta y > 120\text{ px}$ or distinct tracklets on same camera). Distance exceeds $D_{\max}$. Kept as 2 distinct `PhysicalCarton` entities. | Two distinct `COUNT_COMMITTED` events (+2) |
| **Worker body occludes carton at line, re-emerges with new tracker ID** | `PerCameraTripwire` pairs approach history with re-emergence within lane tolerance ($\le 140\text{ px}$, $0.25\text{--}3.2\text{ s}$). Stitches into same physical carton. | Exactly 1 `COUNT_COMMITTED` (+1) |
| **Directional Reversal / Hovering** | Worker approaches line, stops, and retreats to staging dock. Direction sign reverses or displacement $< 12\text{ px}$. | `REVERSED_CANCELLED` (0 count) |
| **Dynamic Corroboration** | Initial observation on Cam 1 flags `REVIEW_REQUIRED`. Cam 2 detects the carton $350\text{ ms}$ later. The state machine correlates the two observations, upgrades the event to `COUNT_COMMITTED` without creating a duplicate event. | Upgraded to `COUNT_COMMITTED` |

---

## 4. Limitations & Safeguards

1. **Uncalibrated Extrinsics Fallback:** If camera extrinsic calibration matrices are unavailable, spatial association falls back to temporal sliding windows and lane bounding boxes. In extreme congestion (e.g., 3 workers in doorway simultaneously without calibration), the system conservatively marks clusters as `is_ambiguous = True` and routes them to `REVIEW_REQUIRED` rather than risking a false count.
2. **Review Isolation:** Under no circumstances does a single uncorroborated camera observation auto-increment the production count.
