# Phase 2 — Individual Camera Tripwire Validation & Geometry Audit

## 1. Executive Summary

This document reports the geometry, coordinate systems, tripwire positions, crossing directions, and boundary stability for all three cameras in the `carton-counter` multi-camera setup:
- `cam_top` (Overhead view)
- `cam_side` (Lateral side view)
- `cam_flank` (Interior oblique flank view)

The geometry was audited against warehouse video recordings (`apps/truck_loading_detection/recorded_video/`), `config/camera_models.yaml`, and the implementation in `apps/truck_loading_detection/three_camera_system/corridor.py`.

---

## 2. Independent Camera Geometric Specifications

### 2.1 Overhead Camera (`cam_top`)
- **Native Video Resolution:** $1024 \times 576$ (scaled / normalized) or $1280 \times 720$.
- **Coordinate System:** Origin $(0, 0)$ at top-left. Horizontal axis $X \in [0, W]$, Vertical axis $Y \in [0, H]$.
- **Viewing Perspective:** Top-down vertical bird's-eye view looking down on the loading doorway threshold.
- **Physical Motion Direction:** Workers carry cartons from the staging dock (right, $X > 512$) across the threshold into the truck trailer (left, $X < 512$).
- **Tripwire Orientation:** `vertical` (evaluates coordinate $X$).
- **Tripwire Normalized Position (`line_pos_frac`):** $0.50$ ($X = 512\text{ px}$ at $1024\text{w}$).
- **Direction Sign (`direction_sign`):** $-1$ (Decreasing $X$ corresponds to forward loading `"IN"` into truck).
- **Hysteresis Buffer:** $\pm 4\%$ of frame width ($\approx \pm 41\text{ px}$). Coordinate $\le 471$ is `"after"`, $\ge 553$ is `"before"`.
- **Minimum Displacement Threshold:** $12.0\text{ px}$.

### 2.2 Lateral Side Camera (`cam_side`)
- **Native Video Resolution:** $1024 \times 576$ / $1280 \times 720$.
- **Coordinate System:** Origin $(0, 0)$ at top-left.
- **Viewing Perspective:** Side-elevation view perpendicular to worker walking path. Shows worker carrying height, carton elevation, and stack boundaries.
- **Physical Motion Direction:** Workers walk from the loading staging area on the left ($X < 512$) through the threshold door toward the truck on the right ($X > 512$).
- **Tripwire Orientation:** `vertical` (evaluates coordinate $X$).
- **Tripwire Normalized Position (`line_pos_frac`):** $0.50$ ($X = 512\text{ px}$ at $1024\text{w}$).
- **Direction Sign (`direction_sign`):** $+1$ (Increasing $X$ corresponds to forward loading `"IN"` into truck).
- **Hysteresis Buffer:** $\pm 4\%$ of frame width ($\approx \pm 41\text{ px}$). Coordinate $\le 471$ is `"before"`, $\ge 553$ is `"after"`.
- **Minimum Displacement Threshold:** $12.0\text{ px}$.

### 2.3 Oblique Flank Camera (`cam_flank`)
- **Native Video Resolution:** $1024 \times 576$ (or $476 \times 848$ in legacy mobile/portrait feeds).
- **Coordinate System:** Origin $(0, 0)$ at top-left.
- **Viewing Perspective:** Oblique interior/flank angle mounted near trailer edge, viewing cartons entering the doorway interior.
- **Physical Motion Direction:** Cartons move from entrance doorway ($X \approx 450\text{--}500$) toward trailer interior wall ($X > 560$).
- **Tripwire Orientation:** `vertical` (evaluates coordinate $X$).
- **Tripwire Normalized Position (`line_pos_frac`):** $0.55$ ($X = 563\text{ px}$ at $1024\text{w}$).
- **Direction Sign (`direction_sign`):** $+1$ (Increasing $X$ corresponds to forward loading `"IN"` into truck).
- **Hysteresis Buffer:** $\pm 4\%$ of frame width ($\approx \pm 41\text{ px}$).
- **Minimum Displacement Threshold:** $12.0\text{ px}$.

---

## 3. Crossing Logic, Boundary Conditions & Stability

### 3.1 Centroid vs. Boundary Evaluation
Crossing evaluation is performed on the tracking centroid:
$$C = \left(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2}\right)$$
The side classification function `determine_side(center, frame_dims)` evaluates:
$$\text{coord} = \begin{cases} C_x & \text{if orientation} = \text{"vertical"} \\ C_y & \text{if orientation} = \text{"horizontal"} \end{cases}$$
$$\text{line\_px} = \text{line\_pos\_frac} \times \text{dim}, \quad \text{hyst\_px} = \text{hysteresis\_frac} \times \text{dim}$$

Zone boundaries:
- If $\text{coord} \le (\text{line\_px} - \text{hyst\_px})$:
  - Return `"before"` if $\text{direction\_sign} > 0$ else `"after"`.
- If $\text{coord} \ge (\text{line\_px} + \text{hyst\_px})$:
  - Return `"after"` if $\text{direction\_sign} > 0$ else `"before"`.
- If inside $[\text{line\_px} - \text{hyst\_px}, \text{line\_px} + \text{hyst\_px}]$:
  - Return `None` (Indeterminate Hysteresis Buffer Zone).

### 3.2 Hovering & Boundary Jitter Suppression
- If a worker stops or hovers directly on the doorway line, the track centroid remains in the hysteresis buffer zone ($\text{side} = \text{None}$).
- The state machine does not emit any crossing candidate while in this deadband.
- Only when the carton centroid emerges fully into the opposing zone ($\text{coord} \ge \text{line\_px} + \text{hyst\_px}$) AND displacement $\ge 12\text{ px}$ is a candidate generated.

### 3.3 Directional Reversal Handling
- If a worker steps into the doorway band from `"before"`, reaches the line, and then retreats backward to `"before"`, the side transition from `"before"` $\to$ `"after"` never occurs.
- If a candidate had reached the transit corridor in `CorridorStateMachine`, retreating into the entry zone triggers `REVERSED_CANCELLED` and emits zero counts.

### 3.4 Worker Occlusion Stitching Across Tripwires
- In real warehouse footage, a worker's torso or head may occlude the carton precisely as it traverses the tripwire boundary.
- `PerCameraTripwire.update()` tracks approach history:
  1. Cartons entering the `"before"` zone are recorded in `_recent_approaches` with track ID, timestamp, confidence, and vertical position ($y$).
  2. If the track is dropped during threshold occlusion and a new tracklet appears in the `"after"` zone within transit window ($0.25\text{--}3.2\text{ s}$) in the same lane ($\Delta y \le 140\text{ px}$), it is paired.
  3. A single candidate with `occlusion_stitched = True` is emitted.
  4. Refractory guard ($1.2\text{ s}$ refractory window within $80\text{ px}$) prevents double-firing.

---

## 4. Test Evidence & Validation Results

The tripwire geometry is verified by existing regression tests in `tests/test_multi_camera_synchronization_and_counting.py`:
- `test_tripwire_direction_signs_and_geometry`: Verifies orientation and direction signs across all three cameras (`top=-1`, `side=+1`, `flank=+1`).
- `test_regression_07_carton_near_line_without_crossing`: Verifies that stationary hovering near the boundary yields zero counts.
- `test_regression_05_forward_then_backward_crossing`: Verifies that moving forward then backward produces net count 0 (+1 IN, +1 OUT).
- `test_regression_18_automatic_line_placement_resolutions`: Verifies that scaling from $640\times 360$ to $1920\times 1080$ scales tripwire pixel boundaries correctly via `line_pos_frac`.

All tripwire validation tests pass with 100% compliance.
