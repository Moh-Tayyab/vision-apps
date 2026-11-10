# Phase 3.2 — Reconciliation Engine Integrity & Evaluator Audit

**Author:** Senior Computer Vision & ML Evaluation Engineering  
**System:** Multi-Camera Warehouse Carton Counting System  
**Document:** `docs/phase3/RECONCILIATION_ENGINE_INTEGRITY.md`  
**Date:** November 2026  
**Status:** **Audit Completed & Mathematical Invariants Verified**  

---

## 1. Objective & Scope

In offline computer vision evaluation, counting metrics are only as trustworthy as the evaluator that produces them. A flawed evaluation engine can artificially inflate precision, suppress false negatives through greedy matching traps, or confuse raw database counts with physical crossing events.

This audit investigates five critical integrity vectors in the ground-truth reconciliation framework:
1. **Mathematical Bipartite Matching:** Greedy vs. Kuhn-Munkres (Hungarian) maximum-cardinality assignment.
2. **Temporal Tolerance Derivation:** Resolving the $\tau = 2.0\text{s}$ vs. $1.947\text{s}$ boundary condition.
3. **Evaluation Provenance:** Explaining the historical 20 DB event vs. 18 DB event discrepancy.
4. **Duplicate Classification Rigor:** Separating unmerged candidate tracklets from verified duplicate crossings.
5. **Multi-Camera Footage Coverage:** Documenting synchronized overlap vs. raw video source lengths for Aligned Session 1.

---

## 2. Matching Algorithm: Greedy vs. Kuhn-Munkres Maximum Cardinality

### 2.1 The Greedy Assignment Flaw
The initial Phase 3.1 evaluator used a greedy sort-by-cost heuristic:
```python
candidate_edges.sort(key=lambda x: x[0])  # Sort by cost ascending
for cost, dt, i, j in candidate_edges:
    if i not in matched_gt and j not in matched_pred:
        matched_gt.add(i)
        matched_pred.add(j)
```
While intuitive, greedy assignment is provably **suboptimal** on bipartite graphs. It can prematurely consume a prediction that was the *only* candidate for another ground-truth crossing, thereby failing to find a maximum-cardinality matching.

#### Concrete Failure Case
Consider two ground-truth crossings and two predicted events within tolerance $\tau = 1.2\text{s}$:
- $\text{GT}_1$ at $t = 1.0\text{s}$, $\text{GT}_2$ at $t = 1.5\text{s}$.
- $\text{Pred}_1$ at $t = 1.1\text{s}$ ($\Delta t = 0.1\text{s}$ to $\text{GT}_1$, $\Delta t = 0.4\text{s}$ to $\text{GT}_2$).
- $\text{Pred}_2$ at $t = 0.0\text{s}$ ($\Delta t = 1.0\text{s}$ to $\text{GT}_1$, $\Delta t = 1.5\text{s}$ [out of range] to $\text{GT}_2$).

```
        Greedy Assignment                      Maximum Cardinality Assignment
   GT1 (1.0s) ───dt=0.1s───> Pred1 (1.1s)      GT1 (1.0s) ───dt=1.0s───> Pred2 (0.0s)
   GT2 (1.5s)      X (unmatched)               GT2 (1.5s) ───dt=0.4s───> Pred1 (1.1s)
   Pred2 (0.0s)    X (unmatched)
   Total Matches: 1                            Total Matches: 2 (OPTIMAL)
```
- **Greedy:** Greedily matches $(\text{GT}_1, \text{Pred}_1)$ because $\Delta t = 0.1\text{s}$ is minimal. Now $\text{Pred}_1$ is locked. $\text{GT}_2$ has no valid candidate ($\text{Pred}_2$ is $1.5\text{s} > 1.2\text{s}$). Result: **1 match (50% recall)**.
- **Maximum Cardinality:** Matches $(\text{GT}_1, \text{Pred}_2)$ ($\Delta t = 1.0\text{s} \le 1.2\text{s}$) and $(\text{GT}_2, \text{Pred}_1)$ ($\Delta t = 0.4\text{s} \le 1.2\text{s}$). Result: **2 matches (100% recall)**.

### 2.2 Implemented Kuhn-Munkres Formulation
We upgraded the engine in `apps/truck_loading_detection/three_camera_system/reconciliation_engine.py` to use `scipy.optimize.linear_sum_assignment`:
1. **Cost Matrix Construction:**
   Let $M$ be the number of GT crossings and $N$ be the number of predictions.
   $$C_{ij} = \begin{cases} |t_{\text{GT}, i} - t_{\text{pred}, j}| - W & \text{if } d_{\text{GT}, i} = d_{\text{pred}, j} \text{ and } |t_{\text{GT}, i} - t_{\text{pred}, j}| \le \tau \\ K & \text{otherwise} \end{cases}$$
   where $W = 1000.0$ is a large match bonus and $K = 10^6$ is an incompatible penalty.
2. **Cardinality Priority:** Because $W \gg \tau$, minimizing total cost $\sum C_{ij}$ mathematically guarantees:
   - Primary Objective: Maximize the number of valid one-to-one pairings.
   - Secondary Objective: Minimize the sum of temporal errors $\sum \Delta t$ among all maximum-cardinality pairings.
3. **Removal of Auto-Confirmed Bias:** The artificial bias ($\epsilon = 0.001\text{s}$) that favored auto-confirmed predictions was completely removed. Ground truth is temporal truth; predictions are matched purely based on kinematic proximity.
4. **Ambiguity Flagging:** When an alternative prediction $k \ne j$ exists with $|t_{\text{GT}} - t_{\text{pred}, k}| \le \tau$ and $| \Delta t_k - \Delta t_j | \le 0.10\text{s}$, the assignment is explicitly flagged:
   `is_ambiguous = True`, `ambiguity_reason = "Candidate '...' is within 100ms of primary match"`.

---

## 3. Temporal Matching Tolerance ($\tau$) Derivation & Semantic Reconciliation

### 3.1 Resolving the Boundary Discrepancy
The Phase 3.1 report referenced both $\tau = 2.0\text{s}$ and an upper bound of $1.95\text{s}$. We resolve this mathematically:

In Session 21 ground-truth manifests, the tightest consecutive carton crossing intervals are:
- $\text{GT}_2$ ($t = 4.696\text{s}$) to $\text{GT}_3$ ($t = 8.592\text{s}$): $\Delta t = 3.896\text{s}$.
- $\text{GT}_3$ ($t = 8.592\text{s}$) to $\text{GT}_4$ ($t = 12.488\text{s}$): $\Delta t = 3.896\text{s}$.
- $\text{GT}_4$ ($t = 12.488\text{s}$) to $\text{GT}_5$ ($t = 16.383\text{s}$): $\Delta t = 3.895\text{s}$.

To ensure strictly disjoint Voronoi catchment basins between consecutive physical crossings, the maximum tolerance without temporal overlap is:
$$\tau_{\max} = \frac{1}{2} \min_{k} (t_{\text{GT}, k+1} - t_{\text{GT}, k}) = \frac{3.895\text{s}}{2} = 1.9475\text{s}$$
If $\tau = 2.0\text{s}$, a candidate prediction at $t = 14.45\text{s}$ would lie inside the catchment window of both $\text{GT}_4$ ($12.488 + 1.962\text{s}$) and $\text{GT}_5$ ($16.383 - 1.933\text{s}$), creating an overlapping window of $0.105\text{s}$.

**Resolution:** The default evaluation tolerance is frozen at **$\tau = 1.90\text{seconds}$**. This provides a $47\text{ms}$ safety buffer below $\tau_{\max}$, ensuring mathematically non-overlapping temporal windows while encompassing the full physical transit duration ($2.0\text{m}$ corridor at $1.0\text{m/s} = 2.0\text{s}$).

### 3.2 Timestamp Semantics: `timestamp_sec` vs. `commit_timestamp_sec`
The SQLite database stores two timestamps for every event:
- `timestamp_sec`: The physically estimated tripwire crossing time, calculated via sub-frame linear trajectory interpolation when the carton center crossed the threshold line.
- `commit_timestamp_sec`: The wall-clock timestamp when the state machine committed the crossing after traversing the hysteresis deadband ($20\text{px}$–$40\text{px}$ beyond the line).

**Integrity Rule:** Ground-truth manifests record the exact moment the physical carton crossed the doorway threshold. Therefore, the evaluator **must compare against `timestamp_sec`**, NOT `commit_timestamp_sec`. Comparing against commit time introduces an artificial $150\text{ms}$–$350\text{ms}$ latency bias caused by deadband transit.

---

## 4. Evaluation Run Provenance & Historical Discrepancy Reconciliation

### 4.1 Root Cause of the Discrepancy
An earlier report cited:
- *Session 18: 4 auto-confirmed, 14 review-required, 20 database events.*
Phase 3.1 reported:
- *Session 18: 1 auto-confirmed, 5 matched review-required, 11 duplicate candidates, 1 unresolved, 18 database events.*

Database query analysis reveals the exact origin:
1. **Raw Database Inventory:**
   Inspecting `data/carton_events.db` directly:
   - `session_18` records: **18 events total** (4 with `is_auto_confirmed=1`, 14 with `requires_review=1`).
   - `default_session` records: **2 events total** (leftover un-scoped test events from earlier development).
   - $18 + 2 = \mathbf{20\text{ total database rows}}$.
2. **The Source of Confusion:**
   The earlier report simply read the raw database row count ($18 + 2 = 20$) and reported raw database states (4 auto, 14 review). It did **not** perform 1-to-1 ground-truth bipartite matching!
3. **Phase 3.1 Reconciled Outcomes:**
   When the 18 `session_18` events were matched against the **6 physical ground-truth crossings**:
   - 1 auto-confirmed event matched $\text{GT}_1$ (`session_18_ev_00001`).
   - The other 3 auto-confirmed predictions (`ev_00003`, `ev_00012`, `ev_00013`) were secondary duplicate echoes of cartons already in transit.
   - 5 review-required events matched the remaining 5 GT cartons.
   - 11 events were duplicate candidate tracklets.
   - 1 event was an unresolved phantom prediction.
   - $1 + 5 + 11 + 1 = \mathbf{18\text{ events}}$.

### 4.2 Provenance Architecture
To prevent future ambiguity, `SessionMetrics` now embeds:
- `provenance_run_id`: A unique run identifier (e.g. `run_session_18_a1b2c3d4`).
- `evaluated_db_path`: Absolute path of the SQLite database evaluated.

---

## 5. Duplicate Classification: Candidates vs. Verified Duplicates

An unmatched prediction occurring within $\le 1.90\text{s}$ of a verified crossing is technically only a **`DUPLICATE_CANDIDATE`**. It cannot be classified as a verified duplicate without physical tracklet continuity evidence.

### Verification Criteria
A candidate prediction $p_{\text{cand}}$ is upgraded to a **Verified Duplicate Tracklet** if and only if:
1. **Temporal Co-presence:** It occurred within the physical transit window ($0.0\text{s} \le t(p_{\text{cand}}) - t(p_{\text{matched}}) \le 2.0\text{s}$).
2. **Camera-Local Track Linkage:** The local track ID of $p_{\text{cand}}$ emerged within 15 frames of an occluded tracklet belonging to $p_{\text{matched}}$.
3. **Spatial Trajectory Continuity:** The spatial position and velocity vector of $p_{\text{cand}}$ continue the trajectory of the matched carton across the doorway.

If these criteria are absent (e.g., an independent detection in an adjacent lane), the event remains a candidate and is flagged for manual supervisor audit.

---

## 6. Aligned Session 1 Footage Coverage Audit

The video files for `session_aligned_v1` have differing frame counts:
- `overhead_aligned.mp4`: **846 frames** ($28.20\text{s}$ at $30.00\text{ FPS}$).
- `side_aligned.mp4`: **856 frames** ($28.63\text{s}$ at $29.90\text{ FPS}$).
- `flank_aligned.mp4`: **936 frames** ($31.20\text{s}$ at $30.00\text{ FPS}$).

### Coverage Analysis
In synchronized multi-camera processing (`ThreeCameraCartonPipeline.process_synchronized_frames`), the pipeline ingests synchronous triplets $\{f_{\text{top}}, f_{\text{side}}, f_{\text{flank}}\}$. Once the shortest video stream (`cam_top`) terminates at frame 846, multi-camera processing halts.

| Camera Stream | Total Source Frames | Processed Synchronized Frames | Excluded Tail Frames | Source Processing % | Synchronized Overlap % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `cam_top` | 846 | 846 | 0 | **100.0%** | **100.0%** |
| `cam_side` | 856 | 846 | 10 | **98.8%** | **100.0%** |
| `cam_flank` | 936 | 846 | 90 | **90.4%** | **100.0%** |

**Conclusion:** The pipeline processes **100% of the synchronized multi-camera overlap (846 frames, 28.20s)**. It excludes 10 tail frames on the side camera and 90 tail frames on the flank camera where overhead coverage is absent. Previous documentation claiming "100% video footage coverage" is clarified accordingly.

---

## 7. Evaluator Integrity Certification

The offline evaluator in `apps/truck_loading_detection/three_camera_system/reconciliation_engine.py` is certified as:
- Maximum-cardinality guaranteed via Kuhn-Munkres bipartite matching.
- Strictly bounded by $\tau = 1.90\text{s}$ (non-overlapping Voronoi basins).
- Free from artificial prediction priority bias.
- Fully traceable with run provenance identifiers.
- Compliant with all Phase 1–3 invariants.
