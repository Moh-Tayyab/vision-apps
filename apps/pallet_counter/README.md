# Pallet Counter — Row-Wise Carton Counting

Counts every carton on a pallet from **two still photos**, row by row.

Port `8004`. Images only — this app has no live video path by design.

---

## Why two photos

A pallet is a stack of rows, each row a rectangular grid of cartons. One photo
only ever shows one side of that grid, so counting visible faces cannot give the
pallet total — most cartons are hidden behind others.

A pallet has four side faces. Opposite faces (1 and 3) show the *same* dimension,
so they are redundant. **Adjacent faces, 90° apart, give both dimensions:**

```
        TOP VIEW of one row
        ┌────┬────┬────┬────┐
  depth │    │    │    │    │   view B counts this = 3
    ↕   ├────┼────┼────┼────┤
        │    │    │    │    │
        └────┴────┴────┴────┘
          ←   width   →
        view A counts this = 4

        row total = 4 × 3 = 12
```

```
row total    = faces in view A row  ×  faces in view B row
pallet total = Σ row totals
```

Rows are counted separately rather than as one multiplication, because rows are
not always identical — a top row may hold fewer cartons than the rows below it.

---

## Usage

### Dashboard

```
http://localhost:8004/
```

Upload both photos, press **Count Pallet**. Returns the total, a per-row table,
and both images annotated with the detected rows.

### API

```bash
curl -X POST "http://localhost:8004/count/pallet?confidence=0.36" \
  -F "front=@view_a.jpeg" \
  -F "side=@view_b.jpeg"
```

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service status and detector configuration |
| `GET` | `/model/info` | Detection backend and last inference time |
| `POST` | `/count/pallet` | Count from two images (`front`, `side`) |
| `GET` | `/` | Upload dashboard |

`POST /count/pallet` parameters:

- `front`, `side` — the two images (required)
- `confidence` — detection threshold, default `0.36`
- `annotate` — include annotated images, default `true`

Response:

```json
{
  "total_count": 48,
  "rows_counted": 4,
  "rows": [
    {"row": 1, "front_count": 3, "side_count": 4, "row_total": 12, "formula": "3 x 4 = 12"}
  ],
  "front_view": {"rows_detected": 4, "total_faces": 14, "tilt_deg": 1.0, "rows": [...]},
  "side_view":  {"rows_detected": 4, "total_faces": 14, "tilt_deg": 8.0, "rows": [...]},
  "warnings": [],
  "front_annotated": "data:image/jpeg;base64,...",
  "side_annotated":  "data:image/jpeg;base64,...",
  "summary_panel":   "data:image/jpeg;base64,..."
}
```

---

## How it works

**1. Detection** — [detector.py](detector.py) runs the hosted Roboflow carton
model on each image and drops boxes that cannot be a single carton face
(too large, too small, implausible aspect ratio).

**2. Duplicate suppression** — [`suppress_duplicates`](row_engine.py) removes a
second box emitted for one carton. The IoU threshold is deliberately high (0.45):
in an oblique photo, axis-aligned boxes around neighbouring cartons clip each
other's corners harmlessly, and a lower threshold would delete real cartons.

**3. Tilt estimation** — a photo taken square-on puts every carton in a row at
the same image height. An oblique photo makes the row recede, so cartons in one
row sit at visibly different heights, and clustering on raw `y` merges or splits
rows. [`estimate_tilt`](row_engine.py) sweeps candidate angles and keeps the one
that separates rows most cleanly — measuring the angle geometrically is
unreliable, because adjacent rows sit closer together than a carton is tall, so
any local estimate is contaminated by boxes from the row above or below.

**4. Row clustering** — boxes are projected onto the axis normal to the row
direction, where a receding row collapses to a tight band, then split wherever
the gap exceeds 0.55 × median box height.

**5. Row pairing** — rows are matched between views by **normalised vertical
position**, not list index, so a view that misses a row does not shift every
later pairing.

**6. Counting** — each paired row contributes `front × side`; the pallet total
is their sum.

---

## Photo guidance

- Both photos from **adjacent faces, 90° apart** — not opposite faces
- Camera roughly **level with the middle of the stack**
- The **whole pallet in frame**, all rows visible
- Square-on is best, but a moderate angle is corrected automatically; a tilt
  above 12° is reported as a warning

---

## Accuracy and limits

Validated against the demo pallet in [demo_images/](demo_images/) — a 3×4×4
stack, ground truth **48**, reported **48**.

Understand what the method can and cannot do:

- **Assumes each row is a full rectangular grid.** A partially filled row is
  reported as if complete. Multiplication cannot represent "grid minus two."
- **Detection error is multiplied, not added.** One missed carton in a 4-wide
  view of a 3-deep pallet loses 3 from the total, not 1.
- **An interior gap is invisible.** A carton missing from the middle of a row,
  hidden behind intact faces on both views, cannot be detected in principle.
- **When one view detects nothing**, the response falls back to the other view's
  face count and adds an explicit warning. That number is a 2D face count, not a
  pallet total.

Suited to verifying **full pallets** at goods-in, where the question is "is this
48 or 60?" It is not an exact audit count of a partially picked pallet.

---

## Running

```bash
docker compose up --build -d pallet-counter
```

Or locally:

```bash
pip install -r requirements.txt
PORT=8004 python main.py
```

Requires `ROBOFLOW_API_KEY` in the project root `.env`.
