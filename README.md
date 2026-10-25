# InteriorDesign

Turn a 2D floor-plan image (a real-estate brochure, a scan, a screenshot) into a structured **plan.json**, then into a walkable 3D model.

By default the whole extraction runs on your own GPU: OpenCV measures the walls, and a small local vision model reads and locates the printed labels. Claude is called only when the result fails validation. Everything after extraction runs without AI: validation, geometry, the 3D model, and the walkthrough.

```
floor-plan image ─┬─► extract/geometry.py (OpenCV) ── walls, grid, scale ──┐
     (crop)       │                                                         ├─► plan.json ─► validate ─► 3D
                  └─► local VLM via Ollama ── label text + positions,  ─────┘                    │
                       entrance arrow, furniture boxes                         errors only ──► Claude corrects the draft
```

## Try the example

```bash
python -m http.server 8000
```

Open http://localhost:8000. It loads [examples/2295-west/plan.json](examples/2295-west/plan.json), which was extracted from [the brochure page](examples/2295-west/brochure.jpg).

- **Dollhouse**: drag to orbit, scroll to zoom, tick "Cut walls" to see over them.
- **Walk in**: first person at 1.6 m eye height, walking at 1.4 m/s with gentle acceleration and head bob, colliding with walls and furniture. Desktop: WASD + mouse; phone: left thumb walks, right thumb looks.
- Double-click any floor, or click the minimap, to walk there.
- **Source image** tab: compare the 3D model with the input.
- **Load plan.json**: open any other plan; drag-and-drop works too.
- **Export .glb / scene.json**: hand the model to Blender, Unity, Unreal, etc.

## Extract a plan from your own image

```bash
pip install -r requirements.txt
ollama pull qwen3-vl:8b-instruct     # the non-thinking build; the default qwen3-vl:8b thinks for minutes

# Hybrid (default): OpenCV + local model, Claude only if validation fails
python extract/extract_plan.py brochure.jpg --crop 295,650,670,1085 -o out/plan.json
```

Then view it: http://localhost:8000/?plan=out/plan.json, or click **Load plan.json**. Next to the plan, `out/plan.debug.png` shows the detected walls, grid cells and room rectangles.

### How the hybrid works

Small local vision models read text well but cannot lay out rooms: asked for the whole plan, qwen3-vl 8B placed rooms 17 ft off on average. So the work is split:

1. **Walls (OpenCV, [geometry.py](extract/geometry.py))**: each brochure draws walls in its own grey, so the wall tone is learned per image from the long thick strokes. Strokes that are thinner or a different shade (furniture outlines) and solid areas thicker than a wall (TV units) are discarded. Wall centrelines make a grid of cells. The outside is flood-filled in from the image border, with window glazing and balcony rails counting as enclosure, and any space holding a printed room label is inside regardless (a plan cut off at the image edge, an open sit-out).
2. **Labels (local model, [hybrid.py](extract/hybrid.py))**: short calls ask the model to *point*, which Qwen-VL models are trained for. Where is each room label and what does it say? Where is the entrance arrow? Where is each furniture symbol? Enclosed rooms without a label are cropped and asked about one at a time. A name guessed this way is tentative: an enclosed space with no printed label and no door is dropped as a shaft.
3. **Scale**: the px-per-foot value that makes the enclosed rooms best match their printed sizes.
4. **Rooms**: each label claims the rectangle of grid cells around it that best fits its printed size, without crossing a solid wall or swallowing another label. A room left well under its printed size may then grow across lines that are not another room's wall: kitchen counters and built-in wardrobes drawn in wall grey. Leftover floor becomes passages, slivers between double lines join a neighbour, and edges a wall's thickness apart are snapped to one line so neighbours share their wall.
5. **Doors and windows**: gaps along shared walls are doors (open-plan when there's no wall at all). On outer walls, any stretch under half the wall's thickness is an opening: a true hole is door-shaped, a glazing line through it makes it a window. The main door is found from the entrance arrow detected by OpenCV; else the model's arrow if it sits by a door-shaped gap; else the door-shaped gap into a foyer, drawing, living or dining room. Furniture is turned to face away from its nearest wall.

On the example brochure (RX 9070 XT, 16 GB): **31 s, 16/16 rooms, median position error 0.1 ft against the hand-checked plan (kitchen 0.2 ft after growing across its counters), 7 of 9 windows, 0 validation errors, no Claude call.** Compare [hybrid.plan.json](examples/2295-west/hybrid.plan.json) with the reference [plan.json](examples/2295-west/plan.json) and see [hybrid.debug.png](examples/2295-west/hybrid.debug.png).

The East-facing plan on the same page, which was never used for tuning and is cut off at the image's left edge, also comes out with 0 errors: the cut-off balconies are skipped, and the remaining rooms, doors and main door are found.

Known limits: toilet ventilators drawn very small are missed; the area table outside the crop isn't read; on plans cut off at the edge, strips along the cut become thin passages.

### Other modes

```bash
# Whole plan from Claude (needs ANTHROPIC_API_KEY)
python extract/extract_plan.py brochure.jpg --crop ... --mode vlm --provider claude -o out/plan.json
# Whole plan from the local model (fast, but room positions are unreliable)
python extract/extract_plan.py brochure.jpg --crop ... --mode vlm --provider ollama -o out/plan.json
# Score any result against a reference
python extract/compare.py out/plan.json examples/2295-west/plan.json
```

| Flag | Default | Notes |
|---|---|---|
| `--crop x0,y0,x1,y1` | none | Isolate one plan on a multi-plan brochure page. |
| `--max-side` | 1100 | Long side in px. Small crops are upscaled so labels stay legible. |
| `--mode` | hybrid | `hybrid` or `vlm` |
| `--local-model` | qwen3-vl:8b-instruct | Ollama vision model for hybrid mode |
| `--fallback` | claude | `claude` or `none`; used only when the hybrid plan fails validation and credentials exist |
| `--provider` | claude | Model for `--mode vlm`: `claude` or `ollama` |
| `--model` | claude-opus-5-5 | Any Claude vision model id, e.g. `claude-haiku-5-5` for lower cost |
| `--effort` | high | Claude reasoning effort: low, medium, high, xhigh, max |
| `--no-furniture` | off | Skip the furniture call |

Claude calls use server-side refusal fallbacks (`fallbacks: "default"`). If the main model declines a request, the API retries it on a fallback model inside the same call.

### Cost per image

| Path | Claude tokens |
|---|---|
| Hybrid, passes validation | **0** (about 45 s on a 16 GB GPU) |
| Hybrid, Claude corrects the draft | ~1,400 image + ~2,000 prompt + ~4,000 draft in, ~4,000 out |
| `--mode vlm --provider claude` | ~1,400 image + ~2,000 prompt in, ~3,000–5,000 out |

Viewing, walking, editing furniture and exporting cost nothing.

## The plan.json format

Specified in [schema/floorplan.schema.json](schema/floorplan.schema.json). The same file is the structured-output schema for the model, so the model cannot return anything else.

- **Units are feet.** The origin is the plan's top-left corner, with x right and y down, exactly as drawn.
- **Rooms** are rectangles to wall centerlines; neighbours share edges exactly. Each keeps its printed `label` and `label_dims` (`12'0"x13'0"`).
- **Doors** connect two rooms (or a room and `outside`): `hinged`, `sliding`, `french`, or `opening` (`width_ft: 0` = fully open plan). One door is `main`.
- **Windows** sit on a room edge: `window`, `ventilator`, `bay`.
- **Furniture** uses a fixed vocabulary (bed, sofa, wardrobe, wc, stove…) with centre, footprint and facing.

Walls are not stored. They are derived from room edges, so a plan can never have walls that disagree with its rooms.

### Validation (no AI)

```bash
python extract/validate.py examples/2295-west/plan.json
```

It checks overlapping rooms, neighbours whose edges miss each other by a few inches (which would double the wall in 3D), doors between rooms that share no wall, rooms unreachable from the main door, rectangles that don't match their printed sizes, and the rebuilt carpet area against the brochure's figure. The example comes within 5.2% (1351 vs 1425 sft).

## Using the output in another app

| You need | Use |
|---|---|
| Semantic plan (rooms, doors, furniture) | `plan.json` |
| Geometry in metres: wall boxes with heights, openings, floors, colliders, spawn point | `node builder/cli.mjs plan.json > scene.json` |
| A mesh for Blender, Unity, Unreal, Godot, a web page | **Export .glb** in the viewer |
| three.js in your own app | `import { planToScene } from './builder/plan2scene.js'` and `import { buildThree } from './builder/scene2three.js'`, then `buildThree(THREE, planToScene(plan)).root` |

`scene.json` is plain data (`walls[]` with `x0,z0,x1,z1,y0,y1,t`, `rooms[]`, `openings[]`, `furniture[]`, `colliders[]`, `spawn`), so a Unity or Godot importer is a short loop that creates one box per wall.

## Layout

```
schema/floorplan.schema.json   the interchange format
extract/prompt.md              instructions for a model that reads the whole plan
extract/extract_plan.py        image -> plan.json (hybrid, or one model via Claude or Ollama)
extract/geometry.py            walls, grid cells, spaces and wall gaps from pixels (OpenCV, no AI)
extract/hybrid.py              local-model labels + geometry -> rooms, doors, windows, furniture
extract/validate.py            consistency checks, no AI
extract/compare.py             score a plan against a reference
builder/plan2scene.js          plan.json -> scene (metres), no dependencies
builder/scene2three.js         scene -> three.js meshes
builder/cli.mjs                plan.json -> scene.json from the command line
viewer/app.js + index.html     dollhouse and walkthrough viewer
examples/2295-west/            brochure input, crop, hand-checked plan.json, hybrid.plan.json + debug image, scene.json
```
