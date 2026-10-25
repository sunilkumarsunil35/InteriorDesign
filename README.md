# InteriorDesign

Turn a 2D floor-plan image (a real-estate brochure, a scan, a screenshot) into a structured **plan.json**, then into a walkable 3D model.

The vision model is used **once per image**, only to read the plan. Everything after that runs locally without AI: validation, geometry, the 3D model, and the walkthrough.

```
floor-plan image ──► extract/extract_plan.py ──► plan.json ──► builder/plan2scene.js ──► scene.json ──► three.js / .glb
     (crop)            Claude or local VLM        (feet)        (no deps, pure JS)        (metres)      any 3D app
                           │
                           └─ extract/validate.py checks it; on errors, one text-only correction pass
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

# Claude (needs ANTHROPIC_API_KEY)
python extract/extract_plan.py brochure.jpg --crop 295,650,670,1085 -o out/plan.json

# Fully local with Ollama and a vision model
ollama pull qwen2.5vl:7b
python extract/extract_plan.py brochure.jpg --crop 295,650,670,1085 --provider ollama -o out/plan.json
```

Then view it: http://localhost:8000/?plan=out/plan.json, or click **Load plan.json**.

| Flag | Default | Notes |
|---|---|---|
| `--crop x0,y0,x1,y1` | none | Isolate one plan on a multi-plan brochure page. The biggest token saving. |
| `--max-side` | 1100 | Long side in px sent to the model. Small crops are upscaled so labels stay legible. |
| `--provider` | claude | `claude` or `ollama` |
| `--model` | claude-opus-5-5 / qwen2.5vl:7b | Any Claude vision model id, e.g. `claude-haiku-5-5` for lower cost. |
| `--effort` | high | Claude reasoning effort: low, medium, high, xhigh, max |
| `--retries` | 1 | Correction passes when validation finds errors |

The Claude call uses server-side refusal fallbacks (`fallbacks: "default"`). If the main model declines a request, the API retries it on a fallback model inside the same call.

### Token budget per image

| Part | Tokens |
|---|---|
| Image (cropped to ~950×1100) | ~1,400 |
| System prompt + schema | ~2,000 (cached on retry) |
| Output plan.json | ~3,000–5,000 |
| Correction pass, if needed | text only; the image and prompt are read from cache |

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

It checks overlapping rooms, doors between rooms that share no wall, rooms unreachable from the main door, rectangles that don't match their printed sizes, and the rebuilt carpet area against the brochure's figure. The example comes within 5.2% (1351 vs 1425 sft).

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
extract/prompt.md              instructions for the vision model
extract/extract_plan.py        image -> plan.json (Claude or Ollama)
extract/validate.py            consistency checks, no AI
builder/plan2scene.js          plan.json -> scene (metres), no dependencies
builder/scene2three.js         scene -> three.js meshes
builder/cli.mjs                plan.json -> scene.json from the command line
viewer/app.js + index.html     dollhouse and walkthrough viewer
examples/2295-west/            brochure input, crop sent to the model, plan.json, scene.json
```
