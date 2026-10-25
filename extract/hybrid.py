"""Hybrid extraction: OpenCV geometry + a local vision model that only points at things.

The local model is good at reading and locating printed text but poor at spatial
layout, so it is asked for three things it does well:
  1. each room label: text, printed size, bounding box
  2. the main-entrance arrow: bounding box
  3. furniture symbols: type, bounding box
Room rectangles, scale, doors and windows all come from the walls (geometry.py).
"""
import base64
import difflib
import json
import re
import sys
import urllib.request

import cv2
import numpy as np

from geometry import Plan, debug_image, find_arrows, thin_runs, wall_runs

OUTDOOR = {"balcony", "sitout", "utility", "terrace"}
STRONG = 0.5  # boundary coverage that a room rectangle may not cross

TYPE_WORDS = [
    ("master_bedroom", ["mbed", "mbedroom", "master", "masterbedroom"]),
    ("bedroom", ["bed", "bedroom", "bedrm", "guest", "kids"]),
    ("toilet", ["toilet", "bath", "wc", "washroom", "mtoilet"]),
    ("kitchen", ["kitchen"]),
    ("dining", ["dining"]),
    ("living", ["living", "family", "hall", "lounge"]),
    ("drawing", ["drawing"]),
    ("pooja", ["pooja", "puja", "prayer"]),
    ("balcony", ["balcony"]),
    ("sitout", ["sitout", "sit", "out", "verandah", "veranda", "deck"]),
    ("utility", ["utility", "wash", "service"]),
    ("closet", ["closet", "dress", "dressing", "wardrobe", "walkin"]),
    ("study", ["study", "office", "library"]),
    ("store", ["store", "storage"]),
    ("foyer", ["foyer", "lobby", "entrance", "entry"]),
    ("passage", ["passage", "corridor"]),
    ("terrace", ["terrace"]),
]
FURNITURE_TYPES = ["bed_double", "bed_single", "wardrobe", "sofa", "armchair", "coffee_table", "side_table",
                   "tv_unit", "dining_table", "study_table", "kitchen_counter", "stove", "sink", "fridge",
                   "wc", "washbasin", "shower", "washing_machine", "dishwasher", "plant"]
AGAINST_WALL = {"bed_double", "bed_single", "wardrobe", "sofa", "armchair", "tv_unit", "kitchen_counter",
                "stove", "sink", "fridge", "wc", "washbasin", "washing_machine", "dishwasher", "study_table"}


# ---------- local model ----------
def ollama(host, model, image_b64, prompt, schema, num_predict=3000):
    body = {
        "model": model, "stream": False, "format": schema, "think": False,
        "options": {"temperature": 0, "num_ctx": 16384, "num_predict": num_predict},
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
    }
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.loads(r.read())
    ev, secs = resp.get("eval_count", 0), max(resp.get("eval_duration", 1) / 1e9, 1e-6)
    print(f"[ollama] {model}: out={ev} tok at {ev / secs:.0f} tok/s", file=sys.stderr)
    return json.loads(resp["message"]["content"])


BOX = {"type": "array", "items": {"type": "number"}}
LABELS_SCHEMA = {
    "type": "object", "required": ["labels", "entrance"],
    "properties": {
        "labels": {"type": "array", "items": {"type": "object", "required": ["text", "dims", "bbox_2d"],
                   "properties": {"text": {"type": "string"}, "dims": {"type": "string"}, "bbox_2d": BOX}}},
        "entrance": {"type": "object", "required": ["bbox_2d"], "properties": {"bbox_2d": BOX}},
    },
}
LABELS_PROMPT = (
    "This is an apartment floor plan. Locate every printed room name label (for example BED ROOM-02, "
    "KITCHEN, POOJA, CLOSET, 4'6\" WIDE BALCONY). For each give the label text, the dimension text printed "
    "under it (empty if none), and bbox_2d around the label text. Also give bbox_2d of the main entrance "
    "arrow (a small solid triangle just outside the outer wall). Output JSON."
)
FURNITURE_SCHEMA = {
    "type": "object", "required": ["items"],
    "properties": {"items": {"type": "array", "items": {"type": "object", "required": ["type", "bbox_2d"],
                   "properties": {"type": {"type": "string", "enum": FURNITURE_TYPES}, "bbox_2d": BOX}}}},
}
FURNITURE_PROMPT = (
    "This is an apartment floor plan drawn from above. Locate every furniture and fixture symbol: beds, "
    "wardrobes, sofas, armchairs, coffee tables, side tables, TV units, dining tables (with chairs), kitchen "
    "counters, stoves, sinks, fridges, WCs, washbasins, showers, washing machines, dishwashers and plants. "
    "Give each one's type and a tight bbox_2d. Output JSON."
)
NAME_SCHEMA = {"type": "object", "required": ["text", "dims"],
               "properties": {"text": {"type": "string"}, "dims": {"type": "string"}}}
NAME_PROMPT = "This is one room cut out of a floor plan. Read the room name and the dimension text printed in it. Use empty strings if there is none."


def to_px(bbox, W, H):
    """Qwen-VL grounding boxes are normalised to 0-1000."""
    x1, y1, x2, y2 = (bbox + [0, 0, 0, 0])[:4]
    return x1 * W / 1000, y1 * H / 1000, x2 * W / 1000, y2 * H / 1000


def b64png(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode()


# ---------- text ----------
def room_type(text):
    words = re.findall(r"[a-z]+", text.lower().replace(".", ""))
    joined = "".join(words)
    for t, keys in TYPE_WORDS:
        if any(k == joined or k in words for k in keys):
            return t
    for w in words:  # tolerate OCR slips such as FOOJA
        for t, keys in TYPE_WORDS:
            if difflib.get_close_matches(w, [k for k in keys if len(k) > 3], n=1, cutoff=0.75):
                return t
    return None


def feet(part):
    """"12'0\"" / "12'" / "120" / "1010" -> feet as float."""
    part = part.strip()
    m = re.match(r"(\d+)\s*'\s*(\d+)?", part)
    if m and int(m.group(1)) <= 40:
        return int(m.group(1)) + int(m.group(2) or 0) / 12
    digits = re.sub(r"\D", "", part)
    if len(digits) in (2, 3):  # VLMs often drop the marks: 120 -> 12'0", 56 -> 5'6"
        return int(digits[:-1]) + int(digits[-1]) / 12
    if len(digits) == 4 and int(digits[2:]) < 12:
        return int(digits[:2]) + int(digits[2:]) / 12
    return None


def parse_dims(dims, text=""):
    """Returns (a, b) feet, (w, None) for '4'6" WIDE ...', or None."""
    parts = re.split(r"\s*[xX×]\s*", dims or "")
    if len(parts) == 2:
        a, b = feet(parts[0]), feet(parts[1])
        if a and b and 2 <= a <= 40 and 2 <= b <= 40:
            return a, b
    m = re.search(r"(\d+\s*'\s*\d*)\"?\s*WIDE", text.upper())
    if m and feet(m.group(1)):
        return feet(m.group(1)), None
    return None


def fmt_ft(v):
    f = int(v + 1e-6)
    return f"{f}'{round((v - f) * 12)}\""


# ---------- fitting ----------
def candidate_rects(p, seed, taken):
    """All cell rectangles containing seed with only inside, untaken cells and no strong wall inside."""
    si, sj = seed
    out = []
    for i0 in range(si, -1, -1):
        if not p.inside[i0, sj] or taken[i0, sj]: break
        if i0 < si and p.cov_v[i0 + 1, sj] >= STRONG: break
        for i1 in range(si, p.nx):
            if not p.inside[i1, sj] or taken[i1, sj]: break
            if i1 > si and p.cov_v[i1, sj] >= STRONG: break
            for j0 in range(sj, -1, -1):
                if not rect_ok(p, taken, i0, i1, j0, j0): break
                if j0 < sj and (p.cov_h[i0:i1 + 1, j0 + 1] >= STRONG).any(): break
                for j1 in range(sj, p.ny):
                    if not rect_ok(p, taken, i0, i1, j1, j1): break
                    if j1 > sj and (p.cov_h[i0:i1 + 1, j1] >= STRONG).any(): break
                    if not rect_ok(p, taken, i0, i1, j0, j1, walls=True): continue
                    out.append((i0, j0, i1, j1))
    return out


def rect_ok(p, taken, i0, i1, j0, j1, walls=False):
    if not p.inside[i0:i1 + 1, j0:j1 + 1].all() or taken[i0:i1 + 1, j0:j1 + 1].any():
        return False
    if walls:
        if (p.cov_v[i0 + 1:i1 + 1, j0:j1 + 1] >= STRONG).any(): return False
        if (p.cov_h[i0:i1 + 1, j0 + 1:j1 + 1] >= STRONG).any(): return False
    return True


def size_cost(w, h, dims, wall=0.5):
    a, b = dims
    if b is None:  # only a width is printed ("4'6\" WIDE BALCONY"); longer is better
        return abs(min(w, h) - (a + wall)) - 0.05 * max(w, h)
    return min(abs(w - a - wall) + abs(h - b - wall), abs(w - b - wall) + abs(h - a - wall))


def solve_scale(p, labels):
    """Pixels per foot that makes the most labelled rooms fit their printed size."""
    taken = np.zeros((p.nx, p.ny), bool)
    per = []
    for lb in labels:
        if lb["dims"] and lb["dims"][1] is not None:
            rects = candidate_rects(p, lb["cell"], taken)
            if rects:
                wh = np.array([(p.xs[i1 + 1] - p.xs[i0], p.ys[j1 + 1] - p.ys[j0]) for i0, j0, i1, j1 in rects], float)
                per.append((wh, lb["dims"]))
    if not per:
        raise RuntimeError("no labelled room with printed dimensions; cannot find the scale")
    best = None
    for s in np.arange(6, 80, 0.1):
        total = 0
        for wh, (a, b) in per:
            w, h = wh[:, 0] / s, wh[:, 1] / s
            c = np.minimum(np.abs(w - a - .5) + np.abs(h - b - .5), np.abs(w - b - .5) + np.abs(h - a - .5)).min()
            total += min(c, 3.0)
        if best is None or total < best[0]:
            best = (total, s)
    return best[1]


def build(bgr, found, furniture_found=None, namer=None):
    """found: output of the labels call. Returns (plan dict, debug image)."""
    p = Plan(bgr)
    H, W = p.H, p.W

    typed = []
    for lb in found["labels"]:
        t = room_type(lb["text"])
        if not t:
            continue
        x1, y1, x2, y2 = to_px(lb["bbox_2d"], W, H)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if not (p.xs[0] <= cx <= p.xs[-1] and p.ys[0] <= cy <= p.ys[-1]):
            print(f"[geometry] skipping '{lb['text']}': outside the detected walls (cut off?)", file=sys.stderr)
            continue
        typed.append((lb, t, cx, cy))
    p.claim_inside([(cx, cy) for _, _, cx, cy in typed])

    labels = []
    for lb, t, cx, cy in typed:
        cell = nearest_inside(p, cx, cy)
        if cell is None:
            continue
        labels.append({"text": lb["text"].strip(), "raw_dims": lb["dims"], "type": t, "cell": cell,
                       "dims": parse_dims(lb["dims"], lb["text"]), "c": (cx, cy)})

    s = solve_scale(p, labels)
    print(f"[geometry] grid {p.nx}x{p.ny}, scale {s:.1f} px/ft, {len(labels)} room labels", file=sys.stderr)

    # Place the best-fitting rooms first so they claim their cells. A room may not
    # swallow another room's label.
    taken = np.zeros((p.nx, p.ny), bool)
    pending = list(labels)
    rooms = []
    while pending:
        scored = []
        for lb in pending:
            blocked = taken.copy()
            for other in pending:
                if other is not lb and other["cell"] != lb["cell"]:
                    blocked[other["cell"]] = True
            rects = candidate_rects(p, lb["cell"], blocked)
            if not rects:
                continue
            def cost(r):
                i0, j0, i1, j1 = r
                w, h = (p.xs[i1 + 1] - p.xs[i0]) / s, (p.ys[j1 + 1] - p.ys[j0]) / s
                return size_cost(w, h, lb["dims"]) if lb["dims"] else -w * h * 0.01
            r = min(rects, key=cost)
            scored.append((cost(r) if lb["dims"] else 50, lb, r))
        if not scored:
            break
        scored.sort(key=lambda t: t[0])
        _, lb, r = scored[0]
        pending.remove(lb)
        i0, j0, i1, j1 = r
        taken[i0:i1 + 1, j0:j1 + 1] = True
        rooms.append({"label": lb, "cells": r})

    grow_undersized(p, rooms, labels, s)
    taken[:] = False
    for r in rooms:
        i0, j0, i1, j1 = r["cells"]
        taken[i0:i1 + 1, j0:j1 + 1] = True

    # Unclaimed floor inside the labelled area becomes passages, or asks the model for a name
    if rooms:
        bi0 = min(r["cells"][0] for r in rooms); bj0 = min(r["cells"][1] for r in rooms)
        bi1 = max(r["cells"][2] for r in rooms); bj1 = max(r["cells"][3] for r in rooms)
        free = p.inside & ~taken
        free[:bi0, :] = free[bi1 + 1:, :] = False
        free[:, :bj0] = free[:, bj1 + 1:] = False
        while free.any():
            best, area = None, 0
            for i in range(p.nx):
                for j in range(p.ny):
                    if free[i, j]:
                        for r in candidate_rects(p, (i, j), ~free):
                            a = (p.xs[r[2] + 1] - p.xs[r[0]]) * (p.ys[r[3] + 1] - p.ys[r[1]])
                            if a > area: best, area = r, a
            if best is None:
                break
            i0, j0, i1, j1 = best
            free[i0:i1 + 1, j0:j1 + 1] = False
            w_ft, h_ft = (p.xs[i1 + 1] - p.xs[i0]) / s, (p.ys[j1 + 1] - p.ys[j0]) / s
            # Slivers between double lines go to a neighbour; strips a person fits through stay passages
            if min(w_ft, h_ft) < 2.5 and absorb(p, rooms, best):
                continue
            if min(w_ft, h_ft) < 1.5:
                continue
            rooms.append({"label": None, "cells": best})

    # Pixel and feet rectangles. Edges on two lines a wall's thickness apart are
    # snapped to one, so neighbours share the wall and the door in it can be found.
    xmap = snap_lines(p, [e for r in rooms for e in (r["cells"][0], r["cells"][2] + 1)], s, vertical=True)
    ymap = snap_lines(p, [e for r in rooms for e in (r["cells"][1], r["cells"][3] + 1)], s, vertical=False)
    x_org, y_org = min(xmap[r["cells"][0]] for r in rooms), min(ymap[r["cells"][1]] for r in rooms)
    q = lambda v: round(v * 4) / 4
    out_rooms, used = [], {}
    for r in rooms:
        i0, j0, i1, j1 = r["cells"]
        px = (xmap[i0], ymap[j0], xmap[i1 + 1], ymap[j1 + 1])
        lb = r["label"]
        if lb is None:
            enclosed = all(c >= STRONG for c in edge_coverages(p, r["cells"]))
            text, dims = ("", "")
            if enclosed and namer:
                text, dims = namer(bgr[px[1]:px[3], px[0]:px[2]])
            t = room_type(text) if text else None
            lb = {"text": text, "raw_dims": dims, "type": t or ("other" if enclosed else "passage")}
        base = re.sub(r"[^a-z0-9]+", "_", (lb["text"] or lb["type"]).lower()).strip("_") or lb["type"]
        used[base] = used.get(base, 0) + 1
        rid = base if used[base] == 1 else f"{base}_{used[base]}"
        dims = parse_dims(lb.get("raw_dims", ""), lb["text"])
        out_rooms.append({
            "id": rid, "type": lb["type"], "label": lb["text"],
            "label_dims": f"{fmt_ft(dims[0])}x{fmt_ft(dims[1])}" if dims and dims[1] else "",
            # Round edges, not sizes, so rooms that share a wall share the exact coordinate
            "x": q((px[0] - x_org) / s), "y": q((px[1] - y_org) / s),
            "w": q((px[2] - x_org) / s) - q((px[0] - x_org) / s),
            "d": q((px[3] - y_org) / s) - q((px[1] - y_org) / s),
            "_px": px, "_guessed": r["label"] is None,
        })

    doors, windows = openings(p, out_rooms, s, found.get("entrance"))
    # An enclosed space with no printed label and no door is a shaft or duct, not part
    # of the flat, even if the model guessed a name for it from the crop
    connected = {d["from"] for d in doors} | {d["to"] for d in doors}
    shafts = {r["id"] for r in out_rooms if r["_guessed"] and r["type"] != "passage" and r["id"] not in connected}
    for sid in shafts:
        print(f"[geometry] dropping '{sid}': enclosed, no printed label, no door (shaft?)", file=sys.stderr)
    out_rooms = [r for r in out_rooms if r["id"] not in shafts]
    windows = [w for w in windows if w["room"] not in shafts]
    furniture = place_furniture(furniture_found or [], out_rooms, s, x_org, y_org, W, H)
    dbg = debug_image(p, [{"id": r["id"], "px": r["_px"]} for r in out_rooms])
    for r in out_rooms:
        r.pop("_px"); r.pop("_guessed")
    plan = {
        "schema_version": "1.0", "units": "ft",
        "meta": {"title": "", "bhk": "", "facing": "unknown", "total_area_sqft": 0, "carpet_area_sqft": 0,
                 "balcony_area_sqft": 0, "north_deg": 0,
                 "notes": f"Hybrid extraction: OpenCV geometry at {s:.1f} px/ft, local-model labels."},
        "defaults": {"wall_height_ft": 10, "ext_wall_thickness_ft": 0.75, "int_wall_thickness_ft": 0.5, "door_height_ft": 7},
        "rooms": out_rooms, "doors": doors, "windows": windows, "furniture": furniture,
    }
    return plan, dbg


def snap_lines(p, used, s, vertical, within=0.75):
    """Map grid-line index -> pixel position, merging used lines closer than `within` ft
    onto the one that carries the most wall."""
    pos = p.xs if vertical else p.ys
    strength = (p.cov_v.sum(axis=1) if vertical else p.cov_h.sum(axis=0))
    idx = sorted(set(used))
    out, group = {}, []
    for i in idx + [None]:
        if group and (i is None or pos[i] - pos[group[-1]] > within * s):
            rep = max(group, key=lambda g: strength[g])
            for g in group:
                out[g] = pos[rep]
            group = []
        if i is not None:
            group.append(i)
    return out


def rect_cost(p, r, dims, s):
    i0, j0, i1, j1 = r
    return size_cost((p.xs[i1 + 1] - p.xs[i0]) / s, (p.ys[j1 + 1] - p.ys[j0]) / s, dims)


def grow_undersized(p, rooms, labels, s, min_gain=1.0, line_penalty=0.1):
    """Built-in counters and wardrobes are often drawn like walls and cut a room
    short. A room well under its printed size may grow into unclaimed floor across
    such lines, when that brings it much closer to the printed size."""
    claimed = np.zeros((p.nx, p.ny), int) - 1
    for k, r in enumerate(rooms):
        i0, j0, i1, j1 = r["cells"]
        claimed[i0:i1 + 1, j0:j1 + 1] = k
    for lb in labels:
        if claimed[lb["cell"]] < 0:
            claimed[lb["cell"]] = -2  # an unplaced label still blocks growth
    for k, r in enumerate(rooms):
        dims = r["label"]["dims"]
        if not dims or dims[1] is None:
            continue
        a0, b0, a1, b1 = r["cells"]
        now = rect_cost(p, r["cells"], dims, s)
        if now <= 1.5:
            continue
        best, best_cost = None, now - min_gain
        for i0 in range(a0, -1, -1):
            for i1 in range(a1, p.nx):
                for j0 in range(b0, -1, -1):
                    for j1 in range(b1, p.ny):
                        area = claimed[i0:i1 + 1, j0:j1 + 1]
                        if ((area != k) & (area != -1)).any() or p.outside[i0:i1 + 1, j0:j1 + 1].any():
                            continue
                        crossed = sum(
                            p.cov_v[i, j] * (p.ys[j + 1] - p.ys[j]) for i in range(i0 + 1, i1 + 1) for j in range(j0, j1 + 1)
                            if p.cov_v[i, j] >= STRONG and not (a0 < i <= a1 and b0 <= j <= b1)
                        ) + sum(
                            p.cov_h[i, j] * (p.xs[i + 1] - p.xs[i]) for i in range(i0, i1 + 1) for j in range(j0 + 1, j1 + 1)
                            if p.cov_h[i, j] >= STRONG and not (a0 <= i <= a1 and b0 < j <= b1)
                        )
                        c = rect_cost(p, (i0, j0, i1, j1), dims, s) + line_penalty * crossed / s
                        if c < best_cost:
                            best, best_cost = (i0, j0, i1, j1), c
        if best:
            print(f"[geometry] {r['label']['text']}: grew across built-in lines, size error {now:.1f} -> "
                  f"{rect_cost(p, best, dims, s):.1f} ft", file=sys.stderr)
            r["cells"] = best
            i0, j0, i1, j1 = best
            claimed[i0:i1 + 1, j0:j1 + 1] = k


def absorb(p, rooms, cells):
    """Extend a neighbouring room over a sliver that shares one of its full sides."""
    i0, j0, i1, j1 = cells
    for r in rooms:
        a0, b0, a1, b1 = r["cells"]
        if (a0, a1) == (i0, i1) and b1 + 1 == j0 and (p.cov_h[i0:i1 + 1, j0] < STRONG).all():
            r["cells"] = (a0, b0, a1, j1); return True
        if (a0, a1) == (i0, i1) and j1 + 1 == b0 and (p.cov_h[i0:i1 + 1, b0] < STRONG).all():
            r["cells"] = (a0, j0, a1, b1); return True
        if (b0, b1) == (j0, j1) and a1 + 1 == i0 and (p.cov_v[i0, j0:j1 + 1] < STRONG).all():
            r["cells"] = (a0, b0, i1, b1); return True
        if (b0, b1) == (j0, j1) and i1 + 1 == a0 and (p.cov_v[a0, j0:j1 + 1] < STRONG).all():
            r["cells"] = (i0, b0, a1, b1); return True


def nearest_inside(p, x, y):
    c = p.cell_at(x, y)
    if c and p.inside[c]:
        return c
    best, bd = None, 1e18
    for i in range(p.nx):
        for j in range(p.ny):
            if p.inside[i, j]:
                x0, y0, x1, y1 = p.cell_box(i, j)
                d = (max(x0 - x, 0, x - x1)) ** 2 + (max(y0 - y, 0, y - y1)) ** 2
                if d < bd: best, bd = (i, j), d
    return best


def edge_coverages(p, cells):
    i0, j0, i1, j1 = cells
    return [wall_runs(p, 'h', p.ys[j0], p.xs[i0], p.xs[i1 + 1])[0], wall_runs(p, 'h', p.ys[j1 + 1], p.xs[i0], p.xs[i1 + 1])[0],
            wall_runs(p, 'v', p.xs[i0], p.ys[j0], p.ys[j1 + 1])[0], wall_runs(p, 'v', p.xs[i1 + 1], p.ys[j0], p.ys[j1 + 1])[0]]


def sides(r):
    x0, y0, x1, y1 = r["_px"]
    return {"top": ('h', y0, x0, x1), "bottom": ('h', y1, x0, x1), "left": ('v', x0, y0, y1), "right": ('v', x1, y0, y1)}


def openings(p, rooms, s, entrance):
    doors, windows, exterior = [], [], []
    min_gap = 1.75 * s
    for a_i, a in enumerate(rooms):
        for side, (o, c, st, en) in sides(a).items():
            shared = []
            for b in rooms[a_i + 1:] + rooms[:a_i]:
                for o2, c2, st2, en2 in sides(b).values():
                    if o2 == o and c2 == c:
                        lo, hi = max(st, st2), min(en, en2)
                        if hi - lo > 0.5 * s:
                            shared.append((lo, hi, b))
            for lo, hi, b in shared:
                if rooms.index(b) < a_i:
                    continue  # each pair once
                cov, gaps = wall_runs(p, o, c, lo, hi)
                gaps = [g for g in gaps if g[1] - g[0] >= min_gap]
                if not gaps:
                    continue
                L = hi - lo
                if cov < 0.05 or max(g[1] - g[0] for g in gaps) > 0.85 * (L - 2 * p.band):
                    doors.append(door(a, b, "opening", 0, 0.5))
                    continue
                for g0, g1 in gaps:
                    width = (g1 - g0) / s
                    outdoor = a["type"] in OUTDOOR or b["type"] in OUTDOOR
                    kind = "hinged" if width <= 4.5 else ("french" if outdoor else "opening")
                    doors.append(door(a, b, kind, min(round(width * 4) / 4, 8), round(((g0 + g1) / 2 - lo) / L, 2)))
            # Exterior: whatever part of this side no other room shares
            if a["type"] in OUTDOOR:
                continue
            for g0, g1, hole in thin_runs(p, o, c, st, en, 1.25 * s):
                mid = (g0 + g1) / 2
                if g1 - g0 < 1.25 * s or any(lo <= mid <= hi for lo, hi, _ in shared):
                    continue
                width = (g1 - g0) / s
                exterior.append({"room": a, "side": side, "o": o, "c": c, "g": (g0, g1), "len": en - st, "st": st,
                                 "door_like": hole > 0.6 and 2.5 <= width <= 4.75})

    main = pick_main_door(p, s, exterior, entrance)
    for e in exterior:
        width = (e["g"][1] - e["g"][0]) / s
        pos = round(((e["g"][0] + e["g"][1]) / 2 - e["st"]) / e["len"], 2)
        if e is main:
            doors.append({"id": "main", "from": e["room"]["id"], "to": "outside", "kind": "hinged",
                          "width_ft": min(round(width * 4) / 4, 4), "pos": pos, "side": e["side"], "main": True})
        else:
            kind = "ventilator" if e["room"]["type"] == "toilet" and width <= 3 else "window"
            windows.append({"id": f"w_{e['room']['id']}_{len(windows)}", "room": e["room"]["id"], "side": e["side"],
                            "kind": kind, "width_ft": min(round(width * 4) / 4, 8), "pos": pos})
    return doors, windows


MAIN_DOOR_ROOMS = ["foyer", "drawing", "living", "dining", "passage"]


def pick_main_door(p, s, exterior, entrance):
    """Entrance arrow found by OpenCV, else the model's arrow if it sits by a real door
    gap, else the door-shaped gap into the room a front door usually opens into."""
    if not exterior:
        return None

    def at(e):
        m = (e["g"][0] + e["g"][1]) / 2
        return (m, e["c"]) if e["o"] == 'h' else (e["c"], m)

    def nearest(pt):
        return min(exterior, key=lambda e: (at(e)[0] - pt[0]) ** 2 + (at(e)[1] - pt[1]) ** 2)

    def ft(e, pt):
        return ((at(e)[0] - pt[0]) ** 2 + (at(e)[1] - pt[1]) ** 2) ** 0.5 / s

    arrows = find_arrows(p, s)
    if arrows:
        best = min(((nearest(a), a) for a in arrows), key=lambda t: ft(*t))
        if ft(*best) < 6:
            print(f"[geometry] main door: entrance arrow found at {tuple(round(v) for v in best[1])} px", file=sys.stderr)
            return best[0]
    if entrance and len(entrance.get("bbox_2d", [])) == 4:
        x1, y1, x2, y2 = to_px(entrance["bbox_2d"], p.W, p.H)
        pt = ((x1 + x2) / 2, (y1 + y2) / 2)
        e = nearest(pt)
        if e["door_like"] and ft(e, pt) < 6:
            print(f"[geometry] main door: model's entrance arrow, by the {e['room']['id']} {e['side']} wall", file=sys.stderr)
            return e
    doorish = [e for e in exterior if e["door_like"]]
    rank = lambda e: MAIN_DOOR_ROOMS.index(e["room"]["type"]) if e["room"]["type"] in MAIN_DOOR_ROOMS else 99
    if doorish and rank(min(doorish, key=rank)) < 99:
        e = min(doorish, key=rank)
        print(f"[geometry] main door: no arrow; door-shaped gap into {e['room']['id']}", file=sys.stderr)
        return e
    print("[geometry] main door: not found", file=sys.stderr)
    return None


def door(a, b, kind, width, pos):
    return {"id": f"{a['id']}_{b['id']}", "from": a["id"], "to": b["id"], "kind": kind,
            "width_ft": width, "pos": pos, "side": "auto", "main": False}


def place_furniture(items, rooms, s, x_org, y_org, W, H):
    out = []
    for it in items:
        x1, y1, x2, y2 = to_px(it["bbox_2d"], W, H)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        room = next((r for r in rooms if r["_px"][0] <= cx <= r["_px"][2] and r["_px"][1] <= cy <= r["_px"][3]), None)
        if room is None or x2 - x1 < 4 or y2 - y1 < 4:
            continue
        rx0, ry0, rx1, ry1 = room["_px"]
        rot = 0
        if it["type"] in AGAINST_WALL:  # back against the nearest wall, facing into the room
            d = {180: ry1 - y2, 0: y1 - ry0, 90: x1 - rx0, 270: rx1 - x2}
            rot = min(d, key=d.get)
        out.append({"type": it["type"], "room": room["id"],
                    "cx": round((cx - x_org) / s, 2), "cy": round((cy - y_org) / s, 2),
                    "w": round((x2 - x1) / s, 2), "d": round((y2 - y1) / s, 2), "rot": rot})
    return out


def extract(bgr, model, host, furniture=True, name_rooms=True):
    """Run the local-model calls and assemble the plan."""
    img = b64png(bgr)
    found = ollama(host, model, img, LABELS_PROMPT, LABELS_SCHEMA)
    items = ollama(host, model, img, FURNITURE_PROMPT, FURNITURE_SCHEMA, 4000)["items"] if furniture else []

    def namer(crop):
        if crop.size == 0:
            return "", ""
        pad = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        r = ollama(host, model, b64png(pad), NAME_PROMPT, NAME_SCHEMA, 200)
        return r["text"], r["dims"]

    return build(bgr, found, items, namer if name_rooms else None)
