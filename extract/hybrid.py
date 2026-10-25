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

from geometry import Plan, debug_image, wall_runs

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

    labels = []
    for lb in found["labels"]:
        t = room_type(lb["text"])
        if not t:
            continue
        x1, y1, x2, y2 = to_px(lb["bbox_2d"], W, H)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
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
            if min(w_ft, h_ft) < 2.5:  # sliver between double lines: give it to a neighbour
                absorb(p, rooms, best)
                continue
            rooms.append({"label": None, "cells": best})

    # Pixel and feet rectangles
    x_org, y_org = min(p.xs[r["cells"][0]] for r in rooms), min(p.ys[r["cells"][1]] for r in rooms)
    q = lambda v: round(v * 4) / 4
    out_rooms, used = [], {}
    for r in rooms:
        i0, j0, i1, j1 = r["cells"]
        px = (p.xs[i0], p.ys[j0], p.xs[i1 + 1], p.ys[j1 + 1])
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
            "_px": px,
        })

    doors, windows = openings(p, out_rooms, s, found.get("entrance"))
    furniture = place_furniture(furniture_found or [], out_rooms, s, x_org, y_org, W, H)
    dbg = debug_image(p, [{"id": r["id"], "px": r["_px"]} for r in out_rooms])
    for r in out_rooms:
        r.pop("_px")
    plan = {
        "schema_version": "1.0", "units": "ft",
        "meta": {"title": "", "bhk": "", "facing": "unknown", "total_area_sqft": 0, "carpet_area_sqft": 0,
                 "balcony_area_sqft": 0, "north_deg": 0,
                 "notes": f"Hybrid extraction: OpenCV geometry at {s:.1f} px/ft, local-model labels."},
        "defaults": {"wall_height_ft": 10, "ext_wall_thickness_ft": 0.75, "int_wall_thickness_ft": 0.5, "door_height_ft": 7},
        "rooms": out_rooms, "doors": doors, "windows": windows, "furniture": furniture,
    }
    return plan, dbg


def absorb(p, rooms, cells):
    """Extend a neighbouring room over a sliver that shares one of its full sides."""
    i0, j0, i1, j1 = cells
    for r in rooms:
        a0, b0, a1, b1 = r["cells"]
        if (a0, a1) == (i0, i1) and b1 + 1 == j0 and (p.cov_h[i0:i1 + 1, j0] < STRONG).all():
            r["cells"] = (a0, b0, a1, j1); return
        if (a0, a1) == (i0, i1) and j1 + 1 == b0 and (p.cov_h[i0:i1 + 1, b0] < STRONG).all():
            r["cells"] = (a0, j0, a1, b1); return
        if (b0, b1) == (j0, j1) and a1 + 1 == i0 and (p.cov_v[i0, j0:j1 + 1] < STRONG).all():
            r["cells"] = (a0, b0, i1, b1); return
        if (b0, b1) == (j0, j1) and i1 + 1 == a0 and (p.cov_v[a0, j0:j1 + 1] < STRONG).all():
            r["cells"] = (i0, b0, a1, b1); return


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
            _, gaps = wall_runs(p, o, c, st, en)
            for g0, g1 in gaps:
                mid = (g0 + g1) / 2
                if g1 - g0 < 1.25 * s or any(lo <= mid <= hi for lo, hi, _ in shared):
                    continue
                exterior.append({"room": a, "side": side, "o": o, "c": c, "g": (g0, g1), "len": en - st, "st": st})

    main = None
    if entrance and exterior:
        ex1, ey1, ex2, ey2 = to_px(entrance["bbox_2d"], p.W, p.H)
        ex, ey = (ex1 + ex2) / 2, (ey1 + ey2) / 2
        def dist(e):
            m = (e["g"][0] + e["g"][1]) / 2
            x, y = (m, e["c"]) if e["o"] == 'h' else (e["c"], m)
            return (x - ex) ** 2 + (y - ey) ** 2
        main = min(exterior, key=dist)
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
