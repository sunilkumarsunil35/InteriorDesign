"""Checks a plan.json for geometric and topological consistency.

Runs locally with no AI. The extractor uses it to decide whether a plan is
good enough or needs a correction pass.

    python extract/validate.py examples/2295-west/plan.json
"""
import json
import re
import sys
from collections import deque

OUTDOOR = {"balcony", "sitout", "utility", "terrace"}
TOL = 0.3  # ft; how far two edges can be apart and still count as shared


def parse_dims(text):
    """"12'0\"x13'0\"" -> (12.0, 13.0); None if unparseable."""
    nums = re.findall(r"(\d+)\s*'\s*(\d+)?", text or "")
    if len(nums) != 2:
        return None
    return tuple(int(f) + int(i or 0) / 12 for f, i in nums)


def edges(r):
    x0, y0, x1, y1 = r["x"], r["y"], r["x"] + r["w"], r["y"] + r["d"]
    return {
        "top": ("h", y0, x0, x1),
        "bottom": ("h", y1, x0, x1),
        "left": ("v", x0, y0, y1),
        "right": ("v", x1, y0, y1),
    }


def shared_edge(a, b):
    """Returns (side_of_a, overlap_start, overlap_end) or None."""
    for side, (o1, c1, s1, e1) in edges(a).items():
        for o2, c2, s2, e2 in edges(b).values():
            if o1 == o2 and abs(c1 - c2) <= TOL:
                s, e = max(s1, s2), min(e1, e2)
                if e - s > 0.5:
                    return side, s, e
    return None


def validate(plan):
    errors, warnings = [], []
    rooms = {r["id"]: r for r in plan["rooms"]}
    if len(rooms) != len(plan["rooms"]):
        errors.append("duplicate room ids")

    for r in plan["rooms"]:
        if r["w"] <= 0 or r["d"] <= 0:
            errors.append(f"room {r['id']}: non-positive size")

    # Overlaps
    rl = plan["rooms"]
    for i, a in enumerate(rl):
        for b in rl[i + 1:]:
            ox = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
            oy = min(a["y"] + a["d"], b["y"] + b["d"]) - max(a["y"], b["y"])
            if ox > TOL and oy > TOL and ox * oy > 1.0:
                errors.append(f"rooms {a['id']} and {b['id']} overlap by {ox * oy:.1f} sqft")

    # Printed dimensions vs rectangle (centerline rect is a bit larger than clear size)
    for r in rl:
        dims = parse_dims(r["label_dims"])
        if not dims:
            continue
        fits = lambda pw, pd: -0.6 <= r["w"] - pw <= 2.0 and -0.6 <= r["d"] - pd <= 2.0
        if not (fits(*dims) or fits(dims[1], dims[0])):
            warnings.append(
                f"room {r['id']}: rect {r['w']}x{r['d']} ft does not match printed {r['label_dims']}"
            )

    # Doors
    graph = {rid: set() for rid in rooms}
    mains = []
    for d in plan["doors"]:
        a = rooms.get(d["from"])
        if not a:
            errors.append(f"door {d['id']}: unknown room '{d['from']}'")
            continue
        if d["to"] == "outside":
            if d["side"] == "auto":
                errors.append(f"door {d['id']}: exterior door needs an explicit side")
            if d["main"]:
                mains.append(d["from"])
            continue
        b = rooms.get(d["to"])
        if not b:
            errors.append(f"door {d['id']}: unknown room '{d['to']}'")
            continue
        se = shared_edge(a, b)
        if not se:
            errors.append(f"door {d['id']}: rooms {a['id']} and {b['id']} share no wall")
            continue
        if d["width_ft"] > se[2] - se[1] + 0.01:
            warnings.append(f"door {d['id']}: {d['width_ft']} ft wider than shared wall ({se[2] - se[1]:.2f} ft)")
        graph[a["id"]].add(b["id"])
        graph[b["id"]].add(a["id"])

    for w in plan["windows"]:
        if w["room"] not in rooms:
            errors.append(f"window {w['id']}: unknown room '{w['room']}'")
    for f in plan["furniture"]:
        if f["room"] not in rooms:
            warnings.append(f"furniture {f['type']}: unknown room '{f['room']}'")

    # Reachability from the main door
    if not mains:
        errors.append("no main entrance (door with to='outside' and main=true)")
    else:
        seen, q = {mains[0]}, deque([mains[0]])
        while q:
            for n in graph[q.popleft()]:
                if n not in seen:
                    seen.add(n)
                    q.append(n)
        cut_off = sorted(set(rooms) - seen)
        if cut_off:
            errors.append(f"unreachable from main door: {', '.join(cut_off)}")

    # Area check: inset each indoor room by half a wall on every side
    t = plan["defaults"]["int_wall_thickness_ft"]
    carpet = sum(
        max(r["w"] - t, 0) * max(r["d"] - t, 0) for r in rl if r["type"] not in OUTDOOR
    )
    stated = plan["meta"]["carpet_area_sqft"]
    report = {"rooms": len(rl), "doors": len(plan["doors"]), "estimated_carpet_sqft": round(carpet)}
    if stated:
        diff = (carpet - stated) / stated
        report["stated_carpet_sqft"] = stated
        report["carpet_diff_pct"] = round(diff * 100, 1)
        if abs(diff) > 0.12:
            warnings.append(f"estimated carpet {carpet:.0f} sqft is {diff:+.0%} off the stated {stated}")

    return errors, warnings, report


def main():
    plan = json.load(open(sys.argv[1], encoding="utf-8"))
    errors, warnings, report = validate(plan)
    print(json.dumps(report, indent=2))
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
