"""Scores an extracted plan.json against a hand-checked reference.

Useful for picking a model: run several models on the same image and compare.

    python extract/compare.py out/plan.json examples/2295-west/plan.json
"""
import json
import re
import sys

from validate import parse_dims, validate


def norm(label):
    return re.sub(r"[^a-z0-9]", "", (label or "").lower())


def centre(r):
    return r["x"] + r["w"] / 2, r["y"] + r["d"] / 2


def match_rooms(got, ref):
    """Pair rooms by printed label first, then by type and nearest centre."""
    pairs, used = [], set()
    for r in ref:
        cands = [g for g in got if id(g) not in used and norm(g["label"]) and norm(g["label"]) == norm(r["label"])]
        if not cands:
            cands = [g for g in got if id(g) not in used and g["type"] == r["type"]]
        if not cands:
            pairs.append((r, None))
            continue
        rx, ry = centre(r)
        best = min(cands, key=lambda g: (centre(g)[0] - rx) ** 2 + (centre(g)[1] - ry) ** 2)
        used.add(id(best))
        pairs.append((r, best))
    extra = [g for g in got if id(g) not in used]
    return pairs, extra


def main():
    got = json.load(open(sys.argv[1], encoding="utf-8"))
    ref = json.load(open(sys.argv[2], encoding="utf-8"))
    pairs, extra = match_rooms(got["rooms"], ref["rooms"])

    print(f"{'reference room':<22}{'extracted as':<22}{'centre off (ft)':>16}{'size off (ft)':>15}  label dims")
    errs = []
    for r, g in pairs:
        if not g:
            print(f"{r['label'] or r['id']:<22}{'MISSING':<22}")
            continue
        off = ((centre(g)[0] - centre(r)[0]) ** 2 + (centre(g)[1] - centre(r)[1]) ** 2) ** 0.5
        size = max(abs(g["w"] - r["w"]), abs(g["d"] - r["d"]))
        errs.append(off)
        dims_ok = parse_dims(g["label_dims"]) == parse_dims(r["label_dims"])
        print(f"{r['label'] or r['id']:<22}{g['label'] or g['id']:<22}{off:>16.1f}{size:>15.1f}  {'ok' if dims_ok else repr(g['label_dims'])}")
    for g in extra:
        print(f"{'(not in reference)':<22}{g['label'] or g['id']:<22}")

    found = sum(1 for _, g in pairs if g)
    e, w, rep = validate(got)
    print()
    print(f"rooms found        {found}/{len(pairs)}  (+{len(extra)} extra)")
    if errs:
        print(f"median centre off  {sorted(errs)[len(errs) // 2]:.1f} ft")
    print(f"doors              {len(got['doors'])} (reference {len(ref['doors'])})")
    print(f"windows            {len(got['windows'])} (reference {len(ref['windows'])})")
    print(f"furniture          {len(got['furniture'])} (reference {len(ref['furniture'])})")
    print(f"meta               title={got['meta']['title']!r} carpet={got['meta']['carpet_area_sqft']}")
    print(f"validation         {len(e)} errors, {len(w)} warnings, {json.dumps(rep)}")
    for x in e:
        print("  error:", x)


if __name__ == "__main__":
    main()
