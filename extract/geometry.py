"""Floor-plan geometry from pixels. No AI.

walls -> grid of cells -> spaces (cells joined where no wall separates them)
-> room rectangles once label positions are known -> doors and windows from
gaps in the walls. Coordinates are pixels of the prepared image unless noted.
"""
from collections import deque

import cv2
import numpy as np

# Two neighbouring cells are one space only when the boundary between them has
# almost no wall. A doorway always leaves jambs, so even a wide door stays a wall.
MERGE_BELOW = 0.05
OUTSIDE_BELOW = 0.3  # enclosure (walls + window glazing) under which outside leaks in


class Plan:
    def __init__(self, bgr):
        self.img = bgr
        self.H, self.W = bgr.shape[:2]
        b, g, r = [bgr[:, :, i].astype(np.int16) for i in range(3)]
        gray = (b + g + r) // 3
        neutral = (np.abs(r - g) < 24) & (np.abs(g - b) < 24)
        k = max(3, round(min(self.W, self.H) / 190))
        self.L = max(20, round(min(self.W, self.H) * 0.045))
        dark = ((gray < 140) & neutral).astype(np.uint8) * 255
        thick = cv2.morphologyEx(dark, cv2.MORPH_OPEN, rect(k, k))
        # Solid areas thicker than any wall are counters, TV units and shafts, not walls
        T = round(min(self.W, self.H) * 0.04)
        blobs = cv2.dilate(cv2.morphologyEx(thick, cv2.MORPH_OPEN, rect(T, T)), rect(3, 3))
        thick = cv2.bitwise_and(thick, cv2.bitwise_not(blobs))
        self.blobs = blobs  # still solid when looking for door and window gaps
        self.hmask = keep_walls(cv2.morphologyEx(thick, cv2.MORPH_OPEN, rect(self.L, 1)), gray, False, round(1.6 * k))
        self.vmask = keep_walls(cv2.morphologyEx(thick, cv2.MORPH_OPEN, rect(1, self.L)), gray, True, round(1.6 * k))
        # Thin long lines (window glazing, balcony rails) count as enclosure, not as walls
        light = ((gray < 200) & neutral).astype(np.uint8) * 255
        self.hthin = cv2.morphologyEx(light, cv2.MORPH_OPEN, rect(self.L, 1)) | self.hmask
        self.vthin = cv2.morphologyEx(light, cv2.MORPH_OPEN, rect(1, self.L)) | self.vmask

        self.xs = lines(self.vmask, vertical=True, min_len=self.L, tol=k * 2)
        self.ys = lines(self.hmask, vertical=False, min_len=self.L, tol=k * 2)
        # Balcony rails are thin lines beyond the outermost walls; they close the plan
        for x in lines(self.vthin, vertical=True, min_len=3 * self.L, tol=k * 2):
            if x > self.xs[-1] + 2 * k: self.xs.append(x)
            if x < self.xs[0] - 2 * k: self.xs.insert(0, x)
        for y in lines(self.hthin, vertical=False, min_len=3 * self.L, tol=k * 2):
            if y > self.ys[-1] + 2 * k: self.ys.append(y)
            if y < self.ys[0] - 2 * k: self.ys.insert(0, y)
        self.xs, self.ys = sorted(set(self.xs)), sorted(set(self.ys))
        self.band = k  # half-width of the strip sampled around a line
        self._cells()

    # ----- cells and spaces -----
    def _cells(self):
        nx, ny = len(self.xs) - 1, len(self.ys) - 1
        self.nx, self.ny = nx, ny
        # Wall coverage on every internal boundary
        self.cov_v = np.zeros((nx + 1, ny))  # boundary at xs[i] between cells (i-1,j) and (i,j)
        self.cov_h = np.zeros((nx, ny + 1))
        self.thin_v = np.zeros((nx + 1, ny))
        self.thin_h = np.zeros((nx, ny + 1))
        for i in range(nx + 1):
            for j in range(ny):
                self.cov_v[i, j] = self.coverage('v', self.xs[i], self.ys[j], self.ys[j + 1])
                self.thin_v[i, j] = self.coverage('v', self.xs[i], self.ys[j], self.ys[j + 1], thin=True)
        for i in range(nx):
            for j in range(ny + 1):
                self.cov_h[i, j] = self.coverage('h', self.ys[j], self.xs[i], self.xs[i + 1])
                self.thin_h[i, j] = self.coverage('h', self.ys[j], self.xs[i], self.xs[i + 1], thin=True)

        # Outside = flood from the grid border through boundaries with no enclosure
        outside = np.zeros((nx, ny), bool)
        q = deque()
        for i in range(nx):
            for j in range(ny):
                if (i in (0, nx - 1) and self.thin_v[0 if i == 0 else nx, j] < OUTSIDE_BELOW) or \
                   (j in (0, ny - 1) and self.thin_h[i, 0 if j == 0 else ny] < OUTSIDE_BELOW):
                    outside[i, j] = True
                    q.append((i, j))
        while q:
            i, j = q.popleft()
            for a, b, c in self._neighbours(i, j, thin=True):
                if c < OUTSIDE_BELOW and not outside[a, b]:
                    outside[a, b] = True
                    q.append((a, b))
        # Cells that are mostly wall are not floor
        solid = np.zeros((nx, ny), bool)
        wall = self.hmask | self.vmask
        for i in range(nx):
            for j in range(ny):
                x0, x1, y0, y1 = self.xs[i], self.xs[i + 1], self.ys[j], self.ys[j + 1]
                if x1 - x0 < 3 or y1 - y0 < 3:
                    solid[i, j] = True
                    continue
                solid[i, j] = (wall[y0:y1, x0:x1] > 0).mean() > 0.6
        self.inside = ~outside & ~solid

        # Spaces: union of inside cells joined across open boundaries
        self.space = -np.ones((nx, ny), int)
        sid = 0
        for i in range(nx):
            for j in range(ny):
                if not self.inside[i, j] or self.space[i, j] >= 0:
                    continue
                self.space[i, j] = sid
                q = deque([(i, j)])
                while q:
                    a, b = q.popleft()
                    for c, d, cov in self._neighbours(a, b):
                        if cov < MERGE_BELOW and self.inside[c, d] and self.space[c, d] < 0:
                            self.space[c, d] = sid
                            q.append((c, d))
                sid += 1
        self.nspaces = sid

    def _neighbours(self, i, j, thin=False):
        cv, ch = (self.thin_v, self.thin_h) if thin else (self.cov_v, self.cov_h)
        if i > 0: yield i - 1, j, cv[i, j]
        if i < self.nx - 1: yield i + 1, j, cv[i + 1, j]
        if j > 0: yield i, j - 1, ch[i, j]
        if j < self.ny - 1: yield i, j + 1, ch[i, j + 1]

    def coverage(self, o, c, s, e, thin=False):
        return wall_runs(self, o, c, s, e, thin)[0]

    def cell_at(self, x, y):
        i = int(np.searchsorted(self.xs, x)) - 1
        j = int(np.searchsorted(self.ys, y)) - 1
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i, j
        return None

    def cell_box(self, i, j):
        return self.xs[i], self.ys[j], self.xs[i + 1], self.ys[j + 1]


def wall_runs(p, o, c, s, e, thin=False):
    """Coverage of the strip along a line, plus the gaps (pixel intervals with no wall)."""
    m = (p.hthin if o == 'h' else p.vthin) if thin else (p.hmask if o == 'h' else p.vmask)
    b, inset = p.band, p.band + 1
    s, e = int(s) + inset, int(e) - inset
    if e - s < 2:
        return 1.0, []
    m = m | p.blobs
    strip = m[c - b:c + b + 1, s:e] if o == 'h' else m[s:e, c - b:c + b + 1].T
    hit = (strip > 0).any(axis=0)
    gaps, start = [], None
    for k, v in enumerate(hit):
        if not v and start is None:
            start = k
        if v and start is not None:
            gaps.append((s + start, s + k)); start = None
    if start is not None:
        gaps.append((s + start, e))
    return float(hit.mean()), gaps


def rect(w, h):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))


def keep_walls(mask, gray, vertical, min_thick):
    """Drop furniture outlines: walls are thick and mid-grey, furniture edges thin and darker."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        x, y, w, h, _ = stats[k]
        thick = w if vertical else h
        if thick >= min_thick and np.median(gray[lab == k]) >= 100:
            keep[k] = True
    return np.where(keep[lab], 255, 0).astype(np.uint8)


def lines(mask, vertical, min_len, tol):
    """Positions of long straight walls, clustered (one position per wall)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    segs = []
    for x, y, w, h, area in stats[1:]:
        length, thick = (h, w) if vertical else (w, h)
        if length >= min_len:
            segs.append(((x + w / 2) if vertical else (y + h / 2), length))
    segs.sort()
    out, cur = [], []
    for pos, length in segs:
        if cur and pos - cur[-1][0] > tol:
            out.append(cur); cur = []
        cur.append((pos, length))
    if cur:
        out.append(cur)
    return [round(sum(p * l for p, l in g) / sum(l for _, l in g)) for g in out]


def max_rectangle(ok, seed):
    """Largest all-True rectangle in a boolean grid that contains seed (i, j). Returns (i0, j0, i1, j1) inclusive."""
    nx, ny = ok.shape
    si, sj = seed
    best, best_area = (si, sj, si, sj), 0
    for i0 in range(si, -1, -1):
        if not ok[i0, sj]: break
        for i1 in range(si, nx):
            if not ok[i1, sj]: break
            if not ok[i0:i1 + 1, sj].all(): break
            j0 = sj
            while j0 - 1 >= 0 and ok[i0:i1 + 1, j0 - 1].all(): j0 -= 1
            j1 = sj
            while j1 + 1 < ny and ok[i0:i1 + 1, j1 + 1].all(): j1 += 1
            area = (i1 - i0 + 1) * (j1 - j0 + 1)
            if area > best_area:
                best, best_area = (i0, j0, i1, j1), area
    return best


def debug_image(p, rooms=None, path=None):
    out = p.img.copy()
    out[p.hmask > 0] = (0, 0, 255)
    out[p.vmask > 0] = (255, 0, 0)
    rng = np.random.default_rng(3)
    colors = rng.integers(60, 230, (max(p.nspaces, 1), 3))
    over = out.copy()
    for i in range(p.nx):
        for j in range(p.ny):
            if p.space[i, j] >= 0:
                x0, y0, x1, y1 = p.cell_box(i, j)
                cv2.rectangle(over, (x0, y0), (x1, y1), tuple(int(c) for c in colors[p.space[i, j]]), -1)
    out = cv2.addWeighted(over, 0.45, out, 0.55, 0)
    for x in p.xs: cv2.line(out, (x, 0), (x, p.H), (0, 160, 0), 1)
    for y in p.ys: cv2.line(out, (0, y), (p.W, y), (0, 160, 0), 1)
    for r in rooms or []:
        x0, y0, x1, y1 = r['px']
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), 3)
        cv2.putText(out, r['id'], (x0 + 6, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    if path:
        cv2.imwrite(path, out)
    return out
