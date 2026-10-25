// plan.json (feet, y down, see schema/floorplan.schema.json)
//   -> engine-neutral scene (metres, y up, x right, z = plan y).
// No dependencies: runs in the browser or in Node (`node builder/cli.mjs plan.json`).

export const FT = 0.3048;
export const OUTDOOR = new Set(['balcony', 'sitout', 'utility', 'terrace']);

const CELL = 0.25; // ft; wall lines are rasterised into cells of this length
const PARAPET_FT = 3.5;
const WINDOW_SPAN = { window: [3, 7], ventilator: [6, 7.5], bay: [1.5, 7] }; // [sill, head] ft
const SIDES = ['top', 'bottom', 'left', 'right'];

function roomEdge(r, side) {
  switch (side) {
    case 'top': return { o: 'h', c: r.y, s: r.x, e: r.x + r.w };
    case 'bottom': return { o: 'h', c: r.y + r.d, s: r.x, e: r.x + r.w };
    case 'left': return { o: 'v', c: r.x, s: r.y, e: r.y + r.d };
    case 'right': return { o: 'v', c: r.x + r.w, s: r.y, e: r.y + r.d };
  }
  throw new Error(`bad side '${side}'`);
}

export function sharedEdge(a, b) {
  for (const sa of SIDES) {
    const ea = roomEdge(a, sa);
    for (const sb of SIDES) {
      const eb = roomEdge(b, sb);
      if (ea.o !== eb.o || Math.abs(ea.c - eb.c) > 0.3) continue;
      const s = Math.max(ea.s, eb.s), e = Math.min(ea.e, eb.e);
      if (e - s > 0.5) return { o: ea.o, c: (ea.c + eb.c) / 2, s, e, side: sa };
    }
  }
  return null;
}

export function planToScene(plan) {
  const D = plan.defaults;
  const H = D.wall_height_ft;
  const rooms = new Map(plan.rooms.map((r) => [r.id, r]));
  const snap = (v) => Math.round(v / CELL);
  const warnings = [];

  // Every room edge is drawn onto a wall line. Two rooms sharing an edge land on
  // the same cells, so shared walls are produced once.
  const lines = new Map();
  const line = (o, c) => {
    const key = `${o}:${snap(c)}`;
    if (!lines.has(key)) lines.set(key, { o, c: snap(c) * CELL, cells: new Map() });
    return lines.get(key);
  };
  for (const r of plan.rooms) {
    for (const side of SIDES) {
      const E = roomEdge(r, side);
      const L = line(E.o, E.c);
      for (let i = snap(E.s); i < snap(E.e); i++) {
        if (!L.cells.has(i)) L.cells.set(i, { owners: new Set(), op: null });
        L.cells.get(i).owners.add(r.id);
      }
    }
  }

  // Openings mark the cells they cover
  const openings = [];
  const cut = (E, center, width, op) => {
    const L = line(E.o, E.c);
    const full = width >= E.e - E.s - 0.01;
    const jamb = full ? 0 : 0.25;
    let a = center - width / 2, b = center + width / 2;
    if (a < E.s + jamb) { b += E.s + jamb - a; a = E.s + jamb; }
    if (b > E.e - jamb) { a -= b - (E.e - jamb); b = E.e - jamb; }
    a = Math.max(a, E.s);
    for (let i = snap(a); i < snap(b); i++) if (L.cells.has(i)) L.cells.get(i).op = op;
    return { a: snap(a) * CELL, b: snap(b) * CELL };
  };

  for (const d of plan.doors) {
    const A = rooms.get(d.from);
    const B = rooms.get(d.to);
    let E;
    if (!A) { warnings.push(`door ${d.id}: unknown room ${d.from}`); continue; }
    if (d.to === 'outside') E = roomEdge(A, d.side === 'auto' ? 'left' : d.side);
    else if (B) E = sharedEdge(A, B);
    if (!E) { warnings.push(`door ${d.id}: no wall between ${d.from} and ${d.to}`); continue; }
    const len = E.e - E.s;
    const width = d.width_ft > 0 ? Math.min(d.width_ft, len) : len;
    const openPlan = d.kind === 'opening' && d.width_ft === 0;
    const head = openPlan ? H : d.kind === 'french' ? Math.max(D.door_height_ft, 7.5) : D.door_height_ft;
    const { a, b } = cut(E, E.s + d.pos * len, width, { kind: 'door', sill: 0, head });
    openings.push({
      id: d.id, type: 'door', kind: d.kind, main: d.main, from: d.from, to: d.to,
      ...segment(E.o, E.c, a, b), sill: 0, head: head * FT,
    });
  }

  for (const w of plan.windows) {
    const A = rooms.get(w.room);
    if (!A) { warnings.push(`window ${w.id}: unknown room ${w.room}`); continue; }
    const E = roomEdge(A, w.side);
    const len = E.e - E.s;
    const [sill, head] = WINDOW_SPAN[w.kind] ?? WINDOW_SPAN.window;
    const { a, b } = cut(E, E.s + w.pos * len, Math.min(w.width_ft, len), { kind: 'window', sill, head });
    openings.push({
      id: w.id, type: 'window', kind: w.kind, room: w.room,
      ...segment(E.o, E.c, a, b), sill: sill * FT, head: head * FT,
    });
  }

  // Walk each line and turn runs of identical cells into wall boxes
  const walls = [];
  for (const L of lines.values()) {
    const idx = [...L.cells.keys()].sort((p, q) => p - q);
    let run = null;
    const flush = () => {
      if (!run) return;
      // Free ends are extended by half a wall so corners close
      const s = run.start * CELL - (L.cells.has(run.start - 1) ? 0 : run.t / 2);
      const e = run.end * CELL + (L.cells.has(run.end) ? 0 : run.t / 2);
      const pieces = [];
      if (!run.op) pieces.push([0, run.top, run.top < H ? 'parapet' : 'wall']);
      else {
        if (run.op.sill > 0) pieces.push([0, run.op.sill, 'sill']);
        if (run.op.head < run.top) pieces.push([run.op.head, run.top, 'lintel']);
      }
      for (const [y0, y1, kind] of pieces) {
        walls.push({ ...segment(L.o, L.c, s, e), y0: y0 * FT, y1: y1 * FT, t: run.t * FT, kind, exterior: run.exterior });
      }
      run = null;
    };
    for (const i of idx) {
      const C = L.cells.get(i);
      const exterior = C.owners.size === 1;
      const outdoorOnly = [...C.owners].every((id) => OUTDOOR.has(rooms.get(id).type));
      const top = exterior && outdoorOnly ? PARAPET_FT : H;
      const t = exterior ? D.ext_wall_thickness_ft : D.int_wall_thickness_ft;
      const sig = `${top}|${t}|${C.op ? `${C.op.kind}${C.op.sill}-${C.op.head}` : ''}`;
      if (run && run.sig === sig && run.end === i) run.end = i + 1;
      else { flush(); run = { sig, start: i, end: i + 1, top, t, op: C.op, exterior }; }
    }
    flush();
  }

  const outRooms = plan.rooms.map((r) => ({
    id: r.id, type: r.type, label: r.label, label_dims: r.label_dims,
    outdoor: OUTDOOR.has(r.type),
    x0: r.x * FT, z0: r.y * FT, x1: (r.x + r.w) * FT, z1: (r.y + r.d) * FT,
  }));

  const furniture = plan.furniture.map((f) => ({
    type: f.type, room: f.room, rot: f.rot,
    x: f.cx * FT, z: f.cy * FT, w: f.w * FT, d: f.d * FT,
  }));

  // Anything below hip height blocks walking; lintels do not
  const colliders = walls
    .filter((w) => w.y0 < 1.0)
    .map((w) => wallBox(w))
    .concat(furniture.filter((f) => f.type !== 'plant').map((f) => ({
      x0: f.x - f.w / 2, z0: f.z - f.d / 2, x1: f.x + f.w / 2, z1: f.z + f.d / 2,
    })));

  const xs = outRooms.flatMap((r) => [r.x0, r.x1]);
  const zs = outRooms.flatMap((r) => [r.z0, r.z1]);
  const bounds = { x0: Math.min(...xs), z0: Math.min(...zs), x1: Math.max(...xs), z1: Math.max(...zs) };

  return {
    format: 'interiordesign.scene', version: '1.0', units: 'm',
    meta: plan.meta, wall_height: H * FT,
    rooms: outRooms, walls, openings, furniture, colliders, bounds,
    spawn: spawnPoint(plan, rooms, openings, outRooms),
    warnings,
  };
}

function segment(o, c, s, e) {
  return o === 'h'
    ? { x0: s * FT, z0: c * FT, x1: e * FT, z1: c * FT }
    : { x0: c * FT, z0: s * FT, x1: c * FT, z1: e * FT };
}

function wallBox(w) {
  const h = w.t / 2;
  return {
    x0: Math.min(w.x0, w.x1) - (w.z0 === w.z1 ? 0 : h), x1: Math.max(w.x0, w.x1) + (w.z0 === w.z1 ? 0 : h),
    z0: Math.min(w.z0, w.z1) - (w.z0 === w.z1 ? h : 0), z1: Math.max(w.z0, w.z1) + (w.z0 === w.z1 ? h : 0),
  };
}

// Just inside the main door, looking into the flat
function spawnPoint(plan, rooms, openings, outRooms) {
  const door = openings.find((o) => o.type === 'door' && o.main);
  const room = outRooms.find((r) => r.id === door?.from) ?? outRooms[0];
  const cx = (room.x0 + room.x1) / 2, cz = (room.z0 + room.z1) / 2;
  if (!door) return { x: cx, z: cz, yaw: 0 };
  const px = (door.x0 + door.x1) / 2, pz = (door.z0 + door.z1) / 2;
  let dx = door.z0 === door.z1 ? 0 : Math.sign(cx - px);
  let dz = door.z0 === door.z1 ? Math.sign(cz - pz) : 0;
  return { x: px + dx * 1.0, z: pz + dz * 1.0, yaw: Math.atan2(-dx, -dz) };
}
