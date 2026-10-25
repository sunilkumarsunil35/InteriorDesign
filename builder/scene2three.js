// scene (from plan2scene.js) -> THREE.Group. THREE is passed in so this module
// works with whatever three.js build the host app already loads.

const FLOOR_STYLE = {
  balcony: ['deck', '#a0714a'], sitout: ['deck', '#a0714a'], terrace: ['deck', '#a0714a'],
  utility: ['tile', '#bfc3c4'], toilet: ['tile', '#cfd5d8'], kitchen: ['tile', '#ddd6c9'],
  bedroom: ['plank', '#c8a57c'], master_bedroom: ['plank', '#c8a57c'], closet: ['plank', '#c8a57c'],
};
const DEFAULT_FLOOR = ['marble', '#ebe6dd'];

export function buildThree(THREE, scene) {
  const M = materials(THREE);
  const root = new THREE.Group();
  root.name = scene.meta?.title || 'floorplan';
  const floors = new THREE.Group(); floors.name = 'floors';
  const ceilings = new THREE.Group(); ceilings.name = 'ceilings';
  const walls = new THREE.Group(); walls.name = 'walls';
  const fittings = new THREE.Group(); fittings.name = 'doors_windows';
  const furniture = new THREE.Group(); furniture.name = 'furniture';
  root.add(floors, ceilings, walls, fittings, furniture);

  for (const r of scene.rooms) {
    const w = r.x1 - r.x0, d = r.z1 - r.z0;
    const [style, color] = FLOOR_STYLE[r.type] ?? DEFAULT_FLOOR;
    const floor = new THREE.Mesh(plane(THREE, w, d, style === 'plank' ? 1.2 : 0.6), M.floor(style, color));
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(r.x0 + w / 2, 0.001, r.z0 + d / 2);
    floor.receiveShadow = true;
    floor.name = `floor_${r.id}`;
    floor.userData.room = r.id;
    floors.add(floor);
    if (!r.outdoor) {
      // Faces down only, so it is invisible from the dollhouse view above
      const ceil = new THREE.Mesh(new THREE.PlaneGeometry(w, d), M.ceiling);
      ceil.rotation.x = Math.PI / 2;
      ceil.position.set(r.x0 + w / 2, scene.wall_height, r.z0 + d / 2);
      ceil.name = `ceiling_${r.id}`;
      ceilings.add(ceil);
    }
  }

  for (const wl of scene.walls) {
    const horiz = wl.z0 === wl.z1;
    const len = Math.hypot(wl.x1 - wl.x0, wl.z1 - wl.z0);
    const h = wl.y1 - wl.y0;
    const geo = horiz ? new THREE.BoxGeometry(len, h, wl.t) : new THREE.BoxGeometry(wl.t, h, len);
    const mesh = new THREE.Mesh(geo, wl.kind === 'parapet' ? M.parapet : M.wall);
    mesh.position.set((wl.x0 + wl.x1) / 2, wl.y0 + h / 2, (wl.z0 + wl.z1) / 2);
    mesh.castShadow = mesh.receiveShadow = true;
    walls.add(mesh);
  }

  for (const o of scene.openings) {
    const horiz = o.z0 === o.z1;
    const len = Math.hypot(o.x1 - o.x0, o.z1 - o.z0);
    const cx = (o.x0 + o.x1) / 2, cz = (o.z0 + o.z1) / 2;
    if (o.type === 'window' || o.kind === 'french' || o.kind === 'sliding') {
      const h = o.head - o.sill;
      const glass = new THREE.Mesh(horiz ? new THREE.BoxGeometry(len, h, 0.02) : new THREE.BoxGeometry(0.02, h, len), M.glass);
      glass.position.set(cx, o.sill + h / 2, cz);
      if (o.type === 'door') {
        // Sliding glass: two panes, one pushed aside so the doorway is walkable
        glass.scale.set(horiz ? 0.5 : 1, 1, horiz ? 1 : 0.5);
        glass.position[horiz ? 'x' : 'z'] -= len / 4;
        glass.position[horiz ? 'z' : 'x'] += 0.04;
      }
      fittings.add(glass);
      fittings.add(frame(THREE, M.frame, o, horiz, len, cx, cz));
    } else if (o.type === 'door' && o.kind === 'hinged') {
      // Leaf standing open at 90 degrees against the hinge jamb, swung into 'from'
      const leaf = new THREE.Mesh(new THREE.BoxGeometry(len - 0.04, o.head - 0.02, 0.04), o.main ? M.mainDoor : M.door);
      const into = intoSign(scene, o, horiz);
      const hingeX = horiz ? o.x0 + 0.02 : cx, hingeZ = horiz ? cz : o.z0 + 0.02;
      leaf.geometry.translate((len - 0.04) / 2, 0, 0);
      leaf.position.set(hingeX, o.head / 2, hingeZ);
      leaf.rotation.y = horiz ? -into * Math.PI / 2 : (into > 0 ? 0 : Math.PI);
      leaf.castShadow = true;
      fittings.add(leaf);
      fittings.add(frame(THREE, M.frame, o, horiz, len, cx, cz));
    }
  }

  for (const f of scene.furniture) {
    const g = new THREE.Group();
    const sideways = f.rot === 90 || f.rot === 270;
    const lw = sideways ? f.d : f.w, ld = sideways ? f.w : f.d; // local width/depth, front = +z
    (FURNITURE[f.type] ?? FURNITURE.box)(THREE, g, lw, ld, M);
    g.position.set(f.x, 0, f.z);
    g.rotation.y = (f.rot * Math.PI) / 180;
    g.traverse((m) => { if (m.isMesh) { m.castShadow = true; m.receiveShadow = true; } });
    g.name = `${f.type}_${f.room}`;
    furniture.add(g);
  }

  return { root, floors, ceilings };
}

function intoSign(scene, o, horiz) {
  const r = scene.rooms.find((q) => q.id === o.from);
  if (!r) return 1;
  return horiz ? Math.sign((r.z0 + r.z1) / 2 - o.z0) || 1 : Math.sign((r.x0 + r.x1) / 2 - o.x0) || 1;
}

function frame(THREE, mat, o, horiz, len, cx, cz) {
  const g = new THREE.Group();
  const t = 0.05, depth = 0.18;
  const post = (offset) => {
    const m = new THREE.Mesh(horiz ? new THREE.BoxGeometry(t, o.head - o.sill, depth) : new THREE.BoxGeometry(depth, o.head - o.sill, t), mat);
    m.position.set(horiz ? cx + offset : cx, (o.head + o.sill) / 2, horiz ? cz : cz + offset);
    g.add(m);
  };
  post(-len / 2 + t / 2); post(len / 2 - t / 2);
  const top = new THREE.Mesh(horiz ? new THREE.BoxGeometry(len, t, depth) : new THREE.BoxGeometry(depth, t, len), mat);
  top.position.set(cx, o.head - t / 2, cz);
  g.add(top);
  return g;
}

function plane(THREE, w, d, tile) {
  // UVs in world units so one texture tiles at the same scale on every floor
  const geo = new THREE.PlaneGeometry(w, d);
  const uv = geo.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, (uv.getX(i) * w) / tile, (uv.getY(i) * d) / tile);
  return geo;
}

function materials(THREE) {
  const std = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.8, ...extra });
  const floorCache = new Map();
  return {
    wall: std('#f4f0e8', { roughness: 0.95 }),
    parapet: std('#e6e0d6', { roughness: 0.95 }),
    ceiling: std('#fbfaf7', { roughness: 1 }),
    glass: new THREE.MeshPhysicalMaterial({ color: '#bcd6e2', transparent: true, opacity: 0.28, roughness: 0.05, metalness: 0, depthWrite: false }),
    frame: std('#5b4636', { roughness: 0.6 }),
    door: std('#a4805c', { roughness: 0.6 }),
    mainDoor: std('#5e3d25', { roughness: 0.5 }),
    wood: std('#8a6345', { roughness: 0.6 }),
    woodLight: std('#b98d63', { roughness: 0.6 }),
    woodDark: std('#4d3727', { roughness: 0.6 }),
    fabric: std('#d8cfc1', { roughness: 1 }),
    fabricDark: std('#7b8a86', { roughness: 1 }),
    linen: std('#f3efe7', { roughness: 1 }),
    accent: std('#7d9a8c', { roughness: 1 }),
    white: std('#f7f7f5', { roughness: 0.35 }),
    porcelain: std('#fbfbfb', { roughness: 0.15 }),
    steel: std('#b9bec2', { roughness: 0.3, metalness: 0.8 }),
    black: std('#1e1f22', { roughness: 0.4 }),
    granite: std('#3b3a3c', { roughness: 0.3 }),
    leaf: std('#4f7a46', { roughness: 0.9 }),
    terracotta: std('#b4673f', { roughness: 0.9 }),
    gold: std('#c9a24a', { roughness: 0.35, metalness: 0.7 }),
    floor(style, color) {
      const key = style + color;
      if (!floorCache.has(key)) floorCache.set(key, std(color, { map: floorTexture(THREE, style, color), roughness: style === 'marble' ? 0.35 : 0.75 }));
      return floorCache.get(key);
    },
  };
}

function floorTexture(THREE, style, color) {
  if (typeof document === 'undefined') return null;
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#ffffff';
  g.fillRect(0, 0, 256, 256);
  if (style === 'plank' || style === 'deck') {
    for (let i = 0; i < 4; i++) {
      g.fillStyle = `hsl(0 0% ${88 + ((i * 37) % 12)}%)`;
      g.fillRect(0, i * 64, 256, 62);
      g.fillStyle = 'rgba(0,0,0,0.25)';
      g.fillRect(0, i * 64 + 62, 256, 2);
      g.fillRect(((i * 97) % 200) + 20, i * 64, 2, 62);
    }
  } else {
    if (style === 'marble') {
      g.strokeStyle = 'rgba(120,110,100,0.12)';
      g.lineWidth = 1.5;
      for (let i = 0; i < 6; i++) {
        g.beginPath();
        g.moveTo(Math.random() * 256, 0);
        g.bezierCurveTo(Math.random() * 256, 80, Math.random() * 256, 170, Math.random() * 256, 256);
        g.stroke();
      }
    }
    g.fillStyle = 'rgba(0,0,0,0.18)';
    g.fillRect(0, 0, 256, 2);
    g.fillRect(0, 0, 2, 256);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

// Each builder fills a footprint lw (x) by ld (z), front facing +z, sitting on y = 0.
function box(THREE, g, w, h, d, x, y, z, mat) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z);
  g.add(m);
  return m;
}
function cyl(THREE, g, r, h, x, y, z, mat, seg = 20) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, seg), mat);
  m.position.set(x, y, z);
  g.add(m);
  return m;
}

function bed(T, g, w, d, M) {
  box(T, g, w, 0.3, d, 0, 0.15, 0, M.wood);
  box(T, g, w - 0.06, 0.2, d - 0.1, 0, 0.4, 0.03, M.linen);
  box(T, g, w + 0.04, 1.05, 0.08, 0, 0.525, -d / 2 + 0.04, M.woodDark);
  const pw = w > 1.2 ? w * 0.38 : w * 0.7;
  for (const sx of w > 1.2 ? [-1, 1] : [0]) box(T, g, pw, 0.12, 0.38, sx * w * 0.23, 0.56, -d / 2 + 0.35, M.white);
  box(T, g, w - 0.02, 0.05, d * 0.45, 0, 0.525, d * 0.25, M.accent);
}

function seat(T, g, w, d, M, mat) {
  box(T, g, w, 0.42, d, 0, 0.21, 0, mat);
  box(T, g, w, 0.45, 0.22, 0, 0.645, -d / 2 + 0.11, mat);
  for (const sx of [-1, 1]) box(T, g, 0.16, 0.22, d, sx * (w / 2 - 0.08), 0.53, 0, mat);
  const n = Math.max(1, Math.round((w - 0.32) / 0.65));
  const cw = (w - 0.32) / n;
  for (let i = 0; i < n; i++) box(T, g, cw - 0.03, 0.08, d - 0.3, -w / 2 + 0.16 + cw * (i + 0.5), 0.46, 0.08, M.linen);
}

function chair(T, g, x, z, rotY, M) {
  const c = new T.Group();
  box(T, c, 0.44, 0.05, 0.44, 0, 0.46, 0, M.fabricDark);
  box(T, c, 0.44, 0.5, 0.05, 0, 0.73, -0.2, M.fabricDark);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) box(T, c, 0.04, 0.45, 0.04, sx * 0.19, 0.225, sz * 0.19, M.woodDark);
  c.position.set(x, 0, z);
  c.rotation.y = rotY;
  g.add(c);
}

const FURNITURE = {
  box: (T, g, w, d, M) => box(T, g, w, 0.8, d, 0, 0.4, 0, M.woodLight),
  bed_double: bed,
  bed_single: bed,
  wardrobe(T, g, w, d, M) {
    box(T, g, w, 2.1, d, 0, 1.05, 0, M.woodLight);
    const n = Math.max(2, Math.round(w / 0.5));
    for (let i = 1; i < n; i++) box(T, g, 0.01, 2.0, 0.01, -w / 2 + (w * i) / n, 1.05, d / 2 + 0.005, M.woodDark);
    for (let i = 0; i < n; i++) box(T, g, 0.02, 0.25, 0.03, -w / 2 + (w * (i + 0.5)) / n + (i % 2 ? -0.08 : 0.08), 1.05, d / 2 + 0.015, M.steel);
  },
  sofa: (T, g, w, d, M) => seat(T, g, w, d, M, M.fabric),
  armchair: (T, g, w, d, M) => seat(T, g, w, d, M, M.wood),
  coffee_table(T, g, w, d, M) {
    box(T, g, w, 0.05, d, 0, 0.4, 0, M.woodDark);
    box(T, g, w * 0.8, 0.35, d * 0.75, 0, 0.19, 0, M.wood);
  },
  side_table(T, g, w, d, M) {
    box(T, g, w * 0.9, 0.55, d * 0.9, 0, 0.275, 0, M.wood);
    cyl(T, g, 0.08, 0.3, 0, 0.7, 0, M.linen);
  },
  tv_unit(T, g, w, d, M) {
    box(T, g, w, 0.45, d, 0, 0.225, 0, M.woodDark);
    box(T, g, Math.min(w * 0.85, 1.45), 0.82, 0.05, 0, 1.25, -d / 2 + 0.06, M.black);
  },
  dining_table(T, g, w, d, M) {
    box(T, g, w, 0.05, d, 0, 0.75, 0, M.glass);
    box(T, g, w - 0.1, 0.03, d - 0.1, 0, 0.71, 0, M.woodDark);
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) box(T, g, 0.06, 0.7, 0.06, sx * (w / 2 - 0.1), 0.35, sz * (d / 2 - 0.1), M.woodDark);
    const long = d >= w;
    const n = Math.max(1, Math.floor((long ? d : w) / 0.62));
    for (let i = 0; i < n; i++) {
      const t = -((long ? d : w) / 2) + ((long ? d : w) * (i + 0.5)) / n;
      for (const s of [-1, 1]) {
        if (long) chair(T, g, s * (w / 2 + 0.25), t, s > 0 ? -Math.PI / 2 : Math.PI / 2, M);
        else chair(T, g, t, s * (d / 2 + 0.25), s > 0 ? Math.PI : 0, M);
      }
    }
  },
  study_table(T, g, w, d, M) {
    box(T, g, w, 0.04, d, 0, 0.74, 0, M.woodLight);
    for (const sx of [-1, 1]) box(T, g, 0.04, 0.72, d, sx * (w / 2 - 0.02), 0.36, 0, M.woodLight);
  },
  bookshelf(T, g, w, d, M) {
    box(T, g, w, 1.9, d, 0, 0.95, 0, M.woodLight);
    for (let i = 1; i < 5; i++) box(T, g, w - 0.04, 0.02, 0.02, 0, i * 0.38, d / 2, M.woodDark);
  },
  kitchen_counter(T, g, w, d, M) {
    box(T, g, w, 0.86, d - 0.04, 0, 0.43, -0.02, M.white);
    box(T, g, w, 0.04, d, 0, 0.88, 0, M.granite);
    box(T, g, w, 0.7, Math.min(0.35, d * 0.6), 0, 1.85, -d / 2 + Math.min(0.35, d * 0.6) / 2, M.woodLight);
  },
  stove(T, g, w, d, M) {
    box(T, g, w * 0.8, 0.02, d * 0.6, 0, 0.91, 0, M.black);
    for (const sx of [-1, 1]) cyl(T, g, 0.08, 0.02, sx * w * 0.2, 0.925, 0, M.steel);
  },
  sink(T, g, w, d, M) {
    box(T, g, w * 0.6, 0.02, d * 0.55, 0, 0.905, 0, M.steel);
    cyl(T, g, 0.015, 0.3, 0, 1.05, -d * 0.35, M.steel, 8);
  },
  fridge(T, g, w, d, M) {
    box(T, g, w * 0.95, 1.8, d * 0.9, 0, 0.9, 0, M.steel);
    box(T, g, 0.03, 0.5, 0.03, w * 0.3, 1.2, d * 0.45 + 0.02, M.black);
  },
  wc(T, g, w, d, M) {
    box(T, g, 0.4, 0.4, 0.5, 0, 0.2, 0.05, M.porcelain);
    box(T, g, 0.42, 0.38, 0.18, 0, 0.6, -Math.min(d, 0.7) / 2 + 0.09, M.porcelain);
  },
  washbasin(T, g, w, d, M) {
    box(T, g, Math.min(w, 0.6), 0.15, Math.min(d, 0.45), 0, 0.8, -Math.min(d, 0.45) / 2 + 0.1, M.porcelain);
    box(T, g, 0.08, 0.72, 0.08, 0, 0.36, -0.05, M.porcelain);
    box(T, g, Math.min(w, 0.55), 0.7, 0.02, 0, 1.45, -Math.min(d, 0.45) / 2 - 0.12, M.steel);
  },
  shower(T, g, w, d, M) {
    box(T, g, w, 0.04, d, 0, 0.02, 0, M.tile ?? M.porcelain);
    box(T, g, w, 2.0, 0.02, 0, 1.0, d / 2, M.glass);
    cyl(T, g, 0.12, 0.02, 0, 2.1, -d / 2 + 0.25, M.steel);
  },
  washing_machine(T, g, w, d, M) {
    box(T, g, w * 0.95, 0.85, d * 0.95, 0, 0.425, 0, M.white);
    const door = cyl(T, g, Math.min(w, 0.6) * 0.3, 0.03, 0, 0.42, d * 0.475 + 0.01, M.black);
    door.rotation.x = Math.PI / 2;
  },
  dishwasher(T, g, w, d, M) {
    box(T, g, w * 0.95, 0.85, d * 0.95, 0, 0.425, 0, M.steel);
    box(T, g, w * 0.6, 0.03, 0.03, 0, 0.78, d * 0.475 + 0.01, M.black);
  },
  pooja_unit(T, g, w, d, M) {
    box(T, g, w, 0.75, d, 0, 0.375, 0, M.woodDark);
    box(T, g, w * 0.9, 0.9, d * 0.6, 0, 1.2, -d * 0.2, M.wood);
    box(T, g, w * 0.95, 0.05, d, 0, 1.68, 0, M.woodDark);
    cyl(T, g, 0.06, 0.12, 0, 0.81, d * 0.25, M.gold, 12);
  },
  plant(T, g, w, d, M) {
    const r = Math.min(w, d) * 0.3;
    cyl(T, g, r, 0.4, 0, 0.2, 0, M.terracotta);
    const s = new T.Mesh(new T.SphereGeometry(r * 1.6, 14, 10), M.leaf);
    s.position.y = 0.4 + r * 1.3;
    g.add(s);
  },
};
