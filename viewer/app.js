import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { planToScene, FT } from '../builder/plan2scene.js';
import { buildThree } from '../builder/scene2three.js';

// Human-scale movement
const EYE = 1.6;        // m, standing eye height
const RADIUS = 0.25;    // m, body radius for collision
const WALK = 1.4;       // m/s
const BRISK = 2.1;      // m/s with Shift
const ACCEL_T = 0.18;   // s, time constant for speeding up / slowing down
const STRIDE = 0.72;    // m per step
const BOB = 0.022;      // m, vertical head bob per step
const SWAY = 0.012;     // m, side-to-side sway
const TURN = 1.9;       // rad/s for keyboard turning
const LOOK = 0.0022;    // rad per mouse pixel

const params = new URLSearchParams(location.search);
const DEFAULT_PLAN = 'examples/2295-west/plan.json';
const DEFAULT_IMAGE = 'examples/2295-west/input-crop.png';
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const isTouch = matchMedia('(pointer: coarse)').matches;

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.45;

const camera = new THREE.PerspectiveCamera(70, 1, 0.05, 400);
camera.rotation.order = 'YXZ';
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.maxPolarAngle = 1.38;
controls.minDistance = 3;
controls.maxDistance = 70;

const hemi = new THREE.HemisphereLight(0xffffff, 0x9a8b78, 1.15);
const sun = new THREE.DirectionalLight(0xfff0da, 2.3);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.02;
scene.add(hemi, sun, sun.target);

const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({ roughness: 1 }));
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.02;
ground.receiveShadow = true;
scene.add(ground);

const clipPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 1.3);

let plan = null, S = null, model = null;
let mode = 'orbit';             // 'orbit' | 'walk' | 'tween'
let orbitPose = null;
let tween = null;
const player = { x: 0, z: 0, yaw: 0, pitch: 0, vx: 0, vz: 0, phase: 0 };
const keys = new Set();
const joy = { id: null, ox: 0, oy: 0, x: 0, y: 0 };
let lookTouch = null;
let mouseDrag = null;

// ---------- theme ----------
function applyTheme() {
  const css = getComputedStyle(document.documentElement);
  scene.background = new THREE.Color(css.getPropertyValue('--scene-sky').trim() || '#dbe4e8');
  ground.material.color.set(css.getPropertyValue('--scene-ground').trim() || '#c5ccc7');
  drawMap();
}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
new MutationObserver(applyTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

// ---------- loading ----------
async function loadFromUrl(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    loadPlan(await res.json());
  } catch (err) {
    $('title').textContent = 'Could not load the plan';
    $('facts').textContent = `${url}: ${err.message}. Use "Load plan.json" to open one.`;
  }
}

function loadPlan(p) {
  plan = p;
  S = planToScene(plan);
  if (model) {
    scene.remove(model.root);
    model.root.traverse((o) => { if (o.isMesh) o.geometry.dispose(); });
  }
  model = buildThree(THREE, S);
  scene.add(model.root);

  const b = S.bounds;
  const cx = (b.x0 + b.x1) / 2, cz = (b.z0 + b.z1) / 2, size = Math.max(b.x1 - b.x0, b.z1 - b.z0);
  const north = ((plan.meta.north_deg || 0) * Math.PI) / 180;
  const off = new THREE.Vector3(0.45 * size, 1.25 * size, 0.85 * size).applyAxisAngle(new THREE.Vector3(0, 1, 0), -north);
  sun.position.set(cx + off.x, off.y, cz + off.z);
  sun.target.position.set(cx, 0, cz);
  const sc = sun.shadow.camera;
  sc.left = sc.bottom = -size * 0.8;
  sc.right = sc.top = size * 0.8;
  sc.near = 1; sc.far = size * 4;
  sc.updateProjectionMatrix();

  controls.target.set(cx, 0, cz);
  const fit = Math.max(1, 1.3 / camera.aspect); // pull back on portrait screens
  camera.position.set(cx + size * 0.35 * fit, size * 1.05 * fit, cz + size * 0.95 * fit);
  controls.update();
  orbitPose = null;
  setMode('orbit');
  fillInfo();
  drawMap();
  if (S.warnings.length) toast(`${S.warnings.length} warning(s) building the plan; see console`), console.warn(S.warnings);
}

function fillInfo() {
  const m = plan.meta;
  $('eyebrow').textContent = 'Plan JSON to 3D';
  $('title').textContent = m.title || 'Untitled plan';
  const facts = [
    m.bhk && `<b>${m.bhk}</b>`,
    m.facing && m.facing !== 'unknown' && `<b>${cap(m.facing)}</b> facing`,
    m.total_area_sqft && `<b>${m.total_area_sqft}</b> sft total`,
    m.carpet_area_sqft && `<b>${m.carpet_area_sqft}</b> sft carpet`,
  ].filter(Boolean);
  $('facts').innerHTML = facts.join('<span aria-hidden="true">·</span>');
  const t = plan.defaults.int_wall_thickness_ft;
  const est = plan.rooms.filter((r) => !['balcony', 'sitout', 'utility', 'terrace'].includes(r.type))
    .reduce((a, r) => a + Math.max(r.w - t, 0) * Math.max(r.d - t, 0), 0);
  const diff = m.carpet_area_sqft ? ((est - m.carpet_area_sqft) / m.carpet_area_sqft) * 100 : null;
  $('check').innerHTML = `${plan.rooms.length} rooms · ${plan.doors.length} doors · ${plan.furniture.length} items<br>`
    + `Rebuilt carpet area <b>${Math.round(est)} sft</b>${diff === null ? '' : ` (${diff >= 0 ? '+' : ''}${diff.toFixed(1)}% vs printed)`}`;
}

// ---------- modes ----------
function setMode(next) {
  mode = next;
  $('btnOrbit').setAttribute('aria-pressed', String(next === 'orbit'));
  $('btnWalk').setAttribute('aria-pressed', String(next === 'walk'));
  controls.enabled = next === 'orbit';
  renderer.clippingPlanes = next === 'orbit' && $('optCutaway').checked ? [clipPlane] : [];
  if (next !== 'walk') {
    $('roomTag').hidden = true;
    $('joy').hidden = true;
    if (document.pointerLockElement) document.exitPointerLock();
  }
  updateHint();
}

function updateHint() {
  const h = $('hint');
  if (mode === 'walk') {
    h.innerHTML = isTouch
      ? 'Left thumb: walk · Right thumb: look around · Map: jump to a room'
      : document.pointerLockElement
        ? '<kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> walk · <kbd>Shift</kbd> brisk · mouse look · <kbd>Esc</kbd> release mouse · <kbd>Tab</kbd> dollhouse'
        : 'Click the view to look with the mouse, or drag · <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> / arrows walk · <kbd>Tab</kbd> dollhouse';
  } else {
    h.innerHTML = isTouch
      ? 'Drag to orbit · Pinch to zoom · Double-tap a floor to walk there'
      : 'Drag to orbit · Scroll to zoom · Double-click a floor to walk there · <kbd>Tab</kbd> walk in';
  }
}

function walkPose(x, z, yaw) {
  return { pos: new THREE.Vector3(x, EYE, z), quat: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, yaw, 0, 'YXZ')) };
}

function startTween(to, done) {
  tween = {
    fromPos: camera.position.clone(), fromQuat: camera.quaternion.clone(),
    toPos: to.pos, toQuat: to.quat, t: 0, dur: reduceMotion ? 0.01 : 1.0, done,
  };
  mode = 'tween';
  controls.enabled = false;
}

function enterWalk(x, z, yaw) {
  if (!S) return;
  if (mode === 'orbit') orbitPose = { pos: camera.position.clone(), quat: camera.quaternion.clone(), target: controls.target.clone() };
  [x, z] = collide(x, z);
  Object.assign(player, { x, z, yaw, pitch: 0, vx: 0, vz: 0, phase: 0 });
  startTween(walkPose(x, z, yaw), () => setMode('walk'));
}

function enterOrbit() {
  if (!S || mode === 'orbit') return;
  const b = S.bounds;
  const size = Math.max(b.x1 - b.x0, b.z1 - b.z0);
  let to = orbitPose;
  if (!to) {
    // Hover above where the player stands, looking down at them
    const pos = new THREE.Vector3(player.x + size * 0.3, size * 0.9, player.z + size * 0.7);
    const target = new THREE.Vector3(player.x, 0, player.z);
    const m = new THREE.Matrix4().lookAt(pos, target, new THREE.Vector3(0, 1, 0));
    to = { pos, quat: new THREE.Quaternion().setFromRotationMatrix(m), target };
  }
  if (document.pointerLockElement) document.exitPointerLock();
  startTween(to, () => { controls.target.copy(to.target); setMode('orbit'); controls.update(); });
}

// ---------- walking ----------
function collide(px, pz) {
  if (!S) return [px, pz];
  for (let iter = 0; iter < 4; iter++) {
    let moved = false;
    for (const b of S.colliders) {
      const cx = Math.min(Math.max(px, b.x0), b.x1), cz = Math.min(Math.max(pz, b.z0), b.z1);
      const dx = px - cx, dz = pz - cz, d2 = dx * dx + dz * dz;
      if (d2 >= RADIUS * RADIUS) continue;
      if (d2 > 1e-9) {
        const d = Math.sqrt(d2);
        px = cx + (dx / d) * RADIUS;
        pz = cz + (dz / d) * RADIUS;
      } else {
        const l = px - b.x0, r = b.x1 - px, t = pz - b.z0, bt = b.z1 - pz, m = Math.min(l, r, t, bt);
        if (m === l) px = b.x0 - RADIUS; else if (m === r) px = b.x1 + RADIUS; else if (m === t) pz = b.z0 - RADIUS; else pz = b.z1 + RADIUS;
      }
      moved = true;
    }
    if (!moved) break;
  }
  return [px, pz];
}

function stepWalk(dt) {
  const k = (c) => (keys.has(c) ? 1 : 0);
  player.yaw += (k('ArrowLeft') - k('ArrowRight')) * TURN * dt;
  let ix = k('KeyD') - k('KeyA') + joy.x;
  let iz = k('KeyS') + k('ArrowDown') - k('KeyW') - k('ArrowUp') + joy.y;
  const len = Math.hypot(ix, iz);
  if (len > 1) { ix /= len; iz /= len; }
  const speed = keys.has('ShiftLeft') || keys.has('ShiftRight') ? BRISK : WALK;
  const sin = Math.sin(player.yaw), cos = Math.cos(player.yaw);
  // forward = (-sin, -cos), right = (cos, -sin)
  const tx = (cos * ix + sin * iz) * speed;
  const tz = (-sin * ix + cos * iz) * speed;
  const a = 1 - Math.exp(-dt / ACCEL_T);
  player.vx += (tx - player.vx) * a;
  player.vz += (tz - player.vz) * a;
  [player.x, player.z] = collide(player.x + player.vx * dt, player.z + player.vz * dt);

  const v = Math.hypot(player.vx, player.vz);
  player.phase += ((v * dt) / STRIDE) * Math.PI;
  const amp = $('optBob').checked && !reduceMotion ? Math.min(1, v / WALK) : 0;
  const bob = (Math.abs(Math.sin(player.phase)) - 0.5) * 2 * BOB * amp;
  const sway = Math.sin(player.phase) * SWAY * amp;
  camera.position.set(player.x + cos * sway, EYE + bob, player.z - sin * sway);
  camera.rotation.set(player.pitch, player.yaw, 0);
  showRoom();
}

let lastRoom = null;
function showRoom() {
  const fx = player.x / FT, fz = player.z / FT;
  const inside = plan.rooms.filter((r) => fx >= r.x && fx <= r.x + r.w && fz >= r.y && fz <= r.y + r.d);
  const r = inside.sort((p, q) => p.w * p.d - q.w * q.d)[0];
  if (r === lastRoom) return;
  lastRoom = r;
  $('roomTag').hidden = !r;
  if (!r) return;
  $('roomName').textContent = r.label || cap(r.type);
  $('roomDims').textContent = r.label_dims ? r.label_dims.replace(/x/i, ' × ') : `${ftIn(r.w)} × ${ftIn(r.d)}`;
}

// ---------- input ----------
addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return;
  if (e.code === 'Tab') {
    e.preventDefault();
    if (mode === 'orbit') enterWalk(S.spawn.x, S.spawn.z, S.spawn.yaw);
    else if (mode === 'walk') enterOrbit();
    return;
  }
  if (mode === 'walk' && e.code.startsWith('Arrow')) e.preventDefault();
  keys.add(e.code);
});
addEventListener('keyup', (e) => keys.delete(e.code));
addEventListener('blur', () => keys.clear());

document.addEventListener('pointerlockchange', updateHint);
document.addEventListener('mousemove', (e) => {
  if (mode !== 'walk') return;
  if (document.pointerLockElement === canvas) look(e.movementX, e.movementY, LOOK);
});
function look(dx, dy, s) {
  player.yaw -= dx * s;
  player.pitch = Math.max(-1.2, Math.min(1.2, player.pitch - dy * s));
}

canvas.addEventListener('pointerdown', (e) => {
  if (mode !== 'walk') return;
  if (e.pointerType === 'touch') {
    canvas.setPointerCapture(e.pointerId);
    if (e.clientX < innerWidth / 2 && joy.id === null) {
      Object.assign(joy, { id: e.pointerId, ox: e.clientX, oy: e.clientY, x: 0, y: 0 });
      const j = $('joy');
      j.style.left = `${e.clientX}px`;
      j.style.top = `${e.clientY}px`;
      $('knob').style.transform = '';
      j.hidden = false;
    } else if (!lookTouch) {
      lookTouch = { id: e.pointerId, x: e.clientX, y: e.clientY };
    }
    return;
  }
  if (!document.pointerLockElement) {
    mouseDrag = { x: e.clientX, y: e.clientY, moved: false };
  }
});
canvas.addEventListener('pointermove', (e) => {
  if (mode !== 'walk') return;
  if (e.pointerId === joy.id) {
    const R = 50;
    let dx = e.clientX - joy.ox, dy = e.clientY - joy.oy;
    const l = Math.hypot(dx, dy);
    if (l > R) { dx *= R / l; dy *= R / l; }
    joy.x = dx / R;
    joy.y = dy / R;
    $('knob').style.transform = `translate(${dx}px, ${dy}px)`;
  } else if (lookTouch && e.pointerId === lookTouch.id) {
    look(e.clientX - lookTouch.x, e.clientY - lookTouch.y, 0.005);
    lookTouch.x = e.clientX;
    lookTouch.y = e.clientY;
  } else if (mouseDrag && e.buttons) {
    look(e.clientX - mouseDrag.x, e.clientY - mouseDrag.y, 0.004);
    mouseDrag.moved = mouseDrag.moved || Math.abs(e.clientX - mouseDrag.x) + Math.abs(e.clientY - mouseDrag.y) > 2;
    mouseDrag.x = e.clientX;
    mouseDrag.y = e.clientY;
  }
});
const endPointer = (e) => {
  if (e.pointerId === joy.id) { joy.id = null; joy.x = joy.y = 0; $('joy').hidden = true; }
  if (lookTouch && e.pointerId === lookTouch.id) lookTouch = null;
  if (mouseDrag && e.pointerType === 'mouse') {
    // A click without dragging asks for mouse-look; failure (phones, some embeds) is fine
    if (!mouseDrag.moved && mode === 'walk' && canvas.requestPointerLock) {
      try { const p = canvas.requestPointerLock(); if (p?.catch) p.catch(() => {}); } catch { /* not supported */ }
    }
    mouseDrag = null;
  }
};
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);

canvas.addEventListener('dblclick', (e) => {
  if (mode !== 'orbit' || !model) return;
  const p = floorHit(e.clientX, e.clientY);
  if (!p) return;
  const dx = p.x - camera.position.x, dz = p.z - camera.position.z;
  enterWalk(p.x, p.z, Math.atan2(-dx, -dz));
});
// Double tap for touch screens
let lastTap = 0;
canvas.addEventListener('pointerup', (e) => {
  if (e.pointerType !== 'touch' || mode !== 'orbit') return;
  const now = performance.now();
  if (now - lastTap < 320) canvas.dispatchEvent(new MouseEvent('dblclick', { clientX: e.clientX, clientY: e.clientY }));
  lastTap = now;
});

const ray = new THREE.Raycaster();
function floorHit(cx, cy) {
  const r = canvas.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1), camera);
  const hit = ray.intersectObjects(model.floors.children, false)[0];
  return hit?.point ?? null;
}

$('btnOrbit').addEventListener('click', enterOrbit);
$('btnWalk').addEventListener('click', () => {
  if (mode === 'orbit') enterWalk(S.spawn.x, S.spawn.z, S.spawn.yaw);
});
$('optCutaway').addEventListener('change', () => setMode(mode === 'tween' ? 'orbit' : mode));

// ---------- minimap ----------
const map = $('minimap');
const mctx = map.getContext('2d');
let mapXf = null;
function drawMap() {
  if (!S) return;
  const css = getComputedStyle(document.documentElement);
  const v = (n) => css.getPropertyValue(n).trim();
  const W = map.width, H = map.height, pad = 18;
  const b = S.bounds;
  const s = Math.min((W - pad * 2) / (b.x1 - b.x0), (H - pad * 2) / (b.z1 - b.z0));
  const ox = (W - (b.x1 - b.x0) * s) / 2 - b.x0 * s, oz = (H - (b.z1 - b.z0) * s) / 2 - b.z0 * s;
  mapXf = { s, ox, oz };
  mctx.clearRect(0, 0, W, H);
  for (const r of S.rooms) {
    mctx.fillStyle = r.outdoor ? v('--map-out') : v('--map-room');
    mctx.fillRect(ox + r.x0 * s, oz + r.z0 * s, (r.x1 - r.x0) * s, (r.z1 - r.z0) * s);
  }
  mctx.strokeStyle = v('--map-wall');
  mctx.lineCap = 'square';
  for (const w of S.walls) {
    if (w.kind === 'lintel') continue;
    mctx.lineWidth = Math.max(1.5, w.t * s * (w.kind === 'sill' ? 0.35 : 1));
    mctx.globalAlpha = w.kind === 'sill' ? 0.6 : 1;
    mctx.beginPath();
    mctx.moveTo(ox + w.x0 * s, oz + w.z0 * s);
    mctx.lineTo(ox + w.x1 * s, oz + w.z1 * s);
    mctx.stroke();
  }
  mctx.globalAlpha = 1;
  mctx.fillStyle = v('--muted');
  mctx.font = `500 ${Math.round(W / 34)}px ${v('--mono')}`;
  mctx.textAlign = 'center';
  mctx.textBaseline = 'middle';
  for (const r of S.rooms) {
    const name = (r.label || '').replace(/^\d['"]?\S*\s+WIDE\s+/i, '').split(/\s+/).slice(0, 2).join(' ');
    if (!name || (r.x1 - r.x0) * s < 34) continue;
    mctx.fillText(name, ox + ((r.x0 + r.x1) / 2) * s, oz + ((r.z0 + r.z1) / 2) * s);
  }
  if (mode === 'walk' || mode === 'tween') {
    const px = ox + player.x * s, pz = oz + player.z * s;
    mctx.save();
    mctx.translate(px, pz);
    mctx.rotate(-player.yaw);
    mctx.fillStyle = v('--accent');
    mctx.beginPath();
    mctx.moveTo(0, -12); mctx.lineTo(7, 7); mctx.lineTo(0, 3); mctx.lineTo(-7, 7);
    mctx.closePath();
    mctx.fill();
    mctx.restore();
  }
}
map.addEventListener('click', (e) => {
  if (!mapXf) return;
  const r = map.getBoundingClientRect();
  const x = (((e.clientX - r.left) / r.width) * map.width - mapXf.ox) / mapXf.s;
  const z = (((e.clientY - r.top) / r.height) * map.height - mapXf.oz) / mapXf.s;
  if (mode === 'walk') {
    [player.x, player.z] = collide(x, z);
    player.vx = player.vz = 0;
  } else if (mode === 'orbit') {
    enterWalk(x, z, player.yaw || S.spawn.yaw);
  }
});

function showSource(on) {
  $('tabPlan').setAttribute('aria-pressed', String(!on));
  $('tabSource').setAttribute('aria-pressed', String(on));
  map.hidden = on;
  $('source').hidden = !on;
}
$('tabPlan').addEventListener('click', () => showSource(false));
$('tabSource').addEventListener('click', () => showSource(true));
const imageUrl = params.get('image') ?? (params.get('plan') ? null : DEFAULT_IMAGE);
if (imageUrl) $('source').src = imageUrl;
else $('tabSource').disabled = true;

// ---------- files ----------
function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
const slug = () => (plan?.meta.title || 'floorplan').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

$('btnGlb').addEventListener('click', () => {
  if (!model) return;
  new GLTFExporter().parse(
    model.root,
    (glb) => { download(new Blob([glb], { type: 'model/gltf-binary' }), `${slug()}.glb`); toast('Exported .glb'); },
    (err) => toast(`GLB export failed: ${err.message}`),
    { binary: true },
  );
});
$('btnScene').addEventListener('click', () => {
  if (!S) return;
  download(new Blob([JSON.stringify(S, null, 1)], { type: 'application/json' }), `${slug()}.scene.json`);
  toast('Exported scene.json');
});
$('btnCopy').addEventListener('click', async () => {
  if (!plan) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(plan, null, 2));
    toast('Plan JSON copied');
  } catch {
    toast('Copy was blocked by the browser');
  }
});

$('btnLoad').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => { if (e.target.files[0]) readPlanFile(e.target.files[0]); e.target.value = ''; });
addEventListener('dragover', (e) => { e.preventDefault(); $('drop').hidden = false; });
addEventListener('dragleave', (e) => { if (!e.relatedTarget) $('drop').hidden = true; });
addEventListener('drop', (e) => {
  e.preventDefault();
  $('drop').hidden = true;
  if (e.dataTransfer.files[0]) readPlanFile(e.dataTransfer.files[0]);
});
async function readPlanFile(file) {
  try {
    const p = JSON.parse(await file.text());
    if (!Array.isArray(p.rooms)) throw new Error('no "rooms" array');
    loadPlan(p);
    $('source').removeAttribute('src');
    $('tabSource').disabled = true;
    showSource(false);
    toast(`Loaded ${file.name}`);
  } catch (err) {
    toast(`${file.name} is not a plan.json: ${err.message}`);
  }
}

let toastTimer;
function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2600);
}

const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, ' ');
const ftIn = (ft) => `${Math.floor(ft)}'${Math.round((ft % 1) * 12)}"`;

// ---------- loop ----------
function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();
applyTheme();

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.05);
  if (mode === 'tween' && tween) {
    tween.t = Math.min(1, tween.t + dt / tween.dur);
    const e = tween.t < 0.5 ? 4 * tween.t ** 3 : 1 - (-2 * tween.t + 2) ** 3 / 2;
    camera.position.lerpVectors(tween.fromPos, tween.toPos, e);
    camera.quaternion.slerpQuaternions(tween.fromQuat, tween.toQuat, e);
    if (tween.t >= 1) { const done = tween.done; tween = null; done(); }
    drawMap();
  } else if (mode === 'walk') {
    stepWalk(dt);
    drawMap();
  } else {
    controls.update();
  }
  renderer.render(scene, camera);
});

loadFromUrl(params.get('plan') || DEFAULT_PLAN);
