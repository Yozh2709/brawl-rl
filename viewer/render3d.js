// 3D view of a Mortis replay (duel or brawl ball) with the game's own models (viewer/models/*.glb, made by tools/sc3d_export.py).
// Like render.js, draw(R, i) is a pure function of (replay, frame): animation time comes from the replay, not a clock.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';

const CHAR_SCALE = 0.1;            // VERIFY: model units -> tiles, times characters.csv Scale / 100
// brawler -> model, characters.csv Scale, clip for the super (Stu's super is a dash: no clip)
const KIT = { Mortis: ['mortis', 1.30, 'ulti'], Stu: ['stu', 1.30, null], Jessie: ['jessie', 1.16, 'attack'] };
const TURRET_SCALE = 1.16;         // characters.csv MechanicTurret
const HEAD = 1.7;                // tiles above the ground for the HP bar
const VIEW_ROWS = 17;              // rows in view, like the game camera; it follows the players along the map
const BALL_D = 0.6;               // VERIFY: ball diameter in tiles (by eye next to Mortis; CSV radius 120 = 0.4 tile is the pickup reach)
const JUMP = 600 * 600;            // squared move per tick above which it's a teleport (respawn, kickoff), as in render.js
// VERIFY: camera pitch, degrees above the ground (by eye: no camera in the exported data); ?tilt=N in the URL to try others
const TILT = +new URLSearchParams(location.search).get('tilt') || 65;
const FADE = 0.15;                 // s: cross-fade between idle / walk / attack clips
const HOLD = 0.45;                 // s: a brawler keeps facing its attack while it walks on (strafes), as in the game
const MR = () => window.MortisRender;
// location_themes.csv: ground + environment + per-tile models; other themes fall back to Hub (the only one exported before)
const THEMES = {
  BBArena: { ground: 'bb_ground', env: 'bb_env', grass: 'bb_grass', wall: 'hub_wall', Crate: 'bb_crate', Barrel: 'bb_barrel' },
  _: { ground: 'hub_ground', env: 'hub_env', grass: 'hub_grass', wall: 'hub_wall' },
};
const MODELS = ['mortis', 'hub_wall', 'hub_grass', 'indestructible', 'hub_ground', 'hub_env', 'spawn_blue', 'spawn_red',
  'bb_crate', 'bb_barrel', 'bb_grass', 'bb_ground', 'bb_env', 'ball', 'stu', 'jessie', 'turret'];

// flat colours until the textures are decoded (tools/sctx.py)
function paint(root, color, shadows = true) {
  root.traverse(o => {
    if (!o.isMesh) return;
    o.castShadow = o.receiveShadow = shadows;
    for (const m of [].concat(o.material)) if (!m.map) {
      if (typeof color === 'function') m.color.set(color(m.name)); else m.color.set(color);
    }
  });
  return root;
}
const hue = s => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7);
const envColor = name => /grass|bush|tree|leaf|plant/i.test(name) ? '#5e9b46' : /water/i.test(name) ? '#4a90c8'
  : `hsl(${hue(name) % 60 + 20}, 22%, ${55 + hue(name) % 15}%)`;

function only(scene, keep) {     // keep one named node (e.g. a grass variant), drop the rest
  const s = scene.clone();
  s.traverse(o => { if (o.isMesh && o.name !== keep && o.parent?.name !== keep) o.visible = false; });
  return s;
}

// every model and sprite sheet as Blobs, in parallel; onLoad(0..1) = bytes so far (sizes not yet known: the average)
async function fetchAll(urls, onLoad) {
  const got = urls.map(() => 0), size = urls.map(() => 0);
  let shown = 0;
  const tick = () => {
    const known = size.filter(Boolean), avg = known.length ? known.reduce((a, b) => a + b) / known.length : 1;
    const all = size.reduce((a, b) => a + (b || avg), 0), have = got.reduce((a, g, i) => a + Math.min(g, size[i] || avg), 0);
    const p = Math.min(0.99, have / all);
    if (p > shown + 0.005) { shown = p; onLoad(p); }
  };
  return Promise.all(urls.map(async (u, i) => {
    const r = await fetch(u);
    if (!r.ok) throw new Error(`${u}: ${r.status}`);
    size[i] = +r.headers.get('content-length') || 1;
    const rd = r.body.getReader(), parts = [];
    for (;;) {
      const { done, value } = await rd.read();
      if (done) break;
      parts.push(value); got[i] += value.length; tick();
    }
    return new Blob(parts);
  }));
}

export async function init(canvas, overlay, redraw, onLoad = () => {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#9cc7e6');
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 400);
  const controls = new OrbitControls(camera, canvas);
  let flip = false;                    // play mode: camera from the top end, so your goal is at the bottom
  controls.enableDamping = false;
  controls.maxPolarAngle = Math.PI * 0.47;
  controls.addEventListener('change', () => redraw && redraw());
  scene.add(new THREE.HemisphereLight('#ffffff', '#7a6648', 1.7));
  const sun = new THREE.DirectionalLight('#fff3dc', 2.4);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -22, right: 22, top: 22, bottom: -22, near: 1, far: 120 });
  sun.shadow.bias = -0.0005;
  scene.add(sun, sun.target);

  const FX = await fetch('fx/fx.json').then(r => r.json()), fxKeys = Object.keys(FX);
  const blobs = await fetchAll([...MODELS.map(n => `models/${n}.glb`), ...fxKeys.map(k => `fx/${k}.png`)], onLoad);
  const loader = new GLTFLoader();
  const G = Object.fromEntries(await Promise.all(MODELS.map(async (n, i) =>
    [n, await loader.parseAsync(await blobs[i].arrayBuffer(), 'models/')])));
  paint(G.hub_wall.scene, '#a07a55');
  paint(G.indestructible.scene, '#6f7480');
  paint(G.hub_grass.scene, '#4f9a3c');
  paint(G.hub_ground.scene, '#d8c28e', false).traverse(o => { if (o.isMesh) o.receiveShadow = true; });
  paint(G.hub_env.scene, envColor);
  paint(G.spawn_blue.scene, '#3b82f6', false);
  paint(G.spawn_red.scene, '#ef4444', false);
  paint(G.mortis.scene, n => /lantern|glow|emis/i.test(n) ? '#ffd27a' : '#5d4a80');
  paint(G.bb_ground.scene, '#8a8f99', false).traverse(o => { if (o.isMesh) o.receiveShadow = true; });
  paint(G.bb_env.scene, envColor);
  paint(G.ball.scene, n => n === 'black' ? '#1f2329' : n === 'seam' ? '#6b7280' : '#f4f4f5');
  const grass = n => [only(G[n].scene, 'grass_1'), only(G[n].scene, 'grass_1b')];
  const grassOf = { hub_grass: grass('hub_grass'), bb_grass: grass('bb_grass') };
  paint(G.stu.scene, '#d9d9d9'); paint(G.jessie.scene, '#d9d9d9'); paint(G.turret.scene, '#d9d9d9');
  // per model: 'idle' / 'walk' / 'attack' / 'ulti' clips ('mortis_walk' -> 'walk')
  const clipsOf = Object.fromEntries(['mortis', 'stu', 'jessie'].map(m =>
    [m, Object.fromEntries(G[m].animations.map(c => [c.name.replace(/^[a-z]+_/, ''), c]))]));

  let mapRoot = null, players = [], plan = null, Rcur = null, otTiles = [], tileAt = new Map(), broken = [];
  // play mode: your aim (game units from you), shown as the game's white range bar; drawn over bushes and walls
  let aimV = null, aimR = null;
  const aimBar = new THREE.Group(), aimMesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: .4, depthTest: false, depthWrite: false }));
  aimMesh.rotation.x = -Math.PI / 2; aimMesh.renderOrder = 3;
  aimBar.add(aimMesh); aimBar.visible = false; scene.add(aimBar);
  // kit objects (turrets, shots, fire / speed zones, thrown turrets): pooled meshes, recoloured every frame
  function pool(make) {
    const items = [];
    return {
      at(k) { while (items.length <= k) { const m = make(); m.traverse(o => { if (o.material) o.material = o.material.clone(); }); scene.add(m); items.push(m); } items[k].visible = true; return items[k]; },
      hideFrom(k) { for (let j = k; j < items.length; j++) items[j].visible = false; },
    };
  }
  const turretPool = pool(() => {
    const g = new THREE.Group(), m = SkeletonUtils.clone(G.turret.scene);
    m.scale.setScalar(CHAR_SCALE * TURRET_SCALE);
    m.traverse(o => { if (o.isMesh) { o.castShadow = true; o.frustumCulled = false; } });
    g.add(m);
    return g;
  });
  // shots, effects, rings, zones, HUD bars: the game's own sprites (blue / red variant), cut from its .sc files by
  // tools/fx_export.py into fx/<key><team>.png sheets. Directed ones lie flat on the ground plane, head along the
  // flight; BILL ones face the camera; all but GROUND ones draw over bushes and walls like the game's 2D layer
  const texLoader = new THREE.TextureLoader();
  const FXT = Object.fromEntries(await Promise.all(fxKeys.map(async (k, i) => {
    const u = URL.createObjectURL(blobs[MODELS.length + i]);
    const t = await texLoader.loadAsync(u); URL.revokeObjectURL(u); t.colorSpace = THREE.SRGBColorSpace; return [k, t];
  })));
  const BILL = new Set(['hit', 'hit_el', 'death', 'spawn', 'kick', 'superkick', 'fire', 'hyper', 'tur_in', 'tur_out', 'throw']);
  const GROUND = new Set(['ring', 'zone', 'spark']); // under the brawlers: depth-tested
  const base = k => k.replace(/[01]$/, '');
  const fxPools = {};
  const fxPool = k => fxPools[k] ||= pool(() => {
    const f = FX[k], t = FXT[k].clone(), g = new THREE.Group(), bill = BILL.has(base(k)), low = GROUND.has(base(k));
    t.repeat.set(1 / f.cols, 1 / f.rows);
    const m = new THREE.Mesh(new THREE.PlaneGeometry(f.w, f.h), new THREE.MeshBasicMaterial({ map: t, transparent: true, depthWrite: false, depthTest: low, toneMapped: false }));
    if (bill) m.position.set(f.cx, -f.cz, 0);
    else { m.rotation.x = -Math.PI / 2; m.position.set(f.cx, 0, f.cz); }
    m.renderOrder = f.shadow ? 5 : low ? 1 : 4;
    g.add(m); g.userData.tex = t; g.userData.bill = bill;
    return g;
  });
  const setFrame = (g, f, k) => {      // sheet cell k, row 0 at the top (textures load flipped)
    const c = k % f.cols, r = Math.floor(k / f.cols);
    g.userData.tex.offset.set(c / f.cols, 1 - (r + 1) / f.rows);
  };
  const SPARK_R = 1300;                // Jessie's spark plug radius (gamedata spark_radius), game units
  const HITS = { hit: 1, shot_hit: 1, turret_hit: 1, debris_hit: 1, spin_hit: 1, bats: 1 };    // events that flash a hit
  const kitOf = (R, p) => KIT[MR().brawler(R, p)] || KIT.Mortis;
  const ball = G.ball.scene.clone();
  const ballMats = [];
  ball.traverse(o => { if (o.isMesh) { o.material = o.material.clone(); ballMats.push(o.material); o.castShadow = true; } });
  const ballFit = new THREE.Box3().setFromObject(ball).getSize(new THREE.Vector3());
  ball.scale.multiplyScalar(BALL_D / Math.max(ballFit.x, ballFit.y, ballFit.z));
  const ballRoll = new THREE.Group();          // rolls (rotation from the position), inside a group that only moves
  ballRoll.add(ball);
  const ballGroup = new THREE.Group();
  ballGroup.add(ballRoll);
  ballGroup.visible = false;
  scene.add(ballGroup);
  window.__m3d = { THREE, scene, camera, controls, G, renderer };     // for poking at it from the console
  const ctx = overlay.getContext('2d');
  let W = 1, H = 1;

  function buildMap(R) {
    if (mapRoot) scene.remove(mapRoot);
    mapRoot = new THREE.Group();
    mapRoot.rotation.x = -Math.PI / 2;           // map models are Z-up, tile (col q, row r) centred at (q, -r)
    const th = THEMES[R.map.theme] || THEMES._;
    mapRoot.add(G[th.ground].scene.clone(), G[th.env].scene.clone());
    const rows = R.map.rows, brk = new Set((R.ot_break || []).map(([r, q]) => r + ',' + q));
    otTiles = []; tileAt = new Map(); broken = [];
    rows.forEach((row, r) => [...row].forEach((ch, q) => {
      const t = R.legend[ch];
      let proto = ch === '1' ? G.spawn_blue.scene : ch === '2' ? G.spawn_red.scene : null;
      if (!proto && t) {
        if (t.name === 'Indestructible') proto = G.indestructible.scene;
        else if (t.name === 'InvisibleIndestructible') proto = null;          // stands / goal frame: the environment model
        else if (t.bush) proto = grassOf[th.grass][(q * 7 + r * 3) % 2];
        else if (th[t.name]) proto = G[th[t.name]].scene;
        else if (t.move && !t.water && t.name !== 'RopeFence') proto = G[th.wall].scene;
      }
      if (!proto) return;
      const m = proto.clone();
      m.position.set(q, -r, 0);
      mapRoot.add(m);
      tileAt.set(q + ',' + r, m);
      if (brk.has(r + ',' + q)) otTiles.push(m);
    }));
    scene.add(mapRoot);
    const cx = rows[0].length / 2 - .5, cz = rows.length / 2 - .5;
    sun.position.set(cx - 12, 40, cz + 18);
    sun.target.position.set(cx, 0, cz);
  }

  function frameCamera(R) {
    const w = R.map.rows[0].length, h = R.map.rows.length, cx = w / 2 - .5, cz = h / 2 - .5;
    const tilt = TILT * Math.PI / 180, half = Math.tan(camera.fov * Math.PI / 360);
    const d = Math.max(Math.min(h, VIEW_ROWS) * Math.sin(tilt) * .56 / half, w * .56 / (half * camera.aspect));
    controls.target.set(cx, 0, cz);
    camera.position.set(cx, Math.sin(tilt) * d, cz + (flip ? -1 : 1) * Math.cos(tilt) * d);
    controls.update();
  }

  // keep the players in view: slide target and camera together along the map (no rotation change)
  function follow(R, F) {          // brawl ball: follow the ball
    const alive = F.players.filter(P => P.hp > 0), h = R.map.rows.length;
    if (!alive.length && !F.ball) return;
    const z = F.ball ? F.ball.y / R.tile - .5 : alive.reduce((s, P) => s + P.y / R.tile - .5, 0) / alive.length;
    const lo = Math.min(h / 2 - .5, VIEW_ROWS / 2 - 2), hi = Math.max(h / 2 - .5, h - VIEW_ROWS / 2 + 1);
    const dz = Math.max(lo, Math.min(hi, z)) - controls.target.z;
    controls.target.z += dz; camera.position.z += dz;
  }

  function makePlayers(R) {
    for (const p of players) scene.remove(p.group);
    players = R.players.map((_, k) => {
      const [kind, scale] = kitOf(R, k), clips = clipsOf[kind];
      const model = SkeletonUtils.clone(G[kind].scene);
      model.scale.setScalar(CHAR_SCALE * scale);
      const mats = [];
      model.traverse(o => { if (o.isMesh) { o.material = o.material.clone(); mats.push(o.material); o.castShadow = true; o.frustumCulled = false; } });
      const mixer = new THREE.AnimationMixer(model);
      const acts = {};
      for (const [name, clip] of Object.entries(clips)) { acts[name] = mixer.clipAction(clip); acts[name].play(); }
      const group = new THREE.Group();
      group.add(model);
      scene.add(group);
      return { group, model, mats, mixer, acts, clips, hidden: false };
    });
  }

  // where an attack / super went (the replay has no aim): the dash's first steps, the new shot, the bats, the throw target
  function attackDir(R, e) {
    const fr = R.frames, n = fr.length, [t, kind, p] = e, at = f => fr[Math.min(n - 1, f)];
    const P = at(t + 1).p[p], name = MR().brawler(R, p);
    if (kind === 'dash' || (kind === 'super' && name === 'Stu')) {
      const Q = at(t + 3).p[p], d = [Q[0] - P[0], Q[1] - P[1]];
      return d[0] * d[0] + d[1] * d[1] > 4 && d[0] * d[0] + d[1] * d[1] < JUMP ? d : null;
    }
    if (kind === 'shot') {
      const tm = MR().teamOf(R, p);
      for (const f of [t + 1, t + 2]) {
        let best = null, bd = 2.5 * R.tile;
        for (const b of at(f).bl || []) if (b[3] === tm) {
          const d = Math.hypot(b[0] - P[0], b[1] - P[1]);
          if (d > 1 && d < bd) { bd = d; best = [b[0] - P[0], b[1] - P[1]]; }
        }
        if (best) return best;
      }
    }
    if (kind === 'super' && name === 'Mortis') { const b = at(t + 2).b[p]; return b[0] ? [b[1] - P[0], b[2] - P[1]] : null; }
    if (kind === 'super' && name === 'Jessie') { const h = (at(t + 1).th || []).find(r => r[0] === p); return h ? [h[3] - h[1], h[4] - h[2]] : null; }
    return null;
  }

  // per player: facing per frame, idle / walk per frame (+ when it began), when attack (dash) / super animations start
  function prep(R) {
    const n = R.frames.length, hold = Math.round(HOLD * 1000 / R.tick_ms);
    return R.players.map((_, p) => {
      const yaw = new Float32Array(n), walk = new Uint8Array(n), since = new Int32Array(n), starts = [];
      const mortis = kitOf(R, p)[0] === 'mortis', turns = new Map();
      for (const e of R.events) if (e[2] === p) {
        if (e[1] === 'super' && kitOf(R, p)[2]) starts.push([e[0] + 1, kitOf(R, p)[2]]);
        else if (e[1] === 'shot') starts.push([e[0] + 1, 'attack']);
        if (/^(shot|dash|super)$/.test(e[1])) { const d = attackDir(R, e); if (d) turns.set(e[0] + 1, d); }
      }
      const moved = new Uint8Array(n);
      let y = MR().teamOf(R, p) === 0 ? Math.PI : 0, until = -1;   // team 0 starts at the bottom, facing up
      for (let f = 0; f < n; f++) {
        const a = R.frames[f].p[p], g = f + 1 < n ? f + 1 : f, b = R.frames[g].p[p], c = R.frames[Math.max(0, g - 1)].p[p];
        const dx = b[0] - c[0], dz = b[1] - c[1], d2 = dx * dx + dz * dz;   // the last frame: the step into it
        moved[f] = d2 > 4 && d2 < JUMP;
        const t = turns.get(f);
        if (t) { y = Math.atan2(t[0], t[1]); until = f + hold; }       // faces the attack, then walks on sideways
        else if (moved[f] && f > until) y = Math.atan2(dx, dz);
        yaw[f] = y;
        if (mortis && a[7] && (f === 0 || !R.frames[f - 1].p[p][7])) starts.push([f, 'attack']);   // Mortis attacks by dashing
      }
      for (let f = 0; f < n; f++) {        // a one-tick stop (a bump, a wall) doesn't flip to idle and back
        walk[f] = moved[f] || (f > 0 && moved[f - 1] && f + 1 < n && moved[f + 1]);
        since[f] = f > 0 && walk[f] === walk[f - 1] ? since[f - 1] : f;
      }
      starts.sort((u, v) => u[0] - v[0]);
      return { yaw, walk, since, starts };
    });
  }

  function lerpAngle(a, b, t) { let d = b - a; d -= Math.round(d / (2 * Math.PI)) * 2 * Math.PI; return a + d * t; }

  // clip weights from the replay time alone: walk / idle fade into each other, an attack fades in over them and out
  function pose(pl, pp, R, i, P) {
    const dt = R.tick_ms / 1000, n = pp.yaw.length - 1, f0 = Math.min(n, Math.floor(i)), clips = pl.clips;
    let act = null, at = 0;
    for (const [f, name] of pp.starts) {
      if (f > i) break;
      const t = (i - f) * dt;
      if (clips[name] && t < clips[name].duration) { act = name; at = t; }
    }
    const wa = act ? Math.max(0, Math.min(1, at / (FADE / 2), (clips[act].duration - at) / FADE)) : 0;
    const cur = pp.walk[f0] ? 'walk' : 'idle', old = pp.walk[f0] ? 'idle' : 'walk', s0 = pp.since[f0];
    const wc = Math.min(1, (i - s0) * dt / FADE), s1 = s0 > 0 ? pp.since[s0 - 1] : 0;
    const w = { [cur]: (1 - wa) * wc, [old]: (1 - wa) * (1 - wc) };
    const tm = { [cur]: (i - s0) * dt % clips[cur].duration, [old]: (i - s1) * dt % clips[old].duration };
    if (act) { w[act] = wa; tm[act] = at; }
    for (const [name, a] of Object.entries(pl.acts)) {
      a.enabled = true;
      a.setEffectiveWeight(w[name] || 0);
      a.time = tm[name] || 0;
    }
    pl.mixer.update(0);
    pl.model.rotation.y = lerpAngle(pp.yaw[f0], pp.yaw[Math.min(n, f0 + 1)], i - f0);
    if (pl.hidden !== !!P.hidden) {
      pl.hidden = !!P.hidden;
      for (const m of pl.mats) { m.transparent = pl.hidden; m.opacity = pl.hidden ? .45 : 1; m.needsUpdate = true; }
    }
  }

  function project(x, y, z) {
    const v = new THREE.Vector3(x, y, z).project(camera);
    return [(v.x + 1) / 2 * W, (1 - v.y) / 2 * H, v.z < 1];
  }

  // game-style HUD: Lilita One (the game's font), black-outlined text, rounded bars in a black frame
  const FONT = s => `${s}px "Lilita One", system-ui, sans-serif`;
  function outline(s, x, y, size, fill) {
    ctx.font = FONT(size); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.lineJoin = 'round';
    ctx.lineWidth = Math.max(2.5, size * .22); ctx.strokeStyle = '#000'; ctx.strokeText(s, x, y);
    ctx.fillStyle = fill; ctx.fillText(s, x, y);
  }
  // the game's own HUD parts (ui.sc enemy_info_badge, fx/hp_bg, hp<team>, ammo): sheet cell `frame` centred
  // at (x, y), k screen px per game px. The bar clips run 100 frames: hp 0 = full, ammo 99 = loaded
  function part(key, frame, x, y, k) {
    const f = FX[key], img = FXT[key].image, cw = img.width / f.cols, ch = img.height / f.rows;
    ctx.drawImage(img, frame % f.cols * cw, Math.floor(frame / f.cols) * ch, cw, ch, x - f.w * k / 2, y - f.h * k / 2, f.w * k, f.h * k);
  }
  function hpBar(x, y, k, v, tm) {
    part('hp_bg', 0, x, y, k);
    part('hp' + tm, Math.round((1 - Math.max(0, Math.min(1, v))) * 99), x, y, k);
  }
  const AMMO_PITCH = 36.8;             // game px between ammo cells (reload_own: 3 cells over 110.5 px)

  function hud(R, F) {
    const dpr = Math.min(2, devicePixelRatio);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    // the game sizes its HUD to the screen, so the badges keep their size against the brawlers: scale by
    // on-screen px per tile (the bar is ~0.85 tile wide), drawn in u-scaled units
    const u = Math.min(1.25, Math.max(.7, (project(11, 0, 16)[0] - project(10, 0, 16)[0]) / 70));
    ctx.setTransform(dpr * u, 0, 0, dpr * u, 0, 0);
    const pj = (a, b, c) => { const [x, y, ok] = project(a, b, c); return [x / u, y / u, ok]; };
    const wx = P => P.x / R.tile - .5, wz = P => P.y / R.tile - .5;
    F.players.forEach((P, p) => {
      if (P.hp <= 0) {
        if (!(P.resp > 0)) return;               // brawl ball: respawn countdown where he fell
        const [x, y, ok] = pj(wx(P), .3, wz(P));
        if (!ok) return;
        outline(`${MR().tag(R, p)} ${Math.ceil(P.resp * R.tick_ms / 1000)}`, x, y, 16, MR().color(R, p));
        return;
      }
      const [x, y, ok] = pj(wx(P), HEAD, wz(P));
      if (!ok) return;
      // the game's badge: name above the bar, HP number on the bar, ammo right under it; super / hyper / long
      // dash live in the side panel, not over the head
      const bw = 60, bk = bw / 96, by = y - 20;
      ctx.globalAlpha = P.hidden ? .6 : 1;
      hpBar(x, by, bk, Math.max(0, P.hp) / MR().maxHp(R, p), MR().teamOf(R, p));
      outline(Math.round(P.hp), x, by + 7, 14, '#fff');
      if (R.mode === 'ball') outline(MR().tag(R, p), x, by - 8, 11, MR().color(R, p));
      const n = MR().maxAmmo(R, p), ak = Math.min(bk * .6 / .7, bw / (n * AMMO_PITCH)), ay = by + 16.4 * bk / .7;
      for (let k = 0; k < n; k++)
        part('ammo', Math.round(Math.max(0, Math.min(1, P.ammo - k)) * 99), x + (k - (n - 1) / 2) * AMMO_PITCH * ak, ay, ak);
      ctx.globalAlpha = 1;
    });
    // turret HP bars
    for (const [o, tx, ty, hp, mx] of F.tu) {
      const [x, y, ok] = pj(tx / R.tile - .5, .9, ty / R.tile - .5);
      if (!ok) continue;
      hpBar(x, y - 2, .45, Math.max(0, hp) / mx, MR().teamOf(R, o));
      outline(Math.round(hp), x, y + 4, 11, '#fff');
    }
    // damage / heal numbers (an event logged in tick t shows from frame t + 1); one number per target, kind and
    // tick, so a spread of pellets or a super's hits read as one hit like in the game
    const nums = new Map();
    for (const [et, kind, a, b, amt] of R.events) {
      if (!MR().NUM[kind] || !amt) continue;
      const age = F.t - et - 1;
      if (age < 0 || age >= 14) continue;
      const key = `${et} ${b} ${kind}`;
      nums.set(key, [et, kind, b, (nums.get(key)?.[3] || 0) + amt]);
    }
    for (const [et, kind, b, amt] of nums.values()) {
      const age = F.t - et - 1;
      const at = MR().unitAt(R, R.frames[Math.min(R.frames.length - 1, et + 1)], b);
      if (!at) continue;
      const [x, y0, ok] = pj(at[0] / R.tile - .5, b < R.players.length ? HEAD : 1, at[1] / R.tile - .5);
      const y = y0 - 34 - age * 1.2;
      if (!ok) continue;
      const s = (kind === 'heal' ? '+' : '-') + Math.round(amt), dx = kind === 'heal' ? 30 : kind === 'burn' ? -30 : 0;
      ctx.globalAlpha = 1 - age / 14;
      outline(s, x + dx, y, 16, MR().NUM[kind]);
      ctx.globalAlpha = 1;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (F.score) {
      const s = 22;
      MR().drawScore(ctx, R, F, W / 2 - s * 3.2, 10, s);
      MR().banner(ctx, R, F, W / 2, H * .4, 26);
    }
  }

  // free ball on the ground, a carried one at the carrier's feet in front of him; it rolls with the position
  function placeBall(R, F) {
    const B = F.ball;
    ballGroup.visible = !!B;
    if (!B) return;
    let x = B.x / R.tile - .5, z = B.y / R.tile - .5, faded = false;
    if (B.carrier >= 0) {
      const pl = players[B.carrier], a = pl.model.rotation.y;
      x = pl.group.position.x + Math.sin(a) * .42; z = pl.group.position.z + Math.cos(a) * .42;
      faded = !!F.players[B.carrier].hidden;
    }
    const r = BALL_D / 2;
    ballGroup.position.set(x, r, z);
    ballRoll.rotation.set(z / r, 0, -x / r);
    for (const m of ballMats) if (m.transparent !== faded) { m.transparent = faded; m.opacity = faded ? .45 : 1; m.needsUpdate = true; }
  }

  return {
    setReplay(R) {
      const sameMap = Rcur && Rcur.map.name === R.map.name && Rcur.map.rows.join() === R.map.rows.join();
      Rcur = R;                       // first: frameCamera fires a controls 'change' -> redraw -> draw(R)
      makePlayers(R);
      plan = prep(R);
      if (!sameMap) { buildMap(R); frameCamera(R); }
    },
    refresh(R) { Rcur = R; plan = prep(R); },     // the same replay grew (live match)
    resize(w, h) {
      W = Math.max(1, w); H = Math.max(1, h);
      renderer.setSize(W, H, false);
      const dpr = Math.min(2, devicePixelRatio);
      overlay.width = W * dpr; overlay.height = H * dpr;
      camera.aspect = W / H; camera.updateProjectionMatrix();
      if (Rcur) frameCamera(Rcur);    // the fit depends on the aspect
    },
    resetCamera() { if (Rcur) frameCamera(Rcur); },
    // play mode: the game's fixed camera (clicks must not orbit it), turned 180° when your goal is at the top
    playView(f) { controls.enabled = false; if (f !== flip) { flip = f; if (Rcur) frameCamera(Rcur); } },
    setAim(R, v) { aimR = R; aimV = v && (v[0] || v[1]) ? v : null; },
    toWorld(x, y, R) {                // a screen point -> game units on the ground
      const b = canvas.getBoundingClientRect(), ray = new THREE.Raycaster(), p = new THREE.Vector3();
      ray.setFromCamera(new THREE.Vector2((x - b.left) / b.width * 2 - 1, 1 - (y - b.top) / b.height * 2), camera);
      return ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), p) ? [(p.x + .5) * R.tile, (p.z + .5) * R.tile] : null;
    },
    draw(R, i) {
      if (R !== Rcur) this.setReplay(R);
      else if (plan[0] && plan[0].yaw.length !== R.frames.length) plan = prep(R);   // play mode: frames arrive live
      const F = window.MortisRender.unpack(R, i);
      follow(R, F);
      F.players.forEach((P, p) => {
        const pl = players[p];
        pl.group.visible = (P.hp > 0 || i === 0) && !(P.fx & 4);   // fx 4 = flying as bats (sprite below)
        pl.group.position.set(P.x / R.tile - .5, 0, P.y / R.tile - .5);
        pose(pl, plan[p], R, i, P);
      });
      placeBall(R, F);
      const me = F.players[R.me];
      aimBar.visible = !!(aimV && R === aimR && me && me.hp > 0);
      if (aimBar.visible) {                  // with the ball: the kick's reach, else the attack dash
        const len = (F.ball && F.ball.carrier === R.me ? R.consts.kick_range : R.consts.dash_range) / R.tile;
        aimBar.position.set(me.x / R.tile - .5, .05, me.y / R.tile - .5);
        aimBar.rotation.y = Math.atan2(aimV[0], aimV[1]);
        aimMesh.scale.set(2 * R.consts.radius / R.tile, len, 1); aimMesh.position.z = len / 2;
      }
      for (const m of otTiles) m.visible = !F.ot;
      for (const m of broken) if (!otTiles.includes(m) || !F.ot) m.visible = true;
      broken = F.brk.map(([q, r]) => tileAt.get(q + ',' + r)).filter(Boolean);
      for (const m of broken) m.visible = false;
      const w = v => v / R.tile - .5;
      F.tu.forEach(([o, x, y], j) => {
        const m = turretPool.at(j); m.position.set(w(x), 0, w(y));
        m.rotation.y = MR().teamOf(R, o) === 0 ? Math.PI : 0;   // faces the enemy side, like a fresh player
      });
      turretPool.hideFrom(F.tu.length);
      // shots carry no id: match each to the nearest one of its kind in the next / previous frame to glide and aim
      const fr = R.frames, i0 = Math.floor(i), fa = i - i0;
      const nxt = fr[Math.min(fr.length - 1, i0 + 1)].bl || [], prv = fr[Math.max(0, i0 - 1)].bl || [];
      const near = (list, x, y, kind) => {
        let best = null, bd = 1.5 * R.tile;
        for (const b of list) { const d = Math.hypot(b[0] - x, b[1] - y); if (b[2] === kind && d < bd) { bd = d; best = b; } }
        return best;
      };
      const used = {}, take = k => fxPool(k).at((used[k] = (used[k] || 0) + 1) - 1);
      const sec = i * R.tick_ms / 1000;
      F.bl.forEach(([x, y, kind, tm]) => {
        const key = kind + (tm ? 1 : 0), f = FX[key];
        if (!f) return;
        const g = near(nxt, x, y, kind), p = i0 > 0 ? near(prv, x, y, kind) : null;
        const cx = g ? x + (g[0] - x) * fa : x, cy = g ? y + (g[1] - y) * fa : y;
        const [dx, dy] = g && (g[0] !== x || g[1] !== y) ? [g[0] - x, g[1] - y] : p ? [x - p[0], y - p[1]] : [0, tm ? 1 : -1];
        const m = take(key);
        m.position.set(w(cx), .6, w(cy)); m.rotation.y = Math.PI - Math.atan2(dy, dx);   // sprites point to -x
        setFrame(m, f, Math.floor(sec * f.rate) % f.n);
        if (f.shadow) take(f.shadow).position.set(w(cx), .03, w(cy));
      });
      // any other sprite: key, ground spot, sheet cell; o = { y, yaw (flat ones), s | sx sy sz, op }
      const spr = (key, x, z, frame, o = {}) => {
        const f = FX[key], m = take(key);
        m.position.set(x, o.y ?? .6, z);
        if (m.userData.bill) m.quaternion.copy(camera.quaternion); else m.rotation.set(0, o.yaw || 0, 0);
        m.scale.set(o.sx ?? o.s ?? 1, o.sy ?? o.s ?? 1, o.sz ?? o.s ?? 1);
        setFrame(m, f, Math.max(0, Math.min(f.n - 1, Math.floor(frame))));
        m.children[0].material.opacity = o.op ?? 1;
      };
      const loop = (key, ph = 0) => (sec + ph) * FX[key].rate % FX[key].n;
      const tk = (k, u) => k + (FX[k + MR().teamOf(R, u)] ? MR().teamOf(R, u) : 0);     // blue / red variant
      const aim = (dx, dy) => Math.PI - Math.atan2(dy, dx);                                // flat sprites point to -x
      const face = p => { const a = players[p].model.rotation.y; return aim(Math.sin(a), Math.cos(a)); };
      F.players.forEach((P, p) => {
        if (P.hp <= 0) return;
        const x = w(P.x), z = w(P.y), op = P.hidden ? .5 : 1;
        if (P.fx & 4) spr(tk('creature', p), x, z, loop('creature0'), { y: .9, yaw: face(p), s: .7, op });
        else spr(tk('ring', p), x, z, loop('ring0'), { y: .03, op });
        if (P.hon > 0) for (let j = 0; j < 3; j++) {           // hypercharged: purple flames round the feet
          const a = sec * 2 + j * 2.09;
          spr('hyper0', x + Math.cos(a) * .35, z + Math.sin(a) * .35, loop('hyper0', j * .21), { y: .3, s: .5, op });
        }
      });
      for (const [o, x, y] of F.tu) spr(tk('ring', o), w(x), w(y), loop('ring0'), { y: .03, s: .8 });
      F.bats.forEach((b, p) => {                               // Mortis super: the bat swarm, along its flight
        if (!b.on) return;
        const r0 = fr[i0].b[p], r1 = fr[Math.min(fr.length - 1, i0 + 1)].b[p], rp = fr[Math.max(0, i0 - 1)].b[p];
        const [dx, dy] = r1[0] && (r1[1] !== r0[1] || r1[2] !== r0[2]) ? [r1[1] - r0[1], r1[2] - r0[2]] : rp[0] ? [r0[1] - rp[1], r0[2] - rp[2]] : [0, 1];
        const key = tk(F.players[p].hon > 0 ? 'batsh' : 'bats', p);
        spr(key, w(b.x), w(b.y), loop(key), { yaw: aim(dx, dy), s: 2 * R.consts.super_radius / R.tile / FX[key].w });
      });
      F.th.forEach(([o, x, y, tx, ty]) => {                    // thrown turret: an arc from the hand to the landing spot
        const d = Math.hypot(tx - x, ty - y) / R.tile;
        spr('throw0', w(x), w(y), loop('throw0'), { y: .3 + Math.min(2, d * .5) });
        spr('shadow_normal_circle', w(x), w(y), 0, { y: .03 });
      });
      F.ar.forEach(([kind, x, y, r, tm]) => {
        const rr = r / R.tile;
        if (kind === 'zone') spr('zone' + tm, w(x), w(y), 0, { y: .02, sx: 2 * rr / FX.zone0.w, sz: 2 * rr / FX.zone0.h });
        else {                                                 // Stu's fire trail: flames spread over the patch
          const n = Math.min(12, Math.max(3, Math.round(3.2 * rr * rr)));
          for (let j = 0; j < n; j++) {
            const q = rr * .75 * Math.sqrt((j + .5) / n), a = j * 2.4;
            spr('fire0', w(x) + Math.cos(a) * q, w(y) + Math.sin(a) * q, loop('fire0', j * .37), { y: 0, s: .6 });
          }
        }
      });
      // one-shot effects from the event log (an event logged in tick t shows from frame t + 1)
      const at = (t, u) => MR().unitAt(R, fr[Math.min(fr.length - 1, t + 1)], u) || MR().unitAt(R, fr[t], u);
      for (const [et, kind, a, b] of R.events) {
        const age = (i - et - 1) * R.tick_ms / 1000;
        if (age < 0 || age > 2.5) continue;
        const once = (key, x, z, o) => { const k = age * FX[key].rate; if (k < FX[key].n) spr(key, x, z, k, o); };
        if (kind in HITS) {
          const u = at(et, b);
          if (u) once(kind === 'shot_hit' && MR().brawler(R, a) === 'Jessie' ? 'hit_el0' : 'hit0', w(u[0]), w(u[1]), {});
        } else if (kind === 'death') {
          const u = fr[et].p[a]; once(tk('death', a), w(u[0]), w(u[1]), { y: .5 });
        } else if (kind === 'respawn') {
          const P = F.players[a]; once('spawn0', w(P.x), w(P.y), { y: .6, op: 1 - age * 1.5 });
        } else if (kind === 'turret' || kind === 'turret_down') {
          const u = kind === 'turret' ? at(et, b) : MR().unitAt(R, fr[et], b);
          if (u) once(kind === 'turret' ? tk('tur_in', a) : 'tur_out0', w(u[0]), w(u[1]), { y: 0, s: .5 });
        } else if (kind === 'kick' || kind === 'superkick') {
          const B = fr[Math.min(fr.length - 1, et + 1)].ball;
          if (B) once(kind + '0', w(B[0]), w(B[1]), { y: .3, s: .8 });
        } else if (kind === 'dash') {
          const P = F.players[a]; once('dash0', w(P.x), w(P.y), { y: .5, yaw: face(a), s: .8 });
        } else if (kind === 'shot') {
          const u = fr[Math.min(fr.length - 1, et + 1)].p[a], y = face(a), fx = w(u[0]) - Math.cos(y) * .6, fz = w(u[1]) + Math.sin(y) * .6;
          if (MR().brawler(R, a) === 'Stu') once(tk('stu_atk', a), fx, fz, { yaw: y });
          else if (MR().brawler(R, a) === 'Jessie') once('hit_el0', fx, fz, { s: .7 });
        } else if (kind === 'gadget') {
          const g = R.players[a].gadget, P = F.players[a];
          if (g === 'combo_spinner') once(tk('spinner', a), w(P.x), w(P.y), { y: .4 });
          else if (g === 'spark_plug') {
            const t = F.tu.find(r => r[0] === a) || [a, P.x, P.y], key = tk('spark', a);
            once(key, w(t[1]), w(t[2]), { y: .05, s: 2 * SPARK_R / R.tile / (FX[key].w * .9) });
          }
        }
      }
      for (const k in fxPools) fxPools[k].hideFrom(used[k] || 0);
      renderer.render(scene, camera);
      hud(R, F);
    },
  };
}

window.MortisRender3D = { init };
dispatchEvent(new Event('mortis3d'));
