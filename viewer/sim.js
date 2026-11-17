// Brawl Ball in the browser, no server: a port of rl/jball.py (the sim: Mortis, and with G.multi Stu and Jessie too),
// the parts of rl/jballenv.py the net needs (vision memory, obs, button masks, action -> inputs; no bots, no trickshot
// hint) and the policy MLP. tools/web_export.py writes its data (web/game.json, web/net*.bin); tools/web_parity.py
// checks it against JAX. LocalPlay mirrors rl/jplay.py's Game, so play.js talks to it like to the server.
(function (root) {
  'use strict';
  const NEIGH = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
  const clip = (x, a, b) => Math.min(Math.max(x, a), b);
  const nrm = (x, y) => Math.sqrt(x * x + y * y);
  const dist = (a, b) => nrm(a[0] - b[0], a[1] - b[1]);
  const unit = v => { const n = nrm(v[0], v[1]); return n > 1e-9 ? [v[0] / n, v[1] / n] : [0, 0]; };
  const fill = (n, f) => Array.from({ length: n }, (_, i) => f(i));
  const roundEven = x => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 ? r - 1 : r; };   // jnp.round

  function segDist(q, a, b) {
    const abx = b[0] - a[0], aby = b[1] - a[1], L = abx * abx + aby * aby;
    const t = L > 1e-9 ? clip(((q[0] - a[0]) * abx + (q[1] - a[1]) * aby) / Math.max(L, 1e-9), 0, 1) : 0;
    return nrm(q[0] - (a[0] + abx * t), q[1] - (a[1] + aby * t));
  }

  function Sim(G) {
    const c = G.c, K = G.k, X = G.x || {}, P = G.P, H = G.H, W = G.W, TILE = c.TILE, R = c.R, team = G.team;
    const mt = !!G.multi, U = mt ? 2 * P : P, KB = mt ? X.KB : 0, NB = P * KB, FZ = mt ? X.FZ : 0;
    const ar = fill(P, i => i), uteam = fill(U, u => team[u % P]);    // unit P + o: player o's turret
    const foe = fill(P, a => fill(P, b => team[a] !== team[b])), ufoe = fill(P, a => fill(U, u => team[a] !== uteam[u]));
    const tile = (g, x, y) => x < 0 || x >= W || y < 0 || y >= H || !!g[y * W + x];
    const cell = v => Math.floor(v / TILE);
    const grids = s => { const o = s.ot ? 1 : 0; return [G.block[o], G.shot[o], G.bush[o]]; };
    const inBush = (bush, p) => !!bush[clip(cell(p[1]), 0, H - 1) * W + clip(cell(p[0]), 0, W - 1)];
    const solid = (pt, shot) => tile(shot, cell(pt[0]), cell(pt[1]));
    const maxHp = (s, p) => K.HP[s.kind[p]];
    const longReady = (s, p) => s.t - s.last_attack[p] >= c.LONG_TICKS && s.ammo[p] >= c.AMMO && s.kind[p] === 0;
    const ahead = o => [0, team[o] === 0 ? -1 : 1];                   // a zero direction shoots / throws straight ahead

    function push(p, block) {                  // one circle out of blocked tiles / the map border (Ball.push)
      let x = p[0], y = p[1];
      for (let it = 0; it < 2; it++) {
        const cx = cell(x), cy = cell(y);
        for (const [dx, dy] of NEIGH) {
          const tx = cx + dx, ty = cy + dy, x0 = tx * TILE, y0 = ty * TILE;
          const ddx = x - clip(x, x0, x0 + TILE), ddy = y - clip(y, y0, y0 + TILE), d = Math.hypot(ddx, ddy);
          if (!(tile(block, tx, ty) && d < R)) continue;
          if (d > 1e-6) { const k = (R - d) / Math.max(d, 1e-6); x += ddx * k; y += ddy * k; continue; }
          const gaps = [x - x0, x0 + TILE - x, y - y0, y0 + TILE - y];
          let side = 0;
          for (let i = 1; i < 4; i++) if (gaps[i] < gaps[side]) side = i;
          const nx = side === 0 ? x0 - R : side === 1 ? x0 + TILE + R : x;
          y = side === 2 ? y0 - R : side === 3 ? y0 + TILE + R : y;
          x = nx;
        }
      }
      return [x, y];
    }

    function fly(bp, bd, L, shot) {           // a free ball: two half steps, each axis bounces on its own
      bp = bp.slice(); bd = bd.slice();
      for (let h = 0; h < 2; h++) for (let ax = 0; ax < 2; ax++) {
        const nx = bp.slice();
        nx[ax] += bd[ax] * L / 2;
        if (solid(nx, shot)) bd[ax] = -bd[ax]; else bp = nx;
      }
      return [bp, bd];
    }

    function clearShot(a, b, shot, kmax = c.SPIN_PTS) {
      const k = Math.max(2, Math.floor(dist(b, a) / 50) + 1);
      for (let i = 0; i < kmax; i++) {
        const f = Math.min(i, k - 1) / (k - 1);
        if (tile(shot, cell(a[0] + (b[0] - a[0]) * f), cell(a[1] + (b[1] - a[1]) * f))) return false;
      }
      return true;
    }

    function vis(pos, alive, reveal, t, bush) {   // v[viewer][target], shared team vision (Ball._vis)
      const shown = pos.map((p, q) => !inBush(bush, p) || t < reveal[q] || !alive[q]);
      const v = fill(P, a => fill(P, q => (shown[q] || dist(pos[a], pos[q]) <= c.BUSH_REVEAL) && alive[a]));
      return fill(P, p => fill(P, q => p === q || G.mate[p].some((m, j) => m && v[j][q])));
    }
    const visible = s => vis(s.pos, s.alive, s.reveal_until, s.t, grids(s)[2]);

    function units(s, pos) {                  // positions and hittable of brawlers (+ turrets)
      return mt ? [pos.concat(s.tur_pos), s.alive.concat(s.tur_on)] : [pos, s.alive];
    }

    // kind (P,) indices into G.kinds (none = all Mortis), n (2,) players a side (none = full teams)
    function reset(kind, n) {
      kind = kind ? kind.slice() : fill(P, () => 0); n = n || [G.T, G.T];
      const z = () => fill(P, () => 0), zb = () => fill(P, () => false), z2 = () => fill(P, () => [0, 0]);
      const zu = () => fill(P, () => fill(U, () => false)), nt = ar.map(p => n[team[p]]);
      const here = ar.map(p => p % G.T < nt[p]), spawn = ar.map(p => G.spawn_tab[nt[p] - 1][p].slice());
      const zn = () => fill(NB, () => 0);
      return {
        t: 0, done: false, winner: -2, kind, here, spawn, respawn: ar.map(p => G.respawn_tab[nt[p] - 1][p].slice()),
        pos: spawn.map(p => p.slice()), vel: z2(), hp: ar.map(p => K.HP[kind[p]] * here[p]), alive: here.slice(),
        ammo: fill(P, () => c.AMMO), reload: z(), last_attack: z(), last_hurt: fill(P, () => -1e6),
        last_by: fill(U, () => -1), ulti: z(), reveal_until: z(), atk_ready: z(), burst: z(), burst_next: z(),
        burst_dir: z2(), dash_rem: z(), dash_dir: z2(), dash_hit: zu(), proj_on: zb(), proj_pos: z2(), proj_dir: z2(),
        proj_rem: z(), proj_hit: zu(), proj_back: zb(), proj_ret: zb(), hyper: z(), hyper_until: z(),
        gad_left: fill(P, () => c.GADGET_USES), gad_ready: z(), slow_until: z(), zone_pos: z2(), zone_until: fill(P, () => -1),
        tur_on: zb(), tur_pos: z2(), tur_hp: z(), tur_max: z(), tur_dmg: z(), tur_next: z(), th_on: zb(), th_a: z2(),
        th_b: z2(), th_t0: z(), th_t1: z(), th_hyper: zb(),
        bul: { on: fill(NB, () => false), pos: fill(NB, () => [0, 0]), dir: fill(NB, () => [0, 0]), rem: zn(), step: zn(),
               r: zn(), dmg: zn(), ch: zn(), kind: zn(), chain: zn(), hit: fill(NB, () => fill(U, () => false)), seq: zn() },
        bseq: 0, fz_pos: fill(P, () => fill(FZ, () => [0, 0])), fz_until: fill(P, () => fill(FZ, () => 0)),
        burn_until: fill(U, () => 0), burn_next: fill(U, () => 0), burn_src: fill(U, () => -1), respawn_at: fill(P, () => -1),
        score: [0, 0], ot: false, freeze_until: 0, ball_pos: G.center.slice(), ball_dir: [0, 0], ball_rem: 0, carrier: -1,
        last_touch: -1, kicker: -1, nopick_until: 0,
      };
    }

    function revive(s, m, at) {               // alive at `at` with full HP and ammo; super charge is kept
      for (let p = 0; p < P; p++) if (m[p]) {
        s.pos[p] = at[p].slice(); s.hp[p] = maxHp(s, p); s.alive[p] = true; s.ammo[p] = c.AMMO; s.reload[p] = 0;
        s.dash_rem[p] = 0; s.burst[p] = 0; s.respawn_at[p] = -1; s.reveal_until[p] = 0; s.last_hurt[p] = -1e6;
      }
    }

    function credit(lastBy, hit) {            // hit [attacker][unit] -> last_by: the highest attacker wins
      return lastBy.map((v, u) => { let top = -1; for (let a = 0; a < P; a++) if (hit[a][u]) top = a; return top >= 0 ? top : v; });
    }

    // new bullets (Ball._spawn): rows {want, order, own, pos, d, kind, rem, step, r, dmg, ch, chain, hit}, each into a
    // free slot of its owner's slice in `order` (a full slice drops it), numbered in creation order -> bullets made
    function spawn(b, bseq, rows) {
      const free = fill(P, o => fill(KB, i => o * KB + i).filter(j => !b.on[j])), taken = fill(P, () => 0);
      let made = 0;
      for (const r of rows.filter(r => r.want).sort((x, y) => x.order - y.order)) {
        const k = taken[r.own]++;
        if (k >= free[r.own].length) continue;
        const j = free[r.own][k], L = nrm(r.d[0], r.d[1]);
        b.on[j] = true; b.pos[j] = r.pos.slice(); b.dir[j] = L > 1e-9 ? [r.d[0] / Math.max(L, 1e-9), r.d[1] / Math.max(L, 1e-9)] : ahead(r.own);
        b.rem[j] = r.rem; b.step[j] = r.step; b.r[j] = r.r; b.dmg[j] = r.dmg; b.ch[j] = r.ch; b.kind[j] = r.kind;
        b.chain[j] = r.chain; b.hit[j] = r.hit.slice(); b.seq[j] = bseq + made++;
      }
      return bseq + made;
    }

    // one tick of every bullet (Ball._fly_bullets): a wall stops it, the first unit on the path is hit, Jessie's orb
    // then jumps to the nearest unhit enemy in reach (from the next tick). -> turret heals (P,), chain jump rows
    function flyBullets(b, act, upos, uok, tur_on, shot, mul, dmg, gain, last_by, E) {
      const heals = fill(P, () => 0), jumps = [], top = fill(U, () => -1), by = fill(U, () => -1);
      for (let j = 0; j < NB; j++) {
        if (!(b.on[j] && act)) continue;
        const o = Math.floor(j / KB), a = b.pos[j], dv = b.dir[j], L = Math.min(b.rem[j], b.step[j]);
        const mid = [a[0] + dv[0] * (L / 2), a[1] + dv[1] * (L / 2)];
        let end = [a[0] + dv[0] * L, a[1] + dv[1] * L];
        const wm = solid(mid, shot), we = solid(end, shot);
        if (wm) end = mid;
        let u = -1, bd = Infinity;
        for (let v = 0; v < U; v++) {
          const enemy = v === P + o && b.kind[j] === 1 ? tur_on[o] : uok[v] && uteam[v] !== team[o];   // Energize
          if (!enemy || b.hit[j][v] || !(segDist(upos[v], a, end) <= b.r[j] + R)) continue;
          const da = dist(upos[v], a);
          if (da < bd) { bd = da; u = v; }
        }
        if (u < 0) {
          b.rem[j] -= L; b.pos[j] = end; b.on[j] = !(wm || we) && b.rem[j] > 0;
          if (!b.on[j] && !b.hit[j].some(Boolean)) E('miss', o, b.kind[j] === 2 ? P + o : -1, 0);
          continue;
        }
        if (u === P + o) { heals[o]++; E('turret_heal', o, u, 0); }
        else {
          const amt = b.dmg[j] * mul[o][u];
          dmg[u] += amt; gain[o] += b.ch[j]; E(b.kind[j] === 2 ? 'turret_hit' : 'shot_hit', o, u, amt);
          if (b.seq[j] > top[u]) { top[u] = b.seq[j]; by[u] = o; }       // the latest bullet wins
        }
        const hit2 = b.hit[j].slice(), src = upos[u];
        hit2[u] = true;
        let v = -1, vd = Infinity;
        for (let w = 0; w < U; w++) {
          if (!(uok[w] && uteam[w] !== team[o]) || hit2[w]) continue;
          const dd = dist(upos[w], src);
          if (dd <= X.CHAIN_DIST && dd < vd) { vd = dd; v = w; }
        }
        if (b.chain[j] > 0 && v >= 0) jumps.push({ want: true, order: b.seq[j], own: o, pos: src.slice(), d: [upos[v][0] - src[0], upos[v][1] - src[1]],
          kind: b.kind[j], rem: X.CHAIN_DIST, step: X.CHAIN_STEP, r: b.r[j], dmg: b.dmg[j] * X.CHAIN_MUL, ch: b.ch[j],
          chain: b.chain[j] - 1, hit: hit2 });
        b.on[j] = false;
      }
      for (let u = 0; u < U; u++) if (by[u] >= 0) last_by[u] = by[u];
      return [heals, jumps];
    }

    // one tick (Ball.step). move/adir/sdir [P][2], atk/sup/gad/hyp [P] bool; ev: [t, kind, a, b, amount] are pushed.
    // A super aim longer than 1 is the vector to the target point (Jessie's throw).
    function step(s0, move, atk, adir, sup, sdir, gad, hyp, ev) {
      const T = s0.t, run = !s0.done, act = run && T >= s0.freeze_until;
      const [block, shot, bush] = grids(s0);
      const s = structuredClone(s0);
      const E = (kind, a, b, amt) => ev && ev.push([T, kind, a, b, Math.round(amt)]);
      const back = s.alive.map((al, p) => !al && s.respawn_at[p] >= 0 && T >= s.respawn_at[p] && run);
      revive(s, back, s.respawn);
      back.forEach((m, p) => m && E('respawn', p, p, 0));
      const live = s.alive.map(a => a && act), holds = ar.map(p => p === s.carrier), ctl = live;
      const kd = s.kind, isM = kd.map(k => k === 0), isS = kd.map(k => k === 1), isJ = kd.map(k => k === 2);
      const maxhp = kd.map(k => K.HP[k]);
      // hypercharge
      let hon = s.hyper_until.map(u => T < u);
      const hy = ar.map(p => ctl[p] && hyp[p] && s.hyper[p] >= 1 && s.ulti[p] >= 1 && !hon[p]);
      const hyper_until = ar.map(p => hy[p] ? T + c.HYPER_TICKS : s.hyper_until[p]);
      let hyper = ar.map(p => hy[p] ? 0 : s.hyper[p]);
      hon = hon.map((h, p) => h || hy[p]);
      hy.forEach((m, p) => m && E('hyper', p, -1, 0));
      const shield = fill(U, u => u < P && hon[u] ? 1 - c.HYPER_SHIELD : 1);   // turrets have no hypercharge
      const mul = fill(P, a => fill(U, u => (hon[a] ? 1 + c.HYPER_DMG : 1) * shield[u]));
      // reload
      const rl = ar.map(p => live[p] && s.ammo[p] < c.AMMO);
      let reload = ar.map(p => s.reload[p] + rl[p]);
      const got = ar.map(p => rl[p] && reload[p] >= K.RELOAD[kd[p]]);
      let ammo = ar.map(p => s.ammo[p] + got[p]);
      reload = ar.map(p => got[p] || (live[p] && ammo[p] >= c.AMMO) ? 0 : reload[p]);
      // the carrier kicks: attack (one ammo) or super (farther)
      const kk = ar.map(p => ctl[p] && holds[p] && atk[p] && ammo[p] >= 1);
      const ks = ar.map(p => ctl[p] && holds[p] && sup[p] && s.ulti[p] >= 1 && !kk[p]);
      const kick = ar.some(p => kk[p] || ks[p]);
      const w = Math.max(s.carrier, 0), skw = ks[w];
      let ball_dir = kick ? unit(skw ? sdir[w] : adir[w]) : s.ball_dir.slice();
      let ball_rem = kick ? (skw ? c.SUPER_KICK : c.KICK_RANGE) : s.ball_rem;
      let ball_pos = kick ? s.pos[w].slice() : s.ball_pos.slice();
      let carrier = kick ? -1 : s.carrier, last_touch = kick ? w : s.last_touch;
      const kicker = kick ? w : s.kicker, nopick_until = kick ? T + c.KICK_NOPICK_TICKS : s.nopick_until;
      let ulti = ar.map(p => kick && p === w && skw ? 0 : s.ulti[p]);
      ammo = ar.map(p => ammo[p] - (kick && p === w && !skw));
      let reveal = ar.map(p => kick && p === w ? T + c.REVEAL_TICKS : s.reveal_until[p]);
      if (kick) E(skw ? 'superkick' : 'kick', w, -1, ball_rem);
      // gadgets: Mortis' combo spinner hits every enemy unit in range with a clear line, Stu drops a speed zone,
      // Jessie's spark plug slows the enemies around her turret (no turret: no use spent)
      const [upos0, ok0] = units(s, s.pos);
      const g = ar.map(p => ctl[p] && gad[p] && s.gad_left[p] > 0 && T >= s.gad_ready[p]);
      const gm = ar.map(p => g[p] && isM[p]);
      const sp = fill(P, a => fill(U, u => gm[a] && ufoe[a][u] && ok0[u] && dist(s.pos[a], upos0[u]) <= c.SPIN_R &&
        clearShot(s.pos[a], upos0[u], shot)));
      const dmg = fill(U, () => 0), gain = fill(P, () => 0), heal = fill(P, () => 0);
      for (let a = 0; a < P; a++) for (let u = 0; u < U; u++) if (sp[a][u]) {
        dmg[u] += c.SPIN_DMG * mul[a][u]; E('spin_hit', a, u, c.SPIN_DMG * mul[a][u]);
      }
      let last_by = credit(s.last_by, sp);
      let last_attack = ar.map(p => gm[p] ? T : s.last_attack[p]);
      reveal = ar.map(p => gm[p] ? T + c.REVEAL_TICKS : reveal[p]);
      const used = gm.slice(), zone_pos = s.zone_pos, zone_until = s.zone_until, slow_until = s.slow_until;
      if (mt) {
        for (let p = 0; p < P; p++) if (g[p] && isS[p]) { zone_pos[p] = s.pos[p].slice(); zone_until[p] = T + X.ZONE_TICKS; used[p] = true; }
        const gj = ar.map(o => g[o] && isJ[o] && s.tur_on[o]);
        const slw = ar.map(q => ar.some(o => gj[o] && foe[o][q] && s.alive[q] && dist(s.tur_pos[o], s.pos[q]) <= X.SPARK_R));
        for (let p = 0; p < P; p++) { if (slw[p]) slow_until[p] = T + X.SPARK_TICKS; if (gj[p]) used[p] = true; }
        for (let o = 0; o < P; o++) for (let q = 0; q < P; q++)
          if (gj[o] && foe[o][q] && s.alive[q] && dist(s.tur_pos[o], s.pos[q]) <= X.SPARK_R) E('slow', o, q, X.SPARK_SLOW);
      }
      const gad_left = ar.map(p => s.gad_left[p] - used[p]);
      const gad_ready = ar.map(p => used[p] ? T + K.GAD_CD[kd[p]] : s.gad_ready[p]);
      used.forEach((m, p) => m && E('gadget', p, -1, 0));
      // attack: Mortis dashes, Stu / Jessie shoot (not with the ball)
      const can = ar.map(p => ctl[p] && atk[p] && !(s.dash_rem[p] > 0) && ammo[p] >= 1 && !holds[p] && T >= s.atk_ready[p]);
      const go = ar.map(p => can[p] && isM[p]), sh = ar.map(p => can[p] && !isM[p]);
      const reach = ar.map(p => c.DASH_RANGE + (T - last_attack[p] >= c.LONG_TICKS && ammo[p] >= c.AMMO ? c.LONG_EXTRA : 0));
      let dash_rem = ar.map(p => go[p] ? reach[p] : s.dash_rem[p]);
      const dash_dir = ar.map(p => go[p] ? unit(adir[p]) : s.dash_dir[p].slice());
      const dash_hit = ar.map(a => s.dash_hit[a].map(h => h && !go[a]));
      let burst = ar.map(p => sh[p] ? K.SHOTS[kd[p]] : s.burst[p]);
      let burst_next = ar.map(p => sh[p] ? T : s.burst_next[p]);
      const burst_dir = ar.map(p => sh[p] ? unit(adir[p]) : s.burst_dir[p].slice());
      const atk_ready = ar.map(p => sh[p] ? T + K.ATK_CD[kd[p]] : s.atk_ready[p]);
      ammo = ar.map(p => ammo[p] - can[p]);
      last_attack = ar.map(p => can[p] ? T : last_attack[p]);
      reveal = ar.map(p => can[p] ? T + c.REVEAL_TICKS : reveal[p]);
      go.forEach((m, p) => m && E('dash', p, -1, reach[p]));
      sh.forEach((m, p) => m && E('shot', p, -1, 0));
      const b = s.bul;
      let bseq = s.bseq;
      if (mt) {
        const fire = ar.map(p => live[p] && burst[p] > 0 && T >= burst_next[p]);
        bseq = spawn(b, bseq, ar.map(p => {
          const bk = isJ[p] ? 1 : 0;
          return { want: fire[p], order: p, own: p, pos: s.pos[p], d: burst_dir[p], kind: bk, rem: X.B_RANGE[bk], step: X.B_STEP[bk],
                   r: X.B_R[bk], dmg: X.B_DMG[bk], ch: X.B_CH[bk], chain: X.B_CHAIN[bk], hit: fill(U, () => false) };
        }));
        burst = ar.map(p => burst[p] - fire[p]);
        burst_next = ar.map(p => fire[p] ? T + K.GAP[kd[p]] : burst_next[p]);
      }
      // super: Mortis bats, Stu nitro dash, Jessie turret throw (not with the ball)
      const su = ar.map(p => ctl[p] && sup[p] && !(dash_rem[p] > 0) && ulti[p] >= 1 && !holds[p] && !(isM[p] && s.proj_on[p]));
      const mb = ar.map(p => su[p] && isM[p]);
      let proj_on = ar.map(p => s.proj_on[p] || mb[p]);
      let proj_pos = ar.map(p => mb[p] ? s.pos[p].slice() : s.proj_pos[p].slice());
      const proj_dir = ar.map(p => mb[p] ? unit(sdir[p]) : s.proj_dir[p].slice());
      let proj_rem = ar.map(p => mb[p] ? c.SUPER_RANGE : s.proj_rem[p]);
      const proj_hit = ar.map(a => s.proj_hit[a].map(h => h && !mb[a]));
      const proj_back = ar.map(p => mb[p] ? hon[p] : s.proj_back[p]);
      let proj_ret = ar.map(p => s.proj_ret[p] && !mb[p]);
      ulti = ar.map(p => su[p] ? 0 : ulti[p]);
      if (mt) for (let p = 0; p < P; p++) {
        if (su[p] && isS[p]) {                  // nitro; Infinitro: Stu's super refills when hypercharged
          dash_rem[p] = X.NITRO; dash_dir[p] = unit(sdir[p]); dash_hit[p] = dash_hit[p].map(() => false);
          if (hon[p]) ulti[p] = 1;
        }
        if (su[p] && isJ[p]) {
          const Ls = nrm(sdir[p][0], sdir[p][1]);
          const dj = Ls > 1e-9 ? [sdir[p][0] / Math.max(Ls, 1e-9), sdir[p][1] / Math.max(Ls, 1e-9)] : ahead(p);
          const dd = Ls <= 1.0001 ? X.THROW_RANGE : Math.min(Ls, X.THROW_RANGE);
          s.th_on[p] = true; s.th_a[p] = s.pos[p].slice(); s.th_b[p] = [s.pos[p][0] + dj[0] * dd, s.pos[p][1] + dj[1] * dd];
          s.th_t0[p] = T; s.th_t1[p] = T + Math.max(1, roundEven(dd / X.THROW_SPEED / c.DT)); s.th_hyper[p] = hon[p];
        }
      }
      last_attack = ar.map(p => su[p] ? T : last_attack[p]);
      reveal = ar.map(p => su[p] ? T + c.REVEAL_TICKS : reveal[p]);
      su.forEach((m, p) => m && E('super', p, -1, 0));
      // movement: dash (fixed direction) or walk, 2 substeps with wall push-out; a friendly speed zone, Spark Plug's slow
      const dashing = ar.map(p => live[p] && dash_rem[p] > 0);
      const start = s.pos.map(p => p.slice());
      const dlen = ar.map(p => Math.min(dash_rem[p], K.DSTEP[kd[p]]));
      const want = ar.map(p => {
        if (!live[p]) return [0, 0];
        if (dashing[p]) return [dash_dir[p][0] * dlen[p], dash_dir[p][1] * dlen[p]];
        let spd = K.SPEED[kd[p]];
        if (mt) {
          const inz = ar.some(o => G.mate[p][o] && zone_until[o] >= T && dist(s.pos[p], zone_pos[o]) <= X.ZONE_R);
          spd = (inz ? spd + X.ZONE_SPEED : spd) - (T < slow_until[p] ? X.SPARK_SLOW : 0);
        }
        spd = Math.max(spd, c.MIN_SPEED) * (hon[p] ? 1 + c.HYPER_SPEED : 1);
        const m = Math.max(1, nrm(move[p][0], move[p][1]));
        return [move[p][0] / m * (spd * c.DT), move[p][1] / m * (spd * c.DT)];
      });
      let pos = start.map(p => p.slice());
      for (let i = 0; i < 2; i++) pos = pos.map((p, q) => live[q] ? push([p[0] + want[q][0] / 2, p[1] + want[q][1] / 2], block) : p);
      for (let p = 0; p < P; p++) {
        const made = (pos[p][0] - start[p][0]) * dash_dir[p][0] + (pos[p][1] - start[p][1]) * dash_dir[p][1];
        if (dashing[p]) dash_rem[p] = made < 0.5 * dlen[p] ? 0 : dash_rem[p] - dlen[p];
      }
      const vel = pos.map((p, q) => [p[0] - start[q][0], p[1] - start[q][1]]);
      if (mt) for (let p = 0; p < P; p++) if (dashing[p] && isS[p]) {   // Stu: fire trail while dashing
        s.fz_pos[p][T % FZ] = pos[p].slice(); s.fz_until[p][T % FZ] = T + X.FIRE_TICKS;
      }
      // the ball: follows its carrier; a free one flies, bounces, stops; then pickups (a tie: the lowest index)
      if (carrier >= 0) ball_pos = pos[carrier].slice();
      else if (act && ball_rem > 0) {
        const L = Math.min(ball_rem, c.BALL_STEP);
        [ball_pos, ball_dir] = fly(ball_pos, ball_dir, L, shot);
        ball_rem -= L;
      }
      if (act && carrier < 0) {
        let wp = -1, best = Infinity;
        for (let p = 0; p < P; p++) {
          const d = dist(pos[p], ball_pos);
          if (s.alive[p] && d <= R + c.BALL_R && !(p === kicker && T < nopick_until) && d < best) { best = d; wp = p; }
        }
        if (wp >= 0) { carrier = last_touch = wp; ball_rem = 0; ball_pos = pos[wp].slice(); E('pick', wp, -1, 0); }
      }
      const [upos, ok] = units(s, pos);
      // Mortis dash hits: an enemy unit touches the path walked this tick, once per dash
      const hit = ar.map(a => fill(U, u => dashing[a] && isM[a] && ufoe[a][u] && ok[u] && !dash_hit[a][u] &&
        segDist(upos[u], start[a], pos[a]) <= 2 * R + c.DASH_HIT_EXTRA));
      for (let a = 0; a < P; a++) for (let u = 0; u < U; u++) if (hit[a][u]) {
        dash_hit[a][u] = true; dmg[u] += c.DASH_DMG * mul[a][u]; gain[a] += c.CHARGE_ATK; E('hit', a, u, c.DASH_DMG * mul[a][u]);
      }
      last_by = credit(last_by, hit);
      // bats; Blood Boomerang bats turn around at the end and home back on Mortis
      const on = ar.map(p => proj_on[p] && act), ret = ar.map(p => on[p] && proj_ret[p]), ps = proj_pos.map(p => p.slice());
      for (let p = 0; p < P; p++) {
        if (!on[p]) continue;
        const dv = ret[p] ? unit([pos[p][0] - proj_pos[p][0], pos[p][1] - proj_pos[p][1]]) : proj_dir[p];
        const pl = ret[p] ? c.BACK_STEP : Math.min(proj_rem[p], c.PROJ_STEP);
        proj_pos[p] = [proj_pos[p][0] + dv[0] * pl, proj_pos[p][1] + dv[1] * pl];
        if (!ret[p]) proj_rem[p] -= pl;
      }
      const bh = ar.map(a => fill(U, u => on[a] && ufoe[a][u] && ok[u] && !proj_hit[a][u] &&
        segDist(upos[u], ps[a], proj_pos[a]) <= c.SUPER_RADIUS + R));
      for (let a = 0; a < P; a++) for (let u = 0; u < U; u++) if (bh[a][u]) {
        const amt = c.SUPER_DMG * mul[a][u];
        proj_hit[a][u] = true; dmg[u] += amt; gain[a] += c.CHARGE_SUP; heal[a] += amt * c.LIFESTEAL; E('bats', a, u, amt);
      }
      last_by = credit(last_by, bh);
      for (let p = 0; p < P; p++) {
        if (on[p] && !ret[p] && proj_rem[p] <= 0 && proj_back[p]) { proj_ret[p] = true; proj_hit[p] = proj_hit[p].map(() => false); }
        const home = ret[p] && (dist(proj_pos[p], pos[p]) <= c.BACK_STEP || !s.alive[p]);
        proj_on[p] = proj_on[p] && (proj_rem[p] > 0 || proj_ret[p]) && !home;
        proj_ret[p] = proj_ret[p] && proj_on[p];
      }
      if (mt) bseq = tail(s, act, pos, reveal, upos, ok, shot, bush, block, mul, b, bseq, dmg, gain, last_by, dash_hit, proj_hit, E);
      // damage, Creepy Harvest (a Mortis killer heals)
      const last_hurt = ar.map(p => dmg[p] > 0 ? T : s.last_hurt[p]);
      reveal = ar.map(p => dmg[p] > 0 ? T + c.REVEAL_TICKS : reveal[p]);
      let hp = ar.map(p => s.hp[p] - dmg[p]);
      const dead = ar.map(p => s.alive[p] && hp[p] <= 0);
      for (let a = 0; a < P; a++) {
        const kills = ar.filter(q => dead[q] && last_by[q] === a).length;
        if (s.alive[a] && isM[a]) heal[a] += kills * c.HARVEST * maxhp[a];
        if (live[a] && !dead[a] && heal[a] > 0) { E('heal', a, a, Math.min(maxhp[a], hp[a] + heal[a]) - hp[a]); hp[a] = Math.min(maxhp[a], hp[a] + heal[a]); }
      }
      if (mt) for (let o = 0; o < P; o++) {     // turrets
        s.tur_hp[o] -= dmg[P + o];
        if (s.tur_on[o] && s.tur_hp[o] <= 0) { s.tur_on[o] = false; E('turret_down', last_by[P + o], P + o, 0); }
      }
      // super charge, hypercharge meter, regen out of combat
      ulti = ar.map(p => Math.min(ulti[p] + gain[p], 1));
      hyper = ar.map(p => Math.min(hyper[p] + (hon[p] ? 0 : gain[p] * K.HMUL[kd[p]]), 1));
      for (let p = 0; p < P; p++) {
        const calm = T - Math.max(last_attack[p], last_hurt[p]) - c.REGEN_DELAY_TICKS;
        if (live[p] && !dead[p] && calm >= 0 && calm % (1000 / c.TICK_MS) === 0 && hp[p] < maxhp[p])
          hp[p] += Math.min(maxhp[p] - hp[p], maxhp[p] * c.REGEN_PER_S);
      }
      // deaths: respawn later, a dead carrier drops the ball where he fell
      hp = ar.map(p => dead[p] ? 0 : hp[p]);
      const respawn_at = ar.map(p => dead[p] ? T + c.RESPAWN_TICKS : s.respawn_at[p]);
      dead.forEach((m, p) => {
        if (!m) return;
        dash_rem[p] = 0; burst[p] = 0; hyper_until[p] = 0; slow_until[p] = 0; s.burn_until[p] = 0; E('death', p, p, 0);
      });
      if (carrier >= 0 && dead[carrier]) { E('drop', carrier, -1, 0); carrier = -1; ball_rem = 0; }
      Object.assign(s, {
        pos, vel, hp, alive: ar.map(p => s.alive[p] && !dead[p]), ammo, reload, last_attack, last_hurt, last_by, ulti,
        reveal_until: reveal, atk_ready, burst, burst_next, burst_dir, dash_rem, dash_dir, dash_hit, proj_on, proj_pos,
        proj_dir, proj_rem, proj_hit, proj_back, proj_ret, hyper, hyper_until, gad_left, gad_ready, slow_until, zone_pos,
        zone_until, bseq, respawn_at, ball_pos, ball_dir, ball_rem, carrier, last_touch, kicker, nopick_until,
      });
      return end(s, run, act, E);
    }

    // the multi part of the step (Ball._tail): bullets fly, Jessie's turret lands, turrets shoot, fire burns.
    // Changes s.tur_* / th_* / burn_*, b, dmg, gain, last_by, dash_hit, proj_hit in place -> bullets made so far
    function tail(s, act, pos, reveal, upos, ok, shot, bush, block, mul, b, bseq, dmg, gain, last_by, dash_hit, proj_hit, E) {
      const T = s.t;
      const [heals, jumps] = flyBullets(b, act, upos, ok, s.tur_on, shot, mul, dmg, gain, last_by, E);
      bseq = spawn(b, bseq, jumps);
      for (let o = 0; o < P; o++) if (heals[o] > 0) s.tur_hp[o] = s.tur_max[o] - (s.tur_max[o] - s.tur_hp[o]) * Math.pow(1 - X.ENERGIZE, heals[o]);
      for (let o = 0; o < P; o++) {             // her turret lands (replaces the old one)
        if (!(act && s.th_on[o] && s.th_t1[o] <= T)) continue;
        const hp0 = X.TUR_HP * (s.th_hyper[o] ? 1 + X.S2_HP : 1);
        s.tur_on[o] = true;
        s.tur_pos[o] = push([clip(s.th_b[o][0], R, G.size[0] - R), clip(s.th_b[o][1], R, G.size[1] - R)], block);
        s.tur_max[o] = s.tur_hp[o] = hp0; s.tur_dmg[o] = X.TUR_DMG * (s.th_hyper[o] ? 1 + X.S2_DMG : 1); s.tur_next[o] = T + X.TUR_EVERY;
        s.burn_until[P + o] = 0;
        for (let a = 0; a < P; a++) { dash_hit[a][P + o] = false; proj_hit[a][P + o] = false; }
        s.th_on[o] = false;
        E('turret', o, P + o, hp0);
      }
      // turrets shoot the nearest visible enemy brawler they can hit, else an enemy turret (old unit positions)
      const v = vis(pos, s.alive, reveal, T, bush), rows = [];
      for (let o = 0; o < P; o++) {
        if (!(act && s.tur_on[o] && T >= s.tur_next[o])) continue;
        let best = -1, bk = Infinity;
        for (let u = 0; u < U; u++) {
          if (!(ok[u] && ufoe[o][u] && (u >= P || v[o][u]))) continue;
          const d = dist(upos[u], s.tur_pos[o]);
          if (!(d <= X.TUR_RANGE) || !clearShot(s.tur_pos[o], upos[u], shot, X.TUR_PTS)) continue;
          const key = d + (u >= P ? 1e5 : 0);
          if (key < bk) { bk = key; best = u; }
        }
        if (best < 0) continue;
        rows.push({ want: true, order: o, own: o, pos: s.tur_pos[o], d: [upos[best][0] - s.tur_pos[o][0], upos[best][1] - s.tur_pos[o][1]],
                    kind: 2, rem: X.TUR_RANGE, step: X.TUR_STEP, r: X.TUR_R, dmg: s.tur_dmg[o], ch: 0, chain: 0, hit: fill(U, () => false) });
        s.tur_next[o] = T + X.TUR_EVERY;
      }
      bseq = spawn(b, bseq, rows);
      // fire: a unit in an enemy zone burns (the first tick at once), the zone made last is the source
      for (let u = 0; u < U; u++) {
        let top = -1, src = -1;
        if (act && ok[u]) for (let o = 0; o < P; o++) {
          if (!ufoe[o][u]) continue;
          for (let k = 0; k < FZ; k++) {
            const key = s.fz_until[o][k] * P + o;
            if (s.fz_until[o][k] > T && dist(upos[u], s.fz_pos[o][k]) <= X.FIRE_R && key > top) { top = key; src = o; }
          }
        }
        if (src >= 0) {
          if (s.burn_until[u] <= T) s.burn_next[u] = T;
          s.burn_until[u] = T + X.BURN_TICKS; s.burn_src[u] = src;
        }
        if (act && ok[u] && s.burn_until[u] > T && s.burn_next[u] <= T) {
          const o = s.burn_src[u], bs = Math.max(o, 0), amt = X.BURN_DMG * mul[bs][u];
          dmg[u] += amt; last_by[u] = o; gain[bs] += X.BURN_CH; s.burn_next[u] = T + X.BURN_EVERY; E('burn', o, u, amt);
        }
      }
      return bseq;
    }

    function end(s, run, act, E) {             // goals, regulation end, overtime, final draw (Ball._end)
      const [bx, by] = s.ball_pos, cteam = s.carrier >= 0 ? team[s.carrier] : -1;
      const ins = [0, 1].map(k => { const g = G.goal[k]; return act && bx > g[0] && bx < g[2] && by > g[1] && by < g[3] && cteam !== k; });
      const score = [s.score[0] + ins[1], s.score[1] + ins[0]];
      const scored = ins[0] || ins[1], amax = score[1] > score[0] ? 1 : 0;
      const win = scored && (Math.max(...score) >= c.MAX_SCORE || s.ot);
      let winner = win ? amax : s.winner;
      const t = s.t + run;
      ins.forEach((m, k) => m && E('goal', s.last_touch, 1 - k, score[1 - k]));
      s.score = score; s.t = t;
      if (scored && !win) {                     // kickoff: everyone back to spawn, ball to the center, short freeze;
        revive(s, s.here, s.spawn);             // turrets, bullets, fire and speed zones, burns and slows are gone
        const no = a => a.map(() => false);
        Object.assign(s, { proj_on: no(s.proj_on), proj_ret: no(s.proj_ret), hyper_until: s.hyper_until.map(() => 0),
          tur_on: no(s.tur_on), th_on: no(s.th_on), fz_until: s.fz_until.map(r => r.map(() => 0)), zone_until: s.zone_until.map(() => -1),
          burn_until: s.burn_until.map(() => 0), slow_until: s.slow_until.map(() => 0), ball_pos: G.center.slice(), ball_rem: 0,
          carrier: -1, kicker: -1, freeze_until: t + c.KICKOFF_TICKS });
        s.bul.on = no(s.bul.on);
      }
      const go = run && !win, reg = go && !s.ot && t === c.REG_TICKS, lead = reg && score[0] !== score[1];
      if (lead) winner = amax;
      if (reg && !lead) { s.ot = true; E('overtime', -1, -1, 0); }
      const tout = go && !lead && t >= c.MAX_TICKS;
      if (tout) winner = -1;
      s.winner = winner; s.done = s.done || win || lead || tout;
      return s;
    }

    return { G, c, P, U, tile, grids, inBush, maxHp, longReady, push, fly, clearShot, visible, reset, step };
  }

  // ---------------------------------------------------------------- the net's view (rl/jballenv.py, hint off)
  function Env(G) {
    const b = Sim(G), c = G.c, K = G.k, X = G.x || {}, P = G.P, TILE = c.TILE, team = G.team, NDIR = c.NDIR, DIRS = G.dirs;
    const mt = !!G.multi, MT = mt ? X.MAXT : G.T, ar = Array.from({ length: P }, (_, i) => i);

    function see(es) {
      const s = es.s, v = b.visible(s);
      for (let p = 0; p < P; p++) {
        for (let q = 0; q < P; q++) if (v[p][q]) {
          es.seen_pos[p][q] = s.pos[q].slice(); es.seen_t[p][q] = s.t; es.seen_hp[p][q] = s.hp[q];
        }
        if (s.carrier < 0 || v[p][s.carrier]) es.ball_seen[p] = s.ball_pos.slice();
      }
      es.vis = v;
      return es;
    }

    function reset(kind, n) {                  // kind (P,) indices into G.kinds, n (2,) players a side
      const s = b.reset(kind, n);
      return see({ s, seen_pos: ar.map(() => s.pos.map(x => x.slice())), seen_t: ar.map(() => ar.map(() => 0)),
                   seen_hp: ar.map(() => ar.map(q => b.maxHp(s, q))), ball_seen: ar.map(() => s.ball_pos.slice()),
                   vis: ar.map(p => ar.map(q => p === q)) });
    }

    function tick(es, move, atk, adir, sup, sdir, gad, hyp, ev) {
      return see(Object.assign({}, es, { s: b.step(es.s, move, atk, adir, sup, sdir, gad, hyp, ev) }));
    }

    function masks(es, p) {                    // legal buttons: none, attack, super, gadget, hypercharge + super
      const s = es.s, holds = p === s.carrier, free = !(s.dash_rem[p] > 0);
      const sup = s.alive[p] && s.ulti[p] >= 1 && (holds || (free && !(s.kind[p] === 0 && s.proj_on[p])));
      return [true, s.alive[p] && s.ammo[p] >= 1 && (holds || (free && s.t >= s.atk_ready[p])), sup,
              s.alive[p] && s.gad_left[p] > 0 && s.t >= s.gad_ready[p] && (s.kind[p] !== 2 || s.tur_on[p]),
              sup && s.hyper[p] >= 1 && !(s.t < s.hyper_until[p])];
    }

    function segClear(grid, a, bb) {
      const n = c.LOS_PTS;
      for (let i = 0; i < n; i++) {
        const f = i / (n - 1);
        if (b.tile(grid, Math.floor((a[0] + (bb[0] - a[0]) * f) / TILE), Math.floor((a[1] + (bb[1] - a[1]) * f) / TILE))) return false;
      }
      return true;
    }

    function obs(es, p) {                      // BallEnv._obs1, float32
      const s = es.s, [block, shot, bush] = b.grids(s), k = team[p], sg = G.sg[p], t = s.t, me = s.pos[p];
      const size = G.size, kd = s.kind, o = [];
      const put = (...xs) => { for (const x of xs) o.push(+x); };
      const cpos = x => k === 0 ? [x[0], x[1]] : [size[0] - x[0], size[1] - x[1]];
      const rel = x => [sg * (x[0] - me[0]) / c.SCALE, sg * (x[1] - me[1]) / c.SCALE];
      const resp = q => s.alive[q] ? 0 : clip((s.respawn_at[q] - t) / c.RESPAWN_TICKS, 0, 1);
      const dst = (x, y) => nrm(x[0] - y[0], x[1] - y[1]) / c.SCALE;
      const mhp = q => b.maxHp(s, q), car = s.carrier, free = car < 0;
      const calm = t - Math.max(s.last_attack[p], s.last_hurt[p]);
      const hot = (i, n) => fill(n, j => +(j === i));
      const kit = q => {                        // multi: brawler, turret, thrown turret, speed zone
        if (!mt) return [];
        const on = +s.tur_on[q], th = +s.th_on[q], zn = +(s.zone_until[q] >= t), tp = rel(s.tur_pos[q]), tb = rel(s.th_b[q]);
        const zp = rel(s.zone_pos[q]);
        return [s.here[q], ...hot(kd[q], X.NK), on, on * tp[0], on * tp[1], on * s.tur_hp[q] / Math.max(s.tur_max[q], 1), th,
                th * tb[0], th * tb[1], th * clip((s.th_t1[q] - t) / 25, 0, 1), zn, zn * zp[0], zn * zp[1]];
      };
      put(s.alive[p], resp(p), ...G.slot[p], car === p, s.kicker === p && t < s.nopick_until, s.hp[p] / mhp(p),
          (s.ammo[p] + s.reload[p] / K.RELOAD[kd[p]]) / c.AMMO);
      for (let i = 0; i <= c.AMMO; i++) put(s.ammo[p] === i);
      put(b.longReady(s, p), clip((t - s.last_attack[p]) / c.LONG_TICKS, 0, 1), s.ulti[p], s.ulti[p] >= 1, s.hyper[p],
          t < s.hyper_until[p], clip((s.hyper_until[p] - t) / 100, 0, 1), s.gad_left[p] / 3, t >= s.gad_ready[p],
          clip((s.gad_ready[p] - t) / K.GAD_CD[kd[p]], 0, 1), s.dash_rem[p] > 0, s.dash_rem[p] / (c.DASH_RANGE + c.LONG_EXTRA),
          sg * s.dash_dir[p][0], sg * s.dash_dir[p][1], s.proj_on[p], ...rel(s.proj_pos[p]), sg * s.proj_dir[p][0],
          sg * s.proj_dir[p][1], s.proj_ret[p], s.proj_back[p], s.proj_rem[p] / c.SUPER_RANGE, t < s.reveal_until[p],
          b.inBush(bush, me), cpos(me)[0] / size[0], cpos(me)[1] / size[1], sg * s.vel[p][0] / c.DASH_STEP,
          sg * s.vel[p][1] / c.DASH_STEP, clip(calm / 60, 0, 2));
      if (mt) {                                 // me: kit, attack wait, burning, slowed, in a friendly speed zone
        const inz = ar.some(z => team[z] === k && s.zone_until[z] >= t && dist(s.zone_pos[z], me) <= X.ZONE_R);
        put(...kit(p), clip((s.atk_ready[p] - t) / 10, 0, 1), s.burst[p] > 0, t < s.burn_until[p], t < s.slow_until[p], inz);
      }
      const sc = k === 0 ? s.score : [s.score[1], s.score[0]], their = G.gc[1 - k], ours = G.gc[k], bs = es.ball_seen[p];
      const ct = team[Math.max(car, 0)];
      put(t / c.MAX_TICKS, clip(t / c.REG_TICKS, 0, 1), s.ot, clip((s.freeze_until - t) / 40, 0, 1), sc[0] / 2, sc[1] / 2,
          ...rel(bs), dst(bs, me), cpos(bs)[0] / size[0], cpos(bs)[1] / size[1], free, !free && ct === k && car !== p,
          !free && ct !== k, free * sg * s.ball_dir[0], free * sg * s.ball_dir[1], free * s.ball_rem / c.SUPER_KICK,
          ...rel(their), dst(their, me), segClear(shot, me, their), ...rel(ours));
      put(0, 0, 0, 0, 0, 0, 0, 0);                                         // trickshot hint: off
      const mate = q => [s.alive[q], resp(q), ...rel(s.pos[q]), dst(s.pos[q], me), s.hp[q] / mhp(q), s.ammo[q] / c.AMMO,
        s.ulti[q] >= 1, s.dash_rem[q] > 0, car === q, s.proj_on[q], segClear(shot, me, s.pos[q]),
        ...kit(q), ...(mt ? [t < s.burn_until[q], t < s.slow_until[q]] : [])];
      const foe = q => {
        const v = es.vis[p][q], sp = es.seen_pos[p][q], r = [sp[0] - me[0], sp[1] - me[1]];
        const L = Math.max(nrm(r[0], r[1]), 1e-6), side = [-r[1] / L * c.R, r[0] / L * c.R];
        const walk = segClear(block, me, sp) && segClear(block, [me[0] + side[0], me[1] + side[1]], [sp[0] + side[0], sp[1] + side[1]])
          && segClear(block, [me[0] - side[0], me[1] - side[1]], [sp[0] - side[0], sp[1] - side[1]]);
        return [s.alive[q], resp(q), v, ...rel(sp), nrm(r[0], r[1]) / c.SCALE, clip((t - es.seen_t[p][q]) / 100, 0, 1),
          es.seen_hp[p][q] / mhp(q), v * sg * s.vel[q][0] / c.DASH_STEP, v * sg * s.vel[q][1] / c.DASH_STEP,
          v && s.dash_rem[q] > 0, v * sg * s.dash_dir[q][0], v * sg * s.dash_dir[q][1], v && s.ulti[q] >= 1,
          v && t < s.hyper_until[q], v && b.inBush(bush, s.pos[q]), v && car === q, s.proj_on[q], ...rel(s.proj_pos[q]),
          sg * s.proj_dir[q][0], sg * s.proj_dir[q][1], s.proj_ret[q], segClear(shot, me, sp), walk,
          ...kit(q), ...(mt ? [v && t < s.burn_until[q], v && t < s.slow_until[q], v && s.burst[q] > 0,
            v * clip((s.atk_ready[q] - t) / 10, 0, 1)] : [])];
      };
      const rows = (fn, qs, n) => {             // n rows, the missing ones (smaller team) zero
        for (const q of qs) put(...fn(q));
        for (let i = qs.length; i < n; i++) put(...fn(0).map(() => 0));
      };
      rows(mate, ar.filter(q => team[q] === k && q !== p).slice(0, G.T - 1), MT - 1);
      rows(foe, ar.filter(q => team[q] !== k).slice(0, G.T), MT);
      if (mt) {
        const nearest = (on, pts, n) => fill(on.length, i => i).map(i => [on[i] ? dist(pts[i], me) : Infinity, i])
          .sort((x, y) => x[0] - y[0]).slice(0, n).map(x => x[1]);
        const bl = s.bul, eb = bl.on.map((x, j) => x && team[Math.floor(j / X.KB)] !== k);   // the nearest enemy bullets
        const bi = nearest(eb, bl.pos, X.N_BUL);
        put(...bi.map(i => eb[i]));
        for (const i of bi) put(...rel(bl.pos[i]).map(x => eb[i] * x));
        for (const i of bi) put(eb[i] * sg * bl.dir[i][0], eb[i] * sg * bl.dir[i][1]);
        for (const i of bi) put(...hot(bl.kind[i], 3).map(x => eb[i] * x));
        const fe = [], fp = [];                 // the nearest enemy fire
        for (let q = 0; q < P; q++) for (let z = 0; z < X.FZ; z++) { fe.push(s.fz_until[q][z] > t && team[q] !== k); fp.push(s.fz_pos[q][z]); }
        const fi = nearest(fe, fp, X.N_FIRE);
        put(...fi.map(i => fe[i]));
        for (const i of fi) put(...rel(fp[i]).map(x => fe[i] * x));
      }
      const cp = cpos(me).map(x => x / TILE), c0 = cp.map(Math.floor), oi = s.ot ? 1 : 0, Gr = c.GRID;
      const blk = [], bsh = [];
      for (let oy = -Gr; oy <= Gr; oy++) for (let ox = -Gr; ox <= Gr; ox++) {
        const gx = c0[0] + ox, gy = c0[1] + oy, inb = gx >= 0 && gx < G.W && gy >= 0 && gy < G.H;
        const gxc = clip(gx, 0, G.W - 1), gyc = clip(gy, 0, G.H - 1);
        const i = k === 0 ? gyc * G.W + gxc : (G.H - 1 - gyc) * G.W + (G.W - 1 - gxc);   // team 1: the map turned 180
        blk.push(!inb || !!G.block[oi][i]); bsh.push(inb && !!G.bush[oi][i]);
      }
      put(...blk, ...bsh, cp[0] - c0[0], cp[1] - c0[1]);
      if (o.length !== G.obs_dim) throw new Error(`obs ${o.length} != ${G.obs_dim}`);
      return Float32Array.from(o);
    }

    function inputs(es, p, a) {                // action [move, button, aim] -> move, aim, super aim, presses
      const s = es.s, sg = G.sg[p], [am, ab, aa] = a, me = s.pos[p];
      const move = am > 0 ? [DIRS[am - 1][0] * sg, DIRS[am - 1][1] * sg] : [0, 0];
      let nf = 0, best = Infinity;
      for (let q = 0; q < P; q++) {
        const d = team[q] !== team[p] && s.alive[q] ? dist(es.seen_pos[p][q], me) : Infinity;
        if (d < best) { best = d; nf = q; }
      }
      const tg = p === s.carrier ? G.att[p] : es.seen_pos[p][nf], pt = [tg[0] - me[0], tg[1] - me[1]];
      const aim = aa < NDIR ? [DIRS[aa][0] * sg, DIRS[aa][1] * sg]
        : nrm(pt[0], pt[1]) > 1.5 ? pt : [DIRS[0][0] * sg * 1.5, DIRS[0][1] * sg * 1.5];
      return { move, aim, saim: aim, press: [ab === 1, ab === 2 || ab === 4, ab === 3, ab === 4] };
    }

    return { b, G, P, reset, tick, masks, obs, inputs };
  }

  // ---------------------------------------------------------------- the policy (rl/jppo.py _mlp + heads)
  function Net(buf, shapes) {
    const f = new Float32Array(buf), layers = [];
    let o = 0;
    for (const [n, m] of shapes) { layers.push({ n, m, w: f.subarray(o, o + n * m), b: f.subarray(o + n * m, o + n * m + m) }); o += n * m + m; }
    function logits(x) {
      for (let l = 0; l < layers.length; l++) {
        const { n, m, w, b } = layers[l], y = Float64Array.from(b);
        for (let i = 0; i < n; i++) { const xi = x[i]; if (xi === 0) continue; const r = i * m; for (let j = 0; j < m; j++) y[j] += xi * w[r + j]; }
        if (l < layers.length - 1) for (let j = 0; j < m; j++) y[j] = Math.max(0, Math.fround(y[j]));
        x = y;
      }
      return x;
    }
    return { logits };
  }

  function logSoftmax(v) {
    const mx = Math.max(...v), lse = mx + Math.log(v.reduce((a, x) => a + Math.exp(x - mx), 0));
    return v.map(x => x - lse);
  }
  function pick(lp, greedy) {
    if (greedy) return lp.indexOf(Math.max(...lp));
    let u = Math.random();
    for (let i = 0; i < lp.length; i++) { u -= Math.exp(lp[i]); if (u < 0) return i; }
    return lp.length - 1;
  }
  function heads(lg, bm, nMove, nBtn) {        // log-probs of move, button (masked), aim
    const a = Array.from(lg);
    return [logSoftmax(a.slice(0, nMove)), logSoftmax(a.slice(nMove, nMove + nBtn).map((x, i) => bm[i] ? x : -1e9)),
            logSoftmax(a.slice(nMove + nBtn))];
  }

  // ---------------------------------------------------------------- a game you play in the page (rl/jplay.py Game)
  function snapshot(env, es) {                 // Match.snapshot: one viewer frame
    const s = es.s, G = env.G, c = G.c, K = G.k, X = G.x || {}, T = G.T, P = G.P, t = s.t, bush = env.b.grids(s)[2];
    const rd = Math.round, r3 = v => rd(v * 1000) / 1000;
    // hidden: the enemies don't see you; with no enemy alive (dead see nothing): in a bush and not revealed
    const shown = q => es.vis[(1 - G.team[q]) * T][q] || !env.b.inBush(bush, s.pos[q]) || t < s.reveal_until[q];
    const p = s.pos.map((x, q) => {
      const kd = s.kind[q], ra = s.respawn_at[q];
      const fx = (t < s.burn_until[q] ? 1 : 0) | (t < s.slow_until[q] ? 2 : 0);
      return [rd(x[0]), rd(x[1]), rd(s.hp[q]), r3(s.ammo[q] + s.reload[q] / K.RELOAD[kd]),
              kd === 0 ? r3(clip((t - s.last_attack[q]) / c.LONG_TICKS, 0, 1)) : 0, +env.b.longReady(s, q), rd(s.ulti[q] * 1e4) / 1e4,
              +(s.dash_rem[q] > 0), +!shown(q), ra >= 0 ? Math.max(0, ra - t) : 0, r3(s.hyper[q]),
              Math.max(0, s.hyper_until[q] - t), s.gad_left[q], fx];
    });
    const f = { t, p, b: s.proj_pos.map((x, q) => [+s.proj_on[q], rd(x[0]), rd(x[1])]) };
    if (G.multi) {
      const bl = [], tu = [], ars = [], th = [];
      s.bul.on.forEach((on, j) => on && bl.push([rd(s.bul.pos[j][0]), rd(s.bul.pos[j][1]), 'sot'[s.bul.kind[j]], G.team[Math.floor(j / X.KB)]]));
      for (let o = 0; o < P; o++) {
        if (s.tur_on[o]) tu.push([o, rd(s.tur_pos[o][0]), rd(s.tur_pos[o][1]), rd(s.tur_hp[o]), rd(s.tur_max[o])]);
        s.fz_until[o].forEach((u, k) => u > t && ars.push(['fire', rd(s.fz_pos[o][k][0]), rd(s.fz_pos[o][k][1]), rd(X.FIRE_R), G.team[o]]));
        if (s.zone_until[o] >= t) ars.push(['zone', rd(s.zone_pos[o][0]), rd(s.zone_pos[o][1]), rd(X.ZONE_R), G.team[o]]);
        if (s.th_on[o]) {
          const a = s.th_a[o], e = s.th_b[o], k = (t - s.th_t0[o]) / Math.max(1, s.th_t1[o] - s.th_t0[o]);
          th.push([o, rd(a[0] + (e[0] - a[0]) * k), rd(a[1] + (e[1] - a[1]) * k), rd(e[0]), rd(e[1])]);
        }
      }
      if (bl.length) f.bl = bl;
      if (tu.length) f.tu = tu;
      if (ars.length) f.ar = ars;
      if (th.length) f.th = th;
    }
    return Object.assign(f, { ball: [rd(s.ball_pos[0]), rd(s.ball_pos[1]), s.carrier, +(s.carrier < 0 && s.ball_rem > 0)],
                              score: s.score.slice(), ot: +s.ot, frz: Math.max(0, s.freeze_until - t) });
  }

  function reason(s, G) {
    if (s.winner < 0) return 'draw';
    if (s.ot) return 'golden goal';
    return Math.max(...s.score) >= G.c.MAX_SCORE ? 'goals' : 'lead at time';
  }

  // setNet(buf, label): the next newGame plays that net. newGame(choice): {me, net} = brawler letters (of G.kinds,
  // e.g. ?play&me=S&net=J); a missing one is random (multi nets only, else everyone is Mortis)
  function LocalPlay(G, buf, label) {
    const env = Env(G), c = G.c, P = G.P, NM = c.NDIR + 1, NB = 5, PRESS = ['atk', 'sup', 'gad', 'hyp'];
    let net = Net(buf, G.net.shapes);
    label = label || G.nets[0].label;
    let game = null, n = 0;

    function newGame(choice) {
      if (game) clearInterval(game.timer);
      const me = 0, side = G.team[me], ALL = G.letters || 'M', L = G.kinds || 'M';
      const kind = Array.from({ length: P }, (_, p) => {
        const want = ((choice || {})[p === me ? 'me' : 'net'] || '').toUpperCase();
        return !G.multi ? 0 : ALL.indexOf(want && L.includes(want) ? want : L[Math.floor(Math.random() * L.length)]);
      });
      const es = env.reset(kind), rep = structuredClone(G.header);
      rep.players.forEach((pl, p) => {
        const name = p === me ? 'You' : label;
        Object.assign(pl, { brawler: G.labels[kind[p]], hp: G.k.HP[kind[p]], name: G.multi ? `${name} (${G.labels[kind[p]]})` : name });
      });
      Object.assign(rep, { me, side, id: n++, frames: [snapshot(env, es)], events: [], result: null });
      const sg = G.sg[me];
      game = { rep, es, me, side, k: 0, out: null, move: [0, 0], aim: [G.dirs[0][0] * sg, G.dirs[0][1] * sg],
               latch: { atk: false, sup: false, gad: false, hyp: false }, t0: performance.now() };
      game.timer = setInterval(() => run(game), c.TICK_MS / 2);
      return structuredClone(rep);
    }

    function act(es, p) {
      const [lm, lb, la] = heads(net.logits(env.obs(es, p)), env.masks(es, p), NM, NB);
      return env.inputs(es, p, [pick(lm), pick(lb), pick(la)]);
    }

    function tick(g) {
      const first = g.k % c.REPEAT === 0, ar = Array.from({ length: P }, (_, i) => i);
      if (first) g.out = ar.map(p => G.team[p] !== g.side ? act(g.es, p) : null);
      const hp = PRESS.map(x => g.latch[x]);
      g.latch = { atk: false, sup: false, gad: false, hyp: false };
      const I = ar.map(p => p === g.me ? { move: g.move, aim: g.aim, saim: g.aim, press: [hp[0], hp[1] || hp[3], hp[2], hp[3]] }
        : g.out[p] ? Object.assign({}, g.out[p], { press: g.out[p].press.map(x => x && first) })
          : { move: [0, 0], aim: [1, 0], saim: [1, 0], press: [false, false, false, false] });
      g.es = env.tick(g.es, I.map(x => x.move), I.map(x => x.press[0]), I.map(x => x.aim), I.map(x => x.press[1]),
                      I.map(x => x.saim), I.map(x => x.press[2]), I.map(x => x.press[3]), g.rep.events);
      g.rep.frames.push(snapshot(env, g.es));
      g.k++;
      const s = g.es.s;
      if (s.done) g.rep.result = { winner: s.winner, ticks: s.t, reason: reason(s, G) };
    }

    function run(g) {                           // keep up with real time; fell far behind (a hidden tab): slow down
      if (g.rep.result) return clearInterval(g.timer);
      const due = Math.floor((performance.now() - g.t0) / c.TICK_MS);
      if (due - g.k > 2) g.t0 += (due - g.k - 2) * c.TICK_MS;
      while (!g.rep.result && g.k < Math.floor((performance.now() - g.t0) / c.TICK_MS)) tick(g);
    }

    function input(d) {
      const g = game;
      if (!g || d.id !== g.rep.id) return { gone: true };
      const mv = d.move || [0, 0], L = nrm(mv[0], mv[1]);
      g.move = L > 1 ? [mv[0] / L, mv[1] / L] : mv.slice();
      const aim = d.aim || [0, 0];
      if (nrm(aim[0], aim[1]) > 1e-6) g.aim = aim.slice();
      for (const x of PRESS) g.latch[x] = g.latch[x] || !!d[x];
      return { frames: g.rep.frames.slice(d.k || 0), events: g.rep.events.slice(d.e || 0), result: g.rep.result };
    }

    const setNet = (b, l) => { net = Net(b, G.net.shapes); label = l; };
    return { newGame, input, setNet, env };
  }

  const api = { Sim, Env, Net, heads, LocalPlay, snapshot };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.BrawlSim = api;
})(this);
