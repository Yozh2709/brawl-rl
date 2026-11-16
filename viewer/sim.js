// Mortis 1v1 Brawl Ball in the browser, no server: a port of rl/jball.py (the sim), the parts of rl/jballenv.py the
// net needs (vision memory, obs, button masks, action -> inputs; no bots, no trickshot hint) and the policy MLP.
// tools/web_export.py writes its data (web/game.json, web/net.bin); tools/web_parity.py checks it against JAX.
// LocalPlay mirrors rl/jplay.py's Game, so play.js talks to it like to the server.
(function (root) {
  'use strict';
  const NEIGH = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
  const clip = (x, a, b) => Math.min(Math.max(x, a), b);
  const nrm = (x, y) => Math.sqrt(x * x + y * y);
  const unit = v => { const n = nrm(v[0], v[1]); return n > 1e-9 ? [v[0] / n, v[1] / n] : [0, 0]; };
  const fill = (n, f) => Array.from({ length: n }, (_, i) => f(i));

  function segDist(q, a, b) {
    const abx = b[0] - a[0], aby = b[1] - a[1], L = abx * abx + aby * aby;
    const t = L > 1e-9 ? clip(((q[0] - a[0]) * abx + (q[1] - a[1]) * aby) / Math.max(L, 1e-9), 0, 1) : 0;
    return nrm(q[0] - (a[0] + abx * t), q[1] - (a[1] + aby * t));
  }

  function Sim(G) {
    const c = G.c, P = G.P, H = G.H, W = G.W, TILE = c.TILE, R = c.R, team = G.team;
    const foe = fill(P, a => fill(P, b => team[a] !== team[b]));
    const tile = (g, x, y) => x < 0 || x >= W || y < 0 || y >= H || !!g[y * W + x];
    const cell = v => Math.floor(v / TILE);
    const grids = s => { const o = s.ot ? 1 : 0; return [G.block[o], G.shot[o], G.bush[o]]; };
    const inBush = (bush, p) => !!bush[clip(cell(p[1]), 0, H - 1) * W + clip(cell(p[0]), 0, W - 1)];
    const longReady = (s, p) => s.t - s.last_attack[p] >= c.LONG_TICKS && s.ammo[p] >= c.AMMO;

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
        if (tile(shot, cell(nx[0]), cell(nx[1]))) bd[ax] = -bd[ax]; else bp = nx;
      }
      return [bp, bd];
    }

    function clearShot(a, b, shot) {
      const k = Math.max(2, Math.floor(nrm(b[0] - a[0], b[1] - a[1]) / 50) + 1);
      for (let i = 0; i < c.SPIN_PTS; i++) {
        const f = Math.min(i, k - 1) / (k - 1);
        if (tile(shot, cell(a[0] + (b[0] - a[0]) * f), cell(a[1] + (b[1] - a[1]) * f))) return false;
      }
      return true;
    }

    function visible(s) {                     // v[viewer][target], shared team vision
      const bush = grids(s)[2];
      const shown = s.pos.map((p, q) => !inBush(bush, p) || s.t < s.reveal_until[q] || !s.alive[q]);
      const v = fill(P, a => fill(P, q =>
        (shown[q] || nrm(s.pos[a][0] - s.pos[q][0], s.pos[a][1] - s.pos[q][1]) <= c.BUSH_REVEAL) && s.alive[a]));
      return fill(P, p => fill(P, q => p === q || G.mate[p].some((m, j) => m && v[j][q])));
    }

    function reset() {
      const z = () => fill(P, () => 0), zb = () => fill(P, () => false), z2 = () => fill(P, () => [0, 0]);
      const zpp = () => fill(P, () => fill(P, () => false));
      return {
        t: 0, done: false, winner: -2, pos: G.spawn.map(p => p.slice()), vel: z2(), hp: fill(P, () => c.MAX_HP),
        alive: fill(P, () => true), ammo: fill(P, () => c.AMMO), reload: z(), last_attack: z(),
        last_hurt: fill(P, () => -1e6), last_by: fill(P, () => -1), ulti: z(), reveal_until: z(), dash_rem: z(),
        dash_dir: z2(), dash_hit: zpp(), proj_on: zb(), proj_pos: z2(), proj_dir: z2(), proj_rem: z(), proj_hit: zpp(),
        proj_back: zb(), proj_ret: zb(), hyper: z(), hyper_until: z(), gad_left: fill(P, () => c.GADGET_USES),
        gad_ready: z(), respawn_at: fill(P, () => -1), score: [0, 0], ot: false, freeze_until: 0,
        ball_pos: G.center.slice(), ball_dir: [0, 0], ball_rem: 0, carrier: -1, last_touch: -1, kicker: -1, nopick_until: 0,
      };
    }

    function revive(s, m, at) {
      for (let p = 0; p < P; p++) if (m[p]) {
        s.pos[p] = at[p].slice(); s.hp[p] = c.MAX_HP; s.alive[p] = true; s.ammo[p] = c.AMMO; s.reload[p] = 0;
        s.dash_rem[p] = 0; s.respawn_at[p] = -1; s.reveal_until[p] = 0; s.last_hurt[p] = -1e6;
      }
    }

    function credit(lastBy, hit) {            // the highest attacker wins
      return lastBy.map((v, b) => { let top = -1; for (let a = 0; a < P; a++) if (hit[a][b]) top = a; return top >= 0 ? top : v; });
    }

    // one tick (Ball.step). move/adir/sdir [P][2], atk/sup/gad/hyp [P] bool; ev: [t, kind, a, b, amount] are pushed
    function step(s0, move, atk, adir, sup, sdir, gad, hyp, ev) {
      const T = s0.t, run = !s0.done, act = run && T >= s0.freeze_until;
      const [block, shot] = grids(s0);
      const s = structuredClone(s0);
      const E = (kind, a, b, amt) => ev && ev.push([T, kind, a, b, Math.round(amt)]);
      const back = s.alive.map((al, p) => !al && s.respawn_at[p] >= 0 && T >= s.respawn_at[p] && run);
      revive(s, back, G.respawn);
      back.forEach((m, p) => m && E('respawn', p, p, 0));
      const ar = fill(P, i => i), live = s.alive.map(a => a && act), holds = ar.map(p => p === s.carrier), ctl = live;
      // hypercharge
      let hon = s.hyper_until.map(u => T < u);
      const hy = ar.map(p => ctl[p] && hyp[p] && s.hyper[p] >= 1 && s.ulti[p] >= 1 && !hon[p]);
      const hyper_until = ar.map(p => hy[p] ? T + c.HYPER_TICKS : s.hyper_until[p]);
      let hyper = ar.map(p => hy[p] ? 0 : s.hyper[p]);
      hon = hon.map((h, p) => h || hy[p]);
      hy.forEach((m, p) => m && E('hyper', p, -1, 0));
      const mul = fill(P, a => fill(P, b => (hon[a] ? 1 + c.HYPER_DMG : 1) * (hon[b] ? 1 - c.HYPER_SHIELD : 1)));
      // reload
      const rl = ar.map(p => live[p] && s.ammo[p] < c.AMMO);
      let reload = ar.map(p => s.reload[p] + rl[p]);
      const got = ar.map(p => rl[p] && reload[p] >= c.RELOAD);
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
      // gadget: combo spinner hits every enemy in range with a clear line
      const ok = s.alive;
      const g = ar.map(p => ctl[p] && gad[p] && s.gad_left[p] > 0 && T >= s.gad_ready[p]);
      const sp = fill(P, a => fill(P, b => g[a] && foe[a][b] && ok[b] &&
        nrm(s.pos[a][0] - s.pos[b][0], s.pos[a][1] - s.pos[b][1]) <= c.SPIN_R && clearShot(s.pos[a], s.pos[b], shot)));
      const dmg = fill(P, () => 0), gain = fill(P, () => 0), heal = fill(P, () => 0);
      for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) if (sp[a][b]) {
        dmg[b] += c.SPIN_DMG * mul[a][b]; E('spin_hit', a, b, c.SPIN_DMG * mul[a][b]);
      }
      let last_by = credit(s.last_by, sp);
      let last_attack = ar.map(p => g[p] ? T : s.last_attack[p]);
      reveal = ar.map(p => g[p] ? T + c.REVEAL_TICKS : reveal[p]);
      const gad_left = ar.map(p => s.gad_left[p] - g[p]);
      const gad_ready = ar.map(p => g[p] ? T + c.SPIN_CD : s.gad_ready[p]);
      g.forEach((m, p) => m && E('gadget', p, -1, 0));
      // attack: dash (not with the ball)
      const can = ar.map(p => ctl[p] && atk[p] && !(s.dash_rem[p] > 0) && ammo[p] >= 1 && !holds[p]);
      const reach = ar.map(p => c.DASH_RANGE + (T - last_attack[p] >= c.LONG_TICKS && ammo[p] >= c.AMMO ? c.LONG_EXTRA : 0));
      let dash_rem = ar.map(p => can[p] ? reach[p] : s.dash_rem[p]);
      const dash_dir = ar.map(p => can[p] ? unit(adir[p]) : s.dash_dir[p].slice());
      const dash_hit = fill(P, a => fill(P, b => s.dash_hit[a][b] && !can[a]));
      ammo = ar.map(p => ammo[p] - can[p]);
      last_attack = ar.map(p => can[p] ? T : last_attack[p]);
      reveal = ar.map(p => can[p] ? T + c.REVEAL_TICKS : reveal[p]);
      can.forEach((m, p) => m && E('dash', p, -1, reach[p]));
      // super: bats (not with the ball)
      const su = ar.map(p => ctl[p] && sup[p] && !(dash_rem[p] > 0) && ulti[p] >= 1 && !holds[p] && !s.proj_on[p]);
      let proj_on = ar.map(p => s.proj_on[p] || su[p]);
      let proj_pos = ar.map(p => su[p] ? s.pos[p].slice() : s.proj_pos[p].slice());
      const proj_dir = ar.map(p => su[p] ? unit(sdir[p]) : s.proj_dir[p].slice());
      let proj_rem = ar.map(p => su[p] ? c.SUPER_RANGE : s.proj_rem[p]);
      const proj_hit = fill(P, a => fill(P, b => s.proj_hit[a][b] && !su[a]));
      const proj_back = ar.map(p => su[p] ? hon[p] : s.proj_back[p]);
      let proj_ret = ar.map(p => s.proj_ret[p] && !su[p]);
      ulti = ar.map(p => su[p] ? 0 : ulti[p]);
      last_attack = ar.map(p => su[p] ? T : last_attack[p]);
      reveal = ar.map(p => su[p] ? T + c.REVEAL_TICKS : reveal[p]);
      su.forEach((m, p) => m && E('super', p, -1, 0));
      // movement: dash (fixed direction) or walk, 2 substeps with wall push-out
      const dashing = ar.map(p => live[p] && dash_rem[p] > 0);
      const start = s.pos.map(p => p.slice());
      const dlen = ar.map(p => Math.min(dash_rem[p], c.DASH_STEP));
      const want = ar.map(p => {
        if (!live[p]) return [0, 0];
        if (dashing[p]) return [dash_dir[p][0] * dlen[p], dash_dir[p][1] * dlen[p]];
        const spd = Math.max(c.SPEED, c.MIN_SPEED) * (hon[p] ? 1 + c.HYPER_SPEED : 1), m = Math.max(1, nrm(move[p][0], move[p][1]));
        return [move[p][0] / m * (spd * c.DT), move[p][1] / m * (spd * c.DT)];
      });
      let pos = start.map(p => p.slice());
      for (let i = 0; i < 2; i++) pos = pos.map((p, q) => live[q] ? push([p[0] + want[q][0] / 2, p[1] + want[q][1] / 2], block) : p);
      for (let p = 0; p < P; p++) {
        const made = (pos[p][0] - start[p][0]) * dash_dir[p][0] + (pos[p][1] - start[p][1]) * dash_dir[p][1];
        if (dashing[p]) dash_rem[p] = made < 0.5 * dlen[p] ? 0 : dash_rem[p] - dlen[p];
      }
      const vel = pos.map((p, q) => [p[0] - start[q][0], p[1] - start[q][1]]);
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
          const d = nrm(pos[p][0] - ball_pos[0], pos[p][1] - ball_pos[1]);
          if (s.alive[p] && d <= R + c.BALL_R && !(p === kicker && T < nopick_until) && d < best) { best = d; wp = p; }
        }
        if (wp >= 0) { carrier = last_touch = wp; ball_rem = 0; ball_pos = pos[wp].slice(); E('pick', wp, -1, 0); }
      }
      // dash hits: an enemy circle touches the path walked this tick, once per dash
      for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) {
        const hit = dashing[a] && foe[a][b] && ok[b] && !dash_hit[a][b] && segDist(pos[b], start[a], pos[a]) <= 2 * R + c.DASH_HIT_EXTRA;
        if (!hit) continue;
        dash_hit[a][b] = true; dmg[b] += c.DASH_DMG * mul[a][b]; gain[a] += c.CHARGE_ATK; E('hit', a, b, c.DASH_DMG * mul[a][b]);
        last_by = credit(last_by, fill(P, i => fill(P, j => i === a && j === b)));
      }
      // bats; Blood Boomerang bats turn around at the end and home back on Mortis
      const on = ar.map(p => proj_on[p] && act), ret = ar.map(p => on[p] && proj_ret[p]), ps = proj_pos.map(p => p.slice());
      for (let p = 0; p < P; p++) {
        if (!on[p]) continue;
        const dv = ret[p] ? unit([pos[p][0] - proj_pos[p][0], pos[p][1] - proj_pos[p][1]]) : proj_dir[p];
        const pl = ret[p] ? c.BACK_STEP : Math.min(proj_rem[p], c.PROJ_STEP);
        proj_pos[p] = [proj_pos[p][0] + dv[0] * pl, proj_pos[p][1] + dv[1] * pl];
        if (!ret[p]) proj_rem[p] -= pl;
      }
      const bh = fill(P, a => fill(P, b => on[a] && foe[a][b] && ok[b] && !proj_hit[a][b] &&
        segDist(pos[b], ps[a], proj_pos[a]) <= c.SUPER_RADIUS + R));
      for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) if (bh[a][b]) {
        const amt = c.SUPER_DMG * mul[a][b];
        proj_hit[a][b] = true; dmg[b] += amt; gain[a] += c.CHARGE_SUP; heal[a] += amt * c.LIFESTEAL; E('bats', a, b, amt);
      }
      last_by = credit(last_by, bh);
      for (let p = 0; p < P; p++) {
        if (on[p] && !ret[p] && proj_rem[p] <= 0 && proj_back[p]) { proj_ret[p] = true; proj_hit[p] = proj_hit[p].map(() => false); }
        const home = ret[p] && (nrm(proj_pos[p][0] - pos[p][0], proj_pos[p][1] - pos[p][1]) <= c.BACK_STEP || !s.alive[p]);
        proj_on[p] = proj_on[p] && (proj_rem[p] > 0 || proj_ret[p]) && !home;
        proj_ret[p] = proj_ret[p] && proj_on[p];
      }
      // damage, Creepy Harvest (the killer heals)
      const last_hurt = ar.map(p => dmg[p] > 0 ? T : s.last_hurt[p]);
      reveal = ar.map(p => dmg[p] > 0 ? T + c.REVEAL_TICKS : reveal[p]);
      let hp = ar.map(p => s.hp[p] - dmg[p]);
      const dead = ar.map(p => s.alive[p] && hp[p] <= 0);
      for (let a = 0; a < P; a++) {
        const kills = ar.filter(b => dead[b] && last_by[b] === a).length;
        if (s.alive[a]) heal[a] += kills * c.HARVEST * c.MAX_HP;
        if (live[a] && !dead[a] && heal[a] > 0) { E('heal', a, a, Math.min(c.MAX_HP, hp[a] + heal[a]) - hp[a]); hp[a] = Math.min(c.MAX_HP, hp[a] + heal[a]); }
      }
      // super charge, hypercharge meter, regen out of combat
      ulti = ar.map(p => Math.min(ulti[p] + gain[p], 1));
      hyper = ar.map(p => Math.min(hyper[p] + (hon[p] ? 0 : gain[p] * c.HYPER_MUL), 1));
      for (let p = 0; p < P; p++) {
        const calm = T - Math.max(last_attack[p], last_hurt[p]) - c.REGEN_DELAY_TICKS;
        if (live[p] && !dead[p] && calm >= 0 && calm % (1000 / c.TICK_MS) === 0 && hp[p] < c.MAX_HP)
          hp[p] += Math.min(c.MAX_HP - hp[p], c.MAX_HP * c.REGEN_PER_S);
      }
      // deaths: respawn later, a dead carrier drops the ball where he fell
      hp = ar.map(p => dead[p] ? 0 : hp[p]);
      const respawn_at = ar.map(p => dead[p] ? T + c.RESPAWN_TICKS : s.respawn_at[p]);
      dead.forEach((m, p) => { if (m) { dash_rem[p] = 0; hyper_until[p] = 0; E('death', p, p, 0); } });
      if (carrier >= 0 && dead[carrier]) { E('drop', carrier, -1, 0); carrier = -1; ball_rem = 0; }
      Object.assign(s, {
        pos, vel, hp, alive: ar.map(p => s.alive[p] && !dead[p]), ammo, reload, last_attack, last_hurt, last_by, ulti,
        reveal_until: reveal, dash_rem, dash_dir, dash_hit, proj_on, proj_pos, proj_dir, proj_rem, proj_hit, proj_back,
        proj_ret, hyper, hyper_until, gad_left, gad_ready, respawn_at, ball_pos, ball_dir, ball_rem, carrier,
        last_touch, kicker, nopick_until,
      });
      return end(s, run, act, E);
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
      if (scored && !win) {                     // kickoff: everyone back to spawn, ball to the center, short freeze
        revive(s, fill(P, () => true), G.spawn);
        s.proj_on = s.proj_on.map(() => false); s.proj_ret = s.proj_ret.map(() => false); s.hyper_until = s.hyper_until.map(() => 0);
        s.ball_pos = G.center.slice(); s.ball_rem = 0; s.carrier = -1; s.kicker = -1; s.freeze_until = t + c.KICKOFF_TICKS;
      }
      const go = run && !win, reg = go && !s.ot && t === c.REG_TICKS, lead = reg && score[0] !== score[1];
      if (lead) winner = amax;
      if (reg && !lead) { s.ot = true; E('overtime', -1, -1, 0); }
      const tout = go && !lead && t >= c.MAX_TICKS;
      if (tout) winner = -1;
      s.winner = winner; s.done = s.done || win || lead || tout;
      return s;
    }

    return { G, c, P, tile, grids, inBush, longReady, push, fly, clearShot, visible, reset, step };
  }

  // ---------------------------------------------------------------- the net's view (rl/jballenv.py, hint off)
  function Env(G) {
    const b = Sim(G), c = G.c, P = G.P, TILE = c.TILE, team = G.team, NDIR = c.NDIR, DIRS = G.dirs;
    const ar = Array.from({ length: P }, (_, i) => i);

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

    function reset() {
      const s = b.reset();
      return see({ s, seen_pos: fill(P, () => s.pos.map(x => x.slice())), seen_t: fill(P, () => fill(P, () => 0)),
                   seen_hp: fill(P, () => fill(P, () => c.MAX_HP)), ball_seen: fill(P, () => s.ball_pos.slice()),
                   vis: fill(P, p => fill(P, q => p === q)) });
    }

    function tick(es, move, atk, adir, sup, sdir, gad, hyp, ev) {
      return see(Object.assign({}, es, { s: b.step(es.s, move, atk, adir, sup, sdir, gad, hyp, ev) }));
    }

    function masks(es, p) {                    // legal buttons: none, attack, super, gadget, hypercharge + super
      const s = es.s, holds = p === s.carrier, free = !(s.dash_rem[p] > 0);
      const sup = s.alive[p] && s.ulti[p] >= 1 && (holds || (free && !s.proj_on[p]));
      return [true, s.alive[p] && s.ammo[p] >= 1 && (holds || free), sup, s.alive[p] && s.gad_left[p] > 0 && s.t >= s.gad_ready[p],
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
      const size = G.size, o = [];
      const put = (...xs) => { for (const x of xs) o.push(+x); };
      const cpos = x => k === 0 ? [x[0], x[1]] : [size[0] - x[0], size[1] - x[1]];
      const rel = x => [sg * (x[0] - me[0]) / c.SCALE, sg * (x[1] - me[1]) / c.SCALE];
      const resp = q => s.alive[q] ? 0 : clip((s.respawn_at[q] - t) / c.RESPAWN_TICKS, 0, 1);
      const dist = (x, y) => nrm(x[0] - y[0], x[1] - y[1]) / c.SCALE;
      const car = s.carrier, free = car < 0;
      const calm = t - Math.max(s.last_attack[p], s.last_hurt[p]);
      put(s.alive[p], resp(p), ...G.slot[p], car === p, s.kicker === p && t < s.nopick_until, s.hp[p] / c.MAX_HP,
          (s.ammo[p] + s.reload[p] / c.RELOAD) / c.AMMO);
      for (let i = 0; i <= c.AMMO; i++) put(s.ammo[p] === i);
      put(b.longReady(s, p), clip((t - s.last_attack[p]) / c.LONG_TICKS, 0, 1), s.ulti[p], s.ulti[p] >= 1, s.hyper[p],
          t < s.hyper_until[p], clip((s.hyper_until[p] - t) / 100, 0, 1), s.gad_left[p] / 3, t >= s.gad_ready[p],
          clip((s.gad_ready[p] - t) / c.SPIN_CD, 0, 1), s.dash_rem[p] > 0, s.dash_rem[p] / (c.DASH_RANGE + c.LONG_EXTRA),
          sg * s.dash_dir[p][0], sg * s.dash_dir[p][1], s.proj_on[p], ...rel(s.proj_pos[p]), sg * s.proj_dir[p][0],
          sg * s.proj_dir[p][1], s.proj_ret[p], s.proj_back[p], s.proj_rem[p] / c.SUPER_RANGE, t < s.reveal_until[p],
          b.inBush(bush, me), cpos(me)[0] / size[0], cpos(me)[1] / size[1], sg * s.vel[p][0] / c.DASH_STEP,
          sg * s.vel[p][1] / c.DASH_STEP, clip(calm / 60, 0, 2));
      const sc = k === 0 ? s.score : [s.score[1], s.score[0]], their = G.gc[1 - k], ours = G.gc[k], bs = es.ball_seen[p];
      const ct = team[Math.max(car, 0)];
      put(t / c.MAX_TICKS, clip(t / c.REG_TICKS, 0, 1), s.ot, clip((s.freeze_until - t) / 40, 0, 1), sc[0] / 2, sc[1] / 2,
          ...rel(bs), dist(bs, me), cpos(bs)[0] / size[0], cpos(bs)[1] / size[1], free, !free && ct === k && car !== p,
          !free && ct !== k, free * sg * s.ball_dir[0], free * sg * s.ball_dir[1], free * s.ball_rem / c.SUPER_KICK,
          ...rel(their), dist(their, me), segClear(shot, me, their), ...rel(ours));
      put(0, 0, 0, 0, 0, 0, 0, 0);                                         // trickshot hint: off
      const mates = ar.filter(q => team[q] === k && q !== p), foes = ar.filter(q => team[q] !== k);
      for (const q of mates.slice(0, G.T - 1))
        put(s.alive[q], resp(q), ...rel(s.pos[q]), dist(s.pos[q], me), s.hp[q] / c.MAX_HP, s.ammo[q] / c.AMMO,
            s.ulti[q] >= 1, s.dash_rem[q] > 0, car === q, s.proj_on[q], segClear(shot, me, s.pos[q]));
      for (const q of foes.slice(0, G.T)) {
        const v = es.vis[p][q], sp = es.seen_pos[p][q], r = [sp[0] - me[0], sp[1] - me[1]];
        const L = Math.max(nrm(r[0], r[1]), 1e-6), side = [-r[1] / L * c.R, r[0] / L * c.R];
        const walk = segClear(block, me, sp) && segClear(block, [me[0] + side[0], me[1] + side[1]], [sp[0] + side[0], sp[1] + side[1]])
          && segClear(block, [me[0] - side[0], me[1] - side[1]], [sp[0] - side[0], sp[1] - side[1]]);
        put(s.alive[q], resp(q), v, ...rel(sp), nrm(r[0], r[1]) / c.SCALE, clip((t - es.seen_t[p][q]) / 100, 0, 1),
            es.seen_hp[p][q] / c.MAX_HP, v * sg * s.vel[q][0] / c.DASH_STEP, v * sg * s.vel[q][1] / c.DASH_STEP,
            v && s.dash_rem[q] > 0, v * sg * s.dash_dir[q][0], v * sg * s.dash_dir[q][1], v && s.ulti[q] >= 1,
            v && t < s.hyper_until[q], v && b.inBush(bush, s.pos[q]), v && car === q, s.proj_on[q], ...rel(s.proj_pos[q]),
            sg * s.proj_dir[q][0], sg * s.proj_dir[q][1], s.proj_ret[q], segClear(shot, me, sp), walk);
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
        const d = team[q] !== team[p] && s.alive[q] ? nrm(es.seen_pos[p][q][0] - me[0], es.seen_pos[p][q][1] - me[1]) : Infinity;
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
    const s = es.s, c = env.G.c, T = env.G.T, t = s.t;
    const p = s.pos.map((x, q) => {
      const e = (1 - env.G.team[q]) * T, ra = s.respawn_at[q], r3 = v => Math.round(v * 1000) / 1000;
      return [Math.round(x[0]), Math.round(x[1]), Math.round(s.hp[q]), r3(s.ammo[q] + s.reload[q] / c.RELOAD),
              r3(clip((t - s.last_attack[q]) / c.LONG_TICKS, 0, 1)), +env.b.longReady(s, q), Math.round(s.ulti[q] * 1e4) / 1e4,
              +(s.dash_rem[q] > 0), +!es.vis[e][q], ra >= 0 ? Math.max(0, ra - t) : 0, r3(s.hyper[q]),
              Math.max(0, s.hyper_until[q] - t), s.gad_left[q], 0];
    });
    return { t, p, b: s.proj_pos.map((x, q) => [+s.proj_on[q], Math.round(x[0]), Math.round(x[1])]),
             ball: [Math.round(s.ball_pos[0]), Math.round(s.ball_pos[1]), s.carrier, +(s.carrier < 0 && s.ball_rem > 0)],
             score: s.score.slice(), ot: +s.ot, frz: Math.max(0, s.freeze_until - t) };
  }

  function reason(s, G) {
    if (s.winner < 0) return 'draw';
    if (s.ot) return 'golden goal';
    return Math.max(...s.score) >= G.c.MAX_SCORE ? 'goals' : 'lead at time';
  }

  function LocalPlay(G, buf) {
    const env = Env(G), net = Net(buf, G.net.shapes), c = G.c, P = G.P, NM = c.NDIR + 1, NB = 5;
    const PRESS = ['atk', 'sup', 'gad', 'hyp'], label = `Net (${Math.round(G.net.samples / 1e6)}M)`;
    let game = null, n = 0;

    function newGame() {
      if (game) clearInterval(game.timer);
      const me = 0, side = G.team[me], es = env.reset();
      const rep = structuredClone(G.header);
      rep.players.forEach((pl, p) => { pl.name = p === me ? 'You' : label; });
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

    return { newGame, input, env, net };
  }

  const api = { Sim, Env, Net, heads, LocalPlay, snapshot };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.BrawlSim = api;
})(this);
