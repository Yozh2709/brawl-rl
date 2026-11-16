// Draws one frame of a Mortis replay (duel or brawl ball) on a 2D canvas.
// A pure function of (replay, frame index): the browser viewer and a Remotion composition can share it.
(function (global) {
  const TEAM = ['#3b82f6', '#ef4444'];
  const TEAM_DARK = ['#1d4ed8', '#b91c1c'];
  const TEAM_NAME = ['Blue', 'Red'];

  // unit u: 0..P-1 brawlers, P..2P-1 turrets (turret P+o belongs to player o)
  const NP = R => R.players.length;
  const teamOf = (R, u) => (R.teams ? R.teams[u % NP(R)] : u % NP(R));
  const teamSize = R => (R.teams ? R.teams.filter(k => k === 0).length : 1);
  const brawler = (R, p) => R.players[p % NP(R)].brawler || 'Mortis';
  const mixed = R => new Set(R.players.map((_, p) => brawler(R, p))).size > 1;
  // short name: A / B in a duel, B1..B3 / R1..R3 in an all-Mortis brawl ball, "B Stu" / "R Stu" in a mixed lineup
  const tag = (R, u) => {
    const p = u % NP(R), t = u >= NP(R) ? ' turret' : '';
    if (R.mode !== 'ball') return 'AB'[p] + t;
    return 'BR'[teamOf(R, p)] + (mixed(R) ? ' ' + brawler(R, p) : p % teamSize(R) + 1) + t;
  };
  const letter = (R, p) => (R.mode !== 'ball' ? 'AB'[p] : mixed(R) ? brawler(R, p)[0] : p % teamSize(R) + 1);
  const color = (R, u) => (u < 0 ? '#fff' : TEAM[teamOf(R, u)]);
  const maxHp = (R, p) => R.players[p].hp || R.consts.hp;
  const maxAmmo = (R, p) => R.players[p].ammo || R.consts.ammo;
  const pretty = s => (s ? s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : '—');
  // damage / heal numbers: event kind -> colour
  const NUM = { hit: '#fff', bats: '#d8b4fe', shot_hit: '#fff', spin_hit: '#fff', turret_hit: '#e5e7eb', debris_hit: '#fde68a',
    burn: '#fb923c', heal: '#4ade80' };
  // where unit u is in a raw frame (a turret: from its 'tu' row), or null
  function unitAt(R, f, u) {
    if (u < NP(R)) return f.p[u];
    const t = (f.tu || []).find(r => r[0] === u - NP(R));
    return t ? [t[1], t[2]] : null;
  }
  const JUMP = 600;   // a bigger move in one tick is a teleport (respawn, kickoff): don't interpolate

  // frame i may be fractional: positions are interpolated toward the next tick for smooth playback
  function unpack(R, i) {
    const last = R.frames.length - 1;
    i = Math.max(0, Math.min(last, i));
    const i0 = Math.floor(i), a = i - i0, f = R.frames[i0], g = R.frames[Math.min(last, i0 + 1)];
    const k = R.frame_keys;
    const lerp = (u, v) => u + (v - u) * a;
    const lerp2 = (u, v) => (Math.hypot(v[0] - u[0], v[1] - u[1]) > JUMP ? [u[0], u[1]] : [lerp(u[0], v[0]), lerp(u[1], v[1])]);
    const F = {
      t: lerp(f.t, g.t),
      players: f.p.map((row, p) => {
        const o = Object.fromEntries(k.map((name, j) => [name, row[j]]));
        [o.x, o.y] = lerp2(row, g.p[p]);
        return o;
      }),
      bats: f.b.map((b, p) => {
        const nb = g.b[p], both = b[0] && nb[0];
        return { on: !!b[0], x: both ? lerp(b[1], nb[1]) : b[1], y: both ? lerp(b[2], nb[2]) : b[2] };
      }),
    };
    // kit objects: no interpolation (lists change from tick to tick)
    F.tu = f.tu || []; F.bl = f.bl || []; F.ar = f.ar || []; F.th = f.th || []; F.brk = f.brk || [];
    if (f.ball) {
      const [x, y] = lerp2(f.ball, g.ball);
      F.ball = { x, y, carrier: f.ball[2], fly: !!f.ball[3] };
      F.score = f.score; F.ot = !!f.ot; F.frz = f.frz;
    }
    return F;
  }

  function floor(g, r, q, px) {
    g.fillStyle = (r + q) % 2 ? '#d6c59b' : '#dccca5';
    g.fillRect(q * px, r * px, px, px);
  }

  function buildMap(R, px) {
    const rows = R.map.rows, H = rows.length, W = rows[0].length;
    const c = document.createElement('canvas');
    c.width = W * px; c.height = H * px;
    const g = c.getContext('2d');
    const leg = R.legend;
    for (let r = 0; r < H; r++) for (let q = 0; q < W; q++) {
      const ch = rows[r][q], x = q * px, y = r * px, t = leg[ch];
      floor(g, r, q, px);
      if (ch === 'J') {                        // invisible wall = outside the field (stands around the goals)
        g.fillStyle = '#5b6170'; g.fillRect(x, y, px, px);
        continue;
      }
      if ('1267'.includes(ch)) {               // spawn (solid ring) / respawn (dashed ring)
        const k = ch === '1' || ch === '6' ? 0 : 1;
        g.strokeStyle = TEAM[k]; g.lineWidth = Math.max(1, px * .06); g.globalAlpha = .55;
        if (ch > '5') g.setLineDash([px * .12, px * .1]);
        g.beginPath(); g.arc(x + px / 2, y + px / 2, px * .38, 0, 7); g.stroke(); g.globalAlpha = 1; g.setLineDash([]);
        continue;
      }
      if (!t) {
        if (ch !== '.') { g.fillStyle = '#0006'; g.font = `${px * .5}px sans-serif`; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(ch, x + px / 2, y + px / 2); }
        continue;
      }
      if (t.water) {
        g.fillStyle = '#3b82c4'; g.fillRect(x, y, px, px);
        g.strokeStyle = '#7fb6e6'; g.lineWidth = Math.max(1, px * .05);
        g.beginPath(); g.moveTo(x + px * .2, y + px * .45); g.quadraticCurveTo(x + px * .35, y + px * .3, x + px * .5, y + px * .45); g.quadraticCurveTo(x + px * .65, y + px * .6, x + px * .8, y + px * .45); g.stroke();
      } else if (t.bush) {
        g.fillStyle = '#3f9142'; g.fillRect(x, y, px, px);
        g.fillStyle = '#2f7a33';
        for (const [a, b] of [[.28, .3], [.7, .35], [.45, .7], [.8, .78], [.15, .8]]) { g.beginPath(); g.arc(x + a * px, y + b * px, px * .16, 0, 7); g.fill(); }
      } else if (t.name === 'RopeFence') {
        g.strokeStyle = '#8b5a2b'; g.lineWidth = Math.max(1, px * .08);
        g.beginPath(); g.moveTo(x, y + px / 2); g.lineTo(x + px, y + px / 2); g.stroke();
        g.fillStyle = '#6b4423'; g.beginPath(); g.arc(x + px / 2, y + px / 2, px * .12, 0, 7); g.fill();
      } else if (t.name === 'PayloadTrack') {
        g.fillStyle = '#cbb98c'; g.fillRect(x, y, px, px);   // walkable path decor
      } else if (t.name === 'Fast') {
        g.fillStyle = '#e9cf63'; g.fillRect(x + px * .1, y + px * .1, px * .8, px * .8);
      } else if (t.name === 'Barrel') {
        g.fillStyle = '#0003'; g.beginPath(); g.ellipse(x + px / 2, y + px * .82, px * .4, px * .14, 0, 0, 7); g.fill();
        g.fillStyle = '#9a5b2e'; g.beginPath(); g.arc(x + px / 2, y + px / 2, px * .42, 0, 7); g.fill();
        g.strokeStyle = '#5e3518'; g.lineWidth = Math.max(1, px * .07);
        g.beginPath(); g.arc(x + px / 2, y + px / 2, px * .28, 0, 7); g.stroke();
      } else if (t.move) {
        const ind = t.name === 'Indestructible';
        g.fillStyle = ind ? '#4b5060' : '#7a5a3c'; g.fillRect(x, y, px, px);
        g.fillStyle = ind ? '#6b7082' : '#9c7a54'; g.fillRect(x + 1, y + 1, px - 2, px * .62);
        g.fillStyle = '#0003'; g.fillRect(x, y + px * .9, px, px * .1);
      }
    }
    // goals: box team k defends, in its color, with a net
    (R.goals || []).forEach((b, k) => {
      const u = px / R.tile, x0 = b[0] * u, y0 = b[1] * u, w = (b[2] - b[0]) * u, h = (b[3] - b[1]) * u;
      g.fillStyle = TEAM[k]; g.globalAlpha = .18; g.fillRect(x0, y0, w, h); g.globalAlpha = .45;
      g.strokeStyle = '#fff'; g.lineWidth = 1;
      for (let s = px / 2; s < w; s += px / 2) { g.beginPath(); g.moveTo(x0 + s, y0); g.lineTo(x0 + s, y0 + h); g.stroke(); }
      for (let s = px / 2; s < h; s += px / 2) { g.beginPath(); g.moveTo(x0, y0 + s); g.lineTo(x0 + w, y0 + s); g.stroke(); }
      g.globalAlpha = 1; g.strokeStyle = TEAM[k]; g.lineWidth = Math.max(2, px * .12); g.strokeRect(x0, y0, w, h);
    });
    return c;
  }

  // an event logged during tick e[0] is visible from frame e[0] + 1
  function recentEvents(R, t, span) {
    const out = [];
    for (const e of R.events) if (e[0] + 1 <= t && e[0] + 1 > t - span) out.push(e);
    return out;
  }

  function clockText(R, t) {
    const c = R.consts, left = c.regulation_ticks ? (t < c.regulation_ticks ? c.regulation_ticks - t : c.max_ticks - t) : t;
    const s = Math.max(0, left) * R.tick_ms / 1000;
    return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  }

  // score + clock box, (x, y) = top-left, s = size unit in px
  function drawScore(ctx, R, F, x, y, s) {
    if (!F.score) return;
    const w = s * 6.4, h = s * 1.5;
    ctx.fillStyle = '#000b'; ctx.beginPath(); ctx.roundRect(x, y, w, h, s * .3); ctx.fill();
    ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
    ctx.font = `bold ${s * 1.05}px sans-serif`;
    ctx.fillStyle = TEAM[0]; ctx.fillText(F.score[0], x + s * .8, y + h / 2 + 1);
    ctx.fillStyle = TEAM[1]; ctx.fillText(F.score[1], x + w - s * .8, y + h / 2 + 1);
    ctx.font = `bold ${s * (F.ot ? .5 : .75)}px ui-monospace, Consolas, monospace`;
    ctx.fillStyle = F.ot ? '#fde047' : '#fff';
    ctx.fillText(F.ot ? 'OVERTIME ' + clockText(R, F.t) : clockText(R, F.t), x + w / 2, y + h / 2 + 1);
  }

  function drawBall(ctx, x, y, r) {
    ctx.fillStyle = '#0005'; ctx.beginPath(); ctx.ellipse(x, y + r * .8, r * .9, r * .35, 0, 0, 7); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.strokeStyle = '#222'; ctx.lineWidth = Math.max(1.5, r * .14);
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#222'; ctx.beginPath();
    for (let k = 0; k < 5; k++) { const a = -Math.PI / 2 + k * 2 * Math.PI / 5; ctx.lineTo(x + Math.cos(a) * r * .38, y + Math.sin(a) * r * .38); }
    ctx.fill();
  }

  function drawFrame(ctx, R, i, o) {
    const px = o.px, u = px / R.tile;
    const F = unpack(R, i);
    ctx.drawImage(o.map, 0, 0);
    const rad = R.consts.radius * u;
    if (F.ot) for (const [r, q] of R.ot_break || []) floor(ctx, r, q, px);   // walls broken by overtime
    for (const [q, r] of F.brk) floor(ctx, r, q, px);                         // walls broken by Stu's Breakthrough

    // Stu's fire trail / speed zone
    for (const [kind, ax, ay, ar, tm] of F.ar) {
      const fire = kind === 'fire';
      ctx.fillStyle = fire ? 'rgba(249,115,22,.35)' : 'rgba(250,204,21,.14)';
      ctx.strokeStyle = fire ? 'rgba(234,88,12,.6)' : TEAM[tm]; ctx.lineWidth = fire ? 1 : 2;
      if (!fire) ctx.setLineDash([6, 5]);
      ctx.beginPath(); ctx.arc(ax * u, ay * u, ar * u, 0, 7); ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
    }
    // Jessie's turrets: a thrown one lands on the dashed ring
    for (const [o, , , tx, ty] of F.th) {
      ctx.strokeStyle = color(R, o); ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.arc(tx * u, ty * u, px * .45, 0, 7); ctx.stroke(); ctx.setLineDash([]);
    }
    for (const [o, tx, ty, hp, mx] of F.tu) {
      const x = tx * u, y = ty * u, s = px * .36, col = color(R, o);
      ctx.fillStyle = '#0004'; ctx.beginPath(); ctx.ellipse(x, y + s * .9, s * 1.1, s * .4, 0, 0, 7); ctx.fill();
      ctx.fillStyle = TEAM_DARK[teamOf(R, o)]; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.roundRect(x - s, y - s, 2 * s, 2 * s, s * .3); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = `bold ${s * 1.2}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('T', x, y + 1);
      const bw = px, bh = Math.max(3, px * .1), bx = x - bw / 2, by = y - s - bh - 3;
      ctx.fillStyle = '#000a'; ctx.fillRect(bx - 1, by - 1, bw + 2, bh + 2);
      ctx.fillStyle = col; ctx.fillRect(bx, by, bw * Math.max(0, hp) / mx, bh);
    }

    // bat swarms
    F.bats.forEach((b, p) => {
      if (!b.on) return;
      const x = b.x * u, y = b.y * u, r = R.consts.super_radius * u;
      ctx.fillStyle = 'rgba(126,34,206,.22)'; ctx.strokeStyle = 'rgba(126,34,206,.7)'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#2e1065';
      for (let k = 0; k < 9; k++) {
        const a = k * 0.7 + F.t * 0.5, rr = r * (0.25 + 0.6 * ((k * 37) % 10) / 10);
        ctx.beginPath(); ctx.ellipse(x + Math.cos(a) * rr, y + Math.sin(a) * rr, px * .14, px * .07, a, 0, 7); ctx.fill();
      }
      ctx.strokeStyle = color(R, p); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
    });

    // a free ball, under the players; a flying one with a trail
    const B = F.ball, br = R.consts.ball_radius ? R.consts.ball_radius * u * .7 : 0;
    if (B && B.carrier < 0) {
      if (B.fly) {
        const g = R.frames[Math.max(0, Math.floor(i) - 3)].ball;
        ctx.strokeStyle = '#fff'; ctx.globalAlpha = .35; ctx.lineCap = 'round'; ctx.lineWidth = br * 1.2;
        ctx.beginPath(); ctx.moveTo(g[0] * u, g[1] * u); ctx.lineTo(B.x * u, B.y * u); ctx.stroke(); ctx.globalAlpha = 1; ctx.lineCap = 'butt';
      }
      drawBall(ctx, B.x * u, B.y * u, br);
    }

    F.players.forEach((P, p) => {
      const fly = P.fx & 4 ? rad * .7 : 0;                  // in the air: drawn above its shadow
      const x = P.x * u, y = P.y * u - fly, col = color(R, p);
      if (P.hidden && R.me != null && teamOf(R, p) !== teamOf(R, R.me)) return;   // play mode: fog of war
      if (P.hp <= 0 && i > 0) {
        ctx.strokeStyle = '#111'; ctx.lineWidth = 3;
        const s = rad * .8;
        ctx.beginPath(); ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s); ctx.moveTo(x + s, y - s); ctx.lineTo(x - s, y + s); ctx.stroke();
        if (P.resp > 0) {                      // respawn countdown
          ctx.font = `bold ${Math.max(10, px * .38)}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
          ctx.lineWidth = 3; ctx.strokeStyle = '#000'; ctx.strokeText(Math.ceil(P.resp * R.tick_ms / 1000), x, y - rad);
          ctx.fillStyle = col; ctx.fillText(Math.ceil(P.resp * R.tick_ms / 1000), x, y - rad);
        }
        return;
      }
      // dash trail: back to where the dash started
      if (P.dash) {
        let j = Math.floor(i); while (j > 0 && R.frames[j - 1].p[p][7]) j--;
        const s = R.frames[Math.max(0, j - 1)].p[p], sx = s[0], sy = s[1];
        ctx.strokeStyle = col; ctx.globalAlpha = .3; ctx.lineCap = 'round'; ctx.lineWidth = rad * 2;
        ctx.beginPath(); ctx.moveTo(sx * u, sy * u); ctx.lineTo(x, y); ctx.stroke(); ctx.globalAlpha = 1; ctx.lineCap = 'butt';
      }
      ctx.globalAlpha = P.hidden ? .45 : 1;
      ctx.fillStyle = '#0004'; ctx.beginPath(); ctx.ellipse(x, y + fly + rad * .7, rad * .9, rad * .4, 0, 0, 7); ctx.fill();
      if (P.hon > 0) { ctx.shadowColor = '#c026d3'; ctx.shadowBlur = 14; }   // hypercharged
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, rad, 0, 7); ctx.fill(); ctx.shadowBlur = 0;
      ctx.strokeStyle = P.hon > 0 ? '#e879f9' : P.fx & 1 ? '#fb923c' : P.fx & 2 ? '#7dd3fc' : '#fff';   // hyper / burning / slowed
      ctx.lineWidth = P.hon > 0 || P.fx & 3 ? 3 : 2;
      if (P.hidden) ctx.setLineDash([4, 3]);
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#fff'; ctx.font = `bold ${rad * 1.1}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(letter(R, p), x, y + 1);
      // super charge ring
      if (P.ulti > 0) {
        const full = P.ulti >= 1;
        ctx.strokeStyle = full ? '#fde047' : '#eab308'; ctx.lineWidth = full ? 4 : 3;
        if (full) { ctx.shadowColor = '#fde047'; ctx.shadowBlur = 8 + 4 * Math.sin(F.t * .4); }
        ctx.beginPath(); ctx.arc(x, y, rad + 4, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * Math.min(1, P.ulti)); ctx.stroke();
        ctx.shadowBlur = 0;
      }
      if (B && B.carrier === p) drawBall(ctx, x + rad * .8, y + rad * .55, br * .8);
      ctx.globalAlpha = 1;
      // hp bar + number
      const bw = px * 1.5, bh = Math.max(4, px * .14), bx = x - bw / 2, by = y - rad - px * .42;
      ctx.fillStyle = '#000a'; ctx.fillRect(bx - 1, by - 1, bw + 2, bh + 2);
      ctx.fillStyle = col; ctx.fillRect(bx, by, bw * Math.max(0, P.hp) / maxHp(R, p), bh);
      ctx.font = `bold ${Math.max(9, px * .3)}px sans-serif`; ctx.textBaseline = 'bottom';
      ctx.lineWidth = 3; ctx.strokeStyle = '#000'; ctx.strokeText(Math.round(P.hp), x, by - 1);
      ctx.fillStyle = '#fff'; ctx.fillText(Math.round(P.hp), x, by - 1);
      // ammo (segments, partial = reloading), Mortis' long-dash bar, hypercharge meter
      const n = maxAmmo(R, p), sw = bw / n, ay = y + rad + px * .12, ah = Math.max(3, px * .1);
      for (let k = 0; k < n; k++) {
        const fill = Math.max(0, Math.min(1, P.ammo - k));
        ctx.fillStyle = '#000a'; ctx.fillRect(bx + k * sw + 1, ay, sw - 2, ah);
        ctx.fillStyle = fill >= 1 ? '#f97316' : '#fdba74'; ctx.fillRect(bx + k * sw + 1, ay, (sw - 2) * fill, ah);
      }
      let ly = ay + ah + 2;
      if (brawler(R, p) === 'Mortis') {
        ctx.fillStyle = '#000a'; ctx.fillRect(bx + 1, ly, bw - 2, ah * .7);
        ctx.fillStyle = P.long ? '#fde047' : '#a16207'; ctx.fillRect(bx + 1, ly, (bw - 2) * P.bar, ah * .7);
        ly += ah * .7 + 2;
      }
      if (P.hyper > 0 || P.hon > 0) {
        ctx.fillStyle = '#000a'; ctx.fillRect(bx + 1, ly, bw - 2, ah * .7);
        ctx.fillStyle = P.hon > 0 || P.hyper >= 1 ? '#e879f9' : '#a21caf';
        ctx.fillRect(bx + 1, ly, (bw - 2) * Math.min(1, P.hon > 0 ? 1 : P.hyper), ah * .7);
      }
    });

    // shots in flight, then turrets still in the air
    const BUL = { s: ['#fb923c', .16], o: ['#bef264', .2], t: ['#e5e7eb', .11], d: ['#a8a29e', .14] };
    for (const [bx, by, kind, tm] of F.bl) {
      const [c, r] = BUL[kind] || ['#fff', .12];
      ctx.fillStyle = c; ctx.strokeStyle = TEAM_DARK[tm]; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(bx * u, by * u, px * r, 0, 7); ctx.fill(); ctx.stroke();
    }
    for (const [o, x, y] of F.th) {
      const s = px * .3;
      ctx.fillStyle = TEAM_DARK[teamOf(R, o)]; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.roundRect(x * u - s, y * u - s - px * .5, 2 * s, 2 * s, s * .3); ctx.fill(); ctx.stroke();
    }

    // damage / heal numbers
    for (const e of recentEvents(R, F.t, 14)) {
      const [et, kind, a, b, amt] = e;
      if (!NUM[kind] || !amt) continue;
      const at = unitAt(R, R.frames[Math.min(R.frames.length - 1, et + 1)], b);
      if (!at) continue;
      const age = F.t - et - 1, x = at[0] * u, y = at[1] * u - rad - px * .9 - age * px * .05;
      ctx.globalAlpha = 1 - age / 14;
      ctx.font = `bold ${Math.max(11, px * .42)}px sans-serif`; ctx.textAlign = 'center';
      const s = (kind === 'heal' ? '+' : '-') + amt, dx = kind === 'heal' ? px * .6 : kind === 'burn' ? -px * .6 : 0;
      ctx.lineWidth = 3; ctx.strokeStyle = '#000'; ctx.strokeText(s, x + dx, y);
      ctx.fillStyle = NUM[kind]; ctx.fillText(s, x + dx, y);
      ctx.globalAlpha = 1;
    }

    if (F.score) {
      drawScore(ctx, R, F, px * .3, px * .3, px * .9);
      banner(ctx, R, F, ctx.canvas.width / 2, ctx.canvas.height / 2, px);
    }
  }

  // big GOAL! / OVERTIME text in the middle of the screen
  function banner(ctx, R, F, x, y, s) {
    let text = null, col = '#fff';
    for (const e of recentEvents(R, F.t, 40)) {
      if (e[1] === 'goal') { text = 'GOAL!'; col = TEAM[e[3]]; }
      if (e[1] === 'overtime') { text = 'OVERTIME'; col = '#fde047'; }
    }
    if (!text) return;
    ctx.font = `900 ${s * 2.2}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.lineWidth = s * .35; ctx.strokeStyle = '#000'; ctx.strokeText(text, x, y);
    ctx.fillStyle = col; ctx.fillText(text, x, y);
  }

  global.MortisRender = { TEAM, TEAM_DARK, TEAM_NAME, NUM, teamOf, tag, brawler, color, maxHp, maxAmmo, pretty, unitAt, unpack,
    buildMap, drawFrame, drawScore, drawBall, banner, clockText };
})(window);
