// index.html?play: you vs the net. Served and stepped by rl/jplay.py (python rl/jplay.py, port 8766); with no play
// server (GitHub Pages) the sim and the net run in the page (sim.js + web/game.json + web/net.bin, tools/web_export.py).
// WASD move, mouse aim, LMB attack (dash), RMB super, E gadget, Q hypercharge, Enter = new game.
// Touch: a move stick anywhere on the left, ATK / SUPER you drag to aim and release (a tap aims itself: with the ball
// at their goal, else at the nearest enemy you see), GADGET and HYPER taps.
// 3D: fixed game camera, turned 180° when you spawn in the top half (WASD and aim follow the screen).
// Uses index.html's globals (R, list, pos, mode, px, open, draw, ...).
(() => {
  if (!new URLSearchParams(location.search).has('play')) return;
  const KEYS = 'WASD move · mouse aim · LMB attack · RMB super · E gadget · Q hypercharge · Enter new game';
  document.querySelector('.hint').textContent = KEYS;
  document.title = 'Mortis vs a neural net'; document.querySelector('header h1').textContent = 'Mortis vs a neural net';
  $('pick').style.display = 'none';
  const keys = new Set(), latch = { atk: false, sup: false, gad: false, hyp: false };
  let game = null, mc = null, top = false, busy = false, lastF = 0;   // mc: the mouse, client px
  let local = null;                                                    // no play server: BrawlSim.LocalPlay
  let G = null, ver = null, pick = null, vtxt = '';                    // local: game.json, version label, net menu
  let bm = null;                                                       // local, multi: brawler menu (yours vs the net's)
  const bins = {};
  async function prep() {                                              // game.json, the version label, the net menu
    if (G) return;
    G = await fetch('web/game.json').then(r => r.json());
    ver = document.createElement('div');                               // build id (tools/web_publish.sh) + net
    ver.style.cssText = 'position:absolute;left:8px;bottom:6px;font-size:11px;color:var(--dim);pointer-events:none';
    $('stage').appendChild(ver);
    vtxt = (await fetch('web/version.txt', { cache: 'no-store' }).then(r => (r.ok ? r.text() : '')).catch(() => '')).trim();
    ver.textContent = vtxt ? `v${vtxt}` : '';
    if (G.nets.length > 1) {                                           // which net you play; a change mid-game = new game
      pick = document.createElement('select');
      pick.title = 'which net you play against';
      G.nets.forEach(n => pick.add(new Option(n.label)));
      pick.onchange = async () => { pick.blur(); if (!local) return; await useNet(pick.selectedIndex); start(); };
      document.querySelector('header h1').after(pick);
    }
    if (G.multi) {                                                     // brawlers: Random or one of G.kinds, ?me=S&net=J preset
      const q = new URLSearchParams(location.search), t = 'Brawl Ball vs a neural net';
      document.title = t; document.querySelector('header h1').textContent = t;
      bm = document.createElement('span');
      bm.style.cssText = 'display:flex;gap:6px;align-items:center';
      const sel = who => {
        const s = document.createElement('select');
        s.title = who === 'me' ? 'your brawler' : "the net's brawler";
        s.add(new Option('Random', ''));
        for (const c of G.kinds) s.add(new Option(G.labels[G.letters.indexOf(c)], c));
        s.value = (q.get(who) || '').toUpperCase();
        if (s.selectedIndex < 0) s.value = '';
        s.onchange = () => { s.blur(); if (local) start(); };          // a change mid-game = new game
        return s;
      };
      const vs = document.createElement('span');
      vs.textContent = 'vs'; vs.style.color = 'var(--dim)';
      bm.append(sel('me'), vs, sel('net'));
      document.querySelector('header h1').after(bm);
    }
  }
  prep().catch(() => {});
  async function useNet(i) {                                           // local: the next game plays G.nets[i]
    const n = G.nets[i];
    bins[n.file] = bins[n.file] || await fetch('web/' + n.file).then(r => r.arrayBuffer());
    if (local) local.setNet(bins[n.file], n.label); else local = BrawlSim.LocalPlay(G, bins[n.file], n.label);
    ver.textContent = (vtxt ? `v${vtxt} · ` : '') + n.label;
  }
  const in3d = () => mode === '3d' && view3;
  const flip = () => (in3d() && top ? -1 : 1);                         // screen -> world (3D camera turned 180°)
  const toWorld = ([x, y]) => {
    if (in3d()) return view3.toWorld(x, y, R);
    const b = cv.getBoundingClientRect();
    return [(x - b.left) * cv.width / b.width / px * R.tile, (y - b.top) * cv.height / b.height / px * R.tile];
  };
  const mouseW = () => mc && R && toWorld(mc);       // re-projected every time: the 3D camera follows the ball
  const banner = (t, c) => { const b = $('banner'); b.style.display = 'block'; b.textContent = t; b.style.color = c || '#fff'; };
  const isTouch = () => document.body.classList.contains('touch');

  async function newGame() {
    if (!local) {
      try {
        const r = await fetch('/play/new' + location.search, { cache: 'no-store' });   // ?play&me=S&net=J (rl.jplay)
        if (r.ok) { if (pick) pick.remove(); if (bm) bm.remove(); return await r.json(); }
      } catch (e) {}
      await prep();
      await useNet(pick ? pick.selectedIndex : 0);
    }
    const q = new URLSearchParams(location.search), s = bm ? bm.querySelectorAll('select') : null;
    return local.newGame(s ? { me: s[0].value, net: s[1].value } : { me: q.get('me'), net: q.get('net') });
  }

  async function start() {
    let r;
    try { r = await newGame(); }
    catch (e) { banner('no play server and no web/ net: run python rl/jplay.py or tools/web_export.py', '#f87171'); return; }
    list = list.filter(x => !x.play);
    list.unshift({ label: 'PLAY: you vs the net', r, play: true });
    fillPick(); open(0);
    game = r; pos = 0;
    top = MR.unpack(r, 0).players[r.me].y < r.map.rows.length * r.tile / 2;     // you spawn in the top half
  }

  // ---- touch controls
  const css = document.createElement('style');
  css.textContent = `
    #tc { position: absolute; inset: 0; pointer-events: none; display: none; }
    body.touch #tc { display: block; }
    body.touch #stage { touch-action: none; user-select: none; -webkit-user-select: none; align-items: flex-start; }
    body.touch aside, body.touch footer, body.touch .hint, body.touch #mapname { display: none; }
    body.touch main { grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 1fr); }
    #stick { position: absolute; width: 120px; height: 120px; margin: -60px; border-radius: 50%; border: 2px solid #fff6; background: #0003; display: none; }
    #knob { position: absolute; left: 50%; top: 50%; width: 50px; height: 50px; margin: -25px; border-radius: 50%; background: #fff8; }
    .tb { position: absolute; pointer-events: auto; border-radius: 50%; display: flex; align-items: center; justify-content: center;
          font: 700 12px system-ui, sans-serif; color: #fff; background: #0007; border: 3px solid #fff8; }
    .tb.off { opacity: .35; }
    #tb-atk { width: 86px; height: 86px; right: 22px; bottom: 26px; border-color: #f87171; }
    #tb-sup { width: 64px; height: 64px; right: 124px; bottom: 20px; border-color: #facc15; }
    #tb-gad { width: 52px; height: 52px; right: 34px; bottom: 130px; border-color: #4ade80; }
    #tb-hyp { width: 52px; height: 52px; right: 104px; bottom: 104px; border-color: #c084fc; }`;
  document.head.appendChild(css);
  const tc = document.createElement('div');
  tc.id = 'tc';
  tc.innerHTML = '<div id="stick"><div id="knob"></div></div>' +
    [['atk', 'ATK'], ['sup', 'SUPER'], ['gad', 'GAD'], ['hyp', 'HYPER']].map(([k, t]) => `<div class="tb" id="tb-${k}" data-k="${k}">${t}</div>`).join('');
  $('stage').appendChild(tc);
  if (matchMedia('(pointer: coarse)').matches) document.body.classList.add('touch');
  const ptr = new Map();                             // pointerId -> {k: stick | atk | sup, x0, y0, v: unit drag or null}
  let touchAim = null, lastAim = null;

  function autoAim() {                               // the ball carrier: their goal; else the nearest enemy you see
    const F = MR.unpack(R, R.frames.length - 1), me = F.players[R.me];
    if (F.ball && F.ball.carrier === R.me && R.goals) {
      const g = R.goals[1 - R.side];
      return [(g[0] + g[2]) / 2 - me.x, (g[1] + g[3]) / 2 - me.y];
    }
    let best = null, bd = Infinity;
    F.players.forEach((P, q) => {
      const d = Math.hypot(P.x - me.x, P.y - me.y);
      if (R.players[q].team !== R.side && P.hp > 0 && !P.hidden && d < bd) { bd = d; best = [P.x - me.x, P.y - me.y]; }
    });
    return best || lastAim;
  }

  $('stage').addEventListener('pointerdown', e => {
    if (e.pointerType !== 'touch') return;
    e.preventDefault();
    document.body.classList.add('touch'); mc = null;
    if (!game || (game.result && performance.now() - (game.endAt || 0) > 1000)) { start(); return; }
    if (game.result) return;
    const k = e.target.dataset && e.target.dataset.k;
    if (k === 'gad' || k === 'hyp') { latch[k] = true; return; }
    if (!k && [...ptr.values()].some(t => t.k === 'stick')) return;
    ptr.set(e.pointerId, { k: k || 'stick', x0: e.clientX, y0: e.clientY, v: null });
    if (!k) {
      const b = $('stage').getBoundingClientRect(), s = $('stick').style;
      s.left = e.clientX - b.left + 'px'; s.top = e.clientY - b.top + 'px'; s.display = 'block';
      $('knob').style.transform = '';
    }
  });
  addEventListener('pointermove', e => {
    const t = ptr.get(e.pointerId);
    if (!t) return;
    const dx = e.clientX - t.x0, dy = e.clientY - t.y0, L = Math.hypot(dx, dy);
    t.v = L > 10 ? [dx / L, dy / L] : null;
    if (t.k === 'stick') { const m = Math.min(L, 60) / Math.max(L, 1e-6); $('knob').style.transform = `translate(${dx * m}px, ${dy * m}px)`; }
  });
  const up = fire => e => {
    const t = ptr.get(e.pointerId);
    if (!t) return;
    ptr.delete(e.pointerId);
    if (t.k === 'stick') $('stick').style.display = 'none';
    else if (fire && game && !game.result) { touchAim = t.v ? [flip() * t.v[0], flip() * t.v[1]] : autoAim(); latch[t.k] = true; }
  };
  addEventListener('pointerup', up(true));
  addEventListener('pointercancel', up(false));

  function touchUI(F) {                              // dim what you can't use now
    const P = F.players[R.me], off = (k, x) => $('tb-' + k).classList.toggle('off', !x);
    off('atk', P.hp > 0 && P.ammo >= 1); off('sup', P.hp > 0 && P.ulti >= 1);
    off('gad', P.hp > 0 && P.gad > 0); off('hyp', P.hp > 0 && P.hyper >= 1 && P.ulti >= 1);
  }

  // ---- the game
  async function send() {
    if (!game || R !== game || game.result || busy) return;
    busy = true;
    const F = R.frames[R.frames.length - 1].p[R.me];
    const f = flip(), mouse = mouseW(), st = [...ptr.values()].find(t => t.k === 'stick');
    const mv = st ? (st.v ? [f * st.v[0], f * st.v[1]] : [0, 0])                 // WASD go by the screen
      : [f * (keys.has('KeyD') - keys.has('KeyA')), f * (keys.has('KeyS') - keys.has('KeyW'))];
    const aim = touchAim || (mouse ? [mouse[0] - F[0], mouse[1] - F[1]] : [0, 0]);
    if (aim[0] || aim[1]) lastAim = aim;
    const body = { id: game.id, k: R.frames.length, e: R.events.length, move: mv, aim, ...latch };
    for (const k in latch) latch[k] = false;
    touchAim = null;
    try {
      const g = game, r = local ? local.input(body)
        : await (await fetch('/play/input', { method: 'POST', body: JSON.stringify(body) })).json();
      if (g === game && r.gone) g.result = { winner: -2, reason: 'gone' };
      else if (g === game && r.frames) {
        g.frames.push(...r.frames);
        if (r.events.length) { g.events.push(...r.events); drawLog(); drawMarks(); }
        if (r.result) { g.result = r.result; g.endAt = performance.now(); drawLog(); }
        $('seek').max = g.frames.length - 1;
      }
    } catch (e) {}
    busy = false;
  }
  setInterval(send, 40);

  function frame(now) {
    const again = isTouch() ? 'tap' : 'Enter';
    if (game && R === game) {
      const end = R.frames.length - 1;
      pos = Math.min(end, Math.max(end - 2, pos + (now - lastF) / R.tick_ms));   // ~1 tick behind, smooth
      if (view3) view3.playView(top);
      const u = px / R.tile, P = MR.unpack(R, pos).players[R.me], mw = mouseW(), mouse = !in3d() && mw;
      const drag = [...ptr.values()].find(t => t.k !== 'stick' && t.v);
      if (view3) view3.setAim(R, !in3d() || P.hp <= 0 ? null : drag ? [flip() * drag.v[0], flip() * drag.v[1]]   // 3D aim bar
        : !isTouch() && mw ? [mw[0] - P.x, mw[1] - P.y] : null);
      draw();
      const tgt = drag && !in3d() ? [P.x + drag.v[0] * R.consts.dash_range, P.y + drag.v[1] * R.consts.dash_range] : mouse;
      if (tgt && P.hp > 0) {                                                      // aim line (2D)
        ctx.strokeStyle = '#fff8'; ctx.lineWidth = 2; ctx.setLineDash([6, 6]);
        ctx.beginPath(); ctx.moveTo(P.x * u, P.y * u); ctx.lineTo(tgt[0] * u, tgt[1] * u); ctx.stroke(); ctx.setLineDash([]);
      }
      if (isTouch()) touchUI(MR.unpack(R, end));
      if (R.result && R.result.reason === 'gone') banner(`Another tab started a game · ${again} = new game here`, '#fde047');
      else if (R.result) {
        const w = R.result.winner, side = R.side ?? R.me;                       // ball: the winning team
        banner(`${w === side ? 'You win' : w === -1 ? 'Draw' : 'Net wins'} (${R.result.reason}) · ${again} = again`,
               w === side ? '#4ade80' : w === -1 ? '#fff' : '#f87171');
      }
    } else if (!game) banner(isTouch() ? 'Tap: start' : 'Click or Enter: start · ' + KEYS, '#fde047');
    lastF = now;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  const dl = document.createElement('button');                                 // save the game as a replay .json
  dl.textContent = '⬇ replay'; dl.title = 'save this game as a replay .json';
  dl.onclick = () => {
    if (!game) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(game)], { type: 'application/json' }));
    a.download = `mortis_play_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`;
    a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  document.querySelector('header').appendChild(dl);

  addEventListener('mousemove', e => { mc = [e.clientX, e.clientY]; });
  cv.addEventListener('contextmenu', e => e.preventDefault());
  $('stage').addEventListener('contextmenu', e => e.preventDefault());
  $('stage').addEventListener('mousedown', e => {
    e.preventDefault();
    if (!game || game.result) { if (!game) start(); return; }
    mc = [e.clientX, e.clientY];
    if (e.button === 0) latch.atk = true;
    else if (e.button === 2) latch.sup = true;
  });
  addEventListener('keydown', e => {
    if (e.code === 'Enter' && (!game || game.result)) { e.preventDefault(); start(); return; }
    if (e.code === 'KeyE' && !e.repeat) latch.gad = true;
    if (e.code === 'KeyQ' && !e.repeat) latch.hyp = true;
    keys.add(e.code);
  });
  addEventListener('keyup', e => keys.delete(e.code));
  addEventListener('blur', () => { keys.clear(); ptr.clear(); $('stick').style.display = 'none'; });
})();
