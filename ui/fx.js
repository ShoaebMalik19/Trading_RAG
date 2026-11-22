/* Trading RAG motion layer. Injected once into the Streamlit page (parent document). */
(function () {
  const W = window, D = document;
  const CFG = W.__SMC_CFG || {};
  const reduce = W.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const touch = W.matchMedia('(pointer: coarse)').matches;
  const C = { up: '#00e59b', dn: '#ff3b5c', amber: '#ffb020', cyan: '#39d0ff' };

  if (W.__smcFxReady) { W.__smcFxEnhance && W.__smcFxEnhance(); return; }
  W.__smcFxReady = true;

  /* ---------------- boot sequence ---------------- */
  function boot() {
    if (!CFG.first || reduce) { D.body.classList.add('smc-booted'); return; }
    const el = D.createElement('div');
    el.id = 'smc-boot';
    el.innerHTML = '<div class="half top"></div><div class="half bot"></div><div class="seam"></div>' +
      '<div class="inner"><div class="term"></div><div class="logo">SMC<span>//</span>RAG</div>' +
      '<div class="pbar"><i></i></div></div><div class="skip">CLICK TO SKIP</div>';
    D.body.appendChild(el);
    D.body.classList.add('smc-booted'); // boot overlay now covers; lift the CSS cover
    const term = el.querySelector('.term'), logo = el.querySelector('.logo'), bar = el.querySelector('.pbar i');
    const lines = [
      ['> mounting knowledge base', CFG.notes + ' strategy notes'],
      ['> chunking markdown', CFG.chunks + ' chunks'],
      ['> embedding model', CFG.embed || 'local'],
      ['> chroma vector index', 'online'],
      ['> generator', CFG.live ? (CFG.llm + ' · live') : 'awaiting api key'],
    ];
    let done = false, t = [];
    const finish = () => {
      if (done) return; done = true; t.forEach(clearTimeout);
      el.classList.add('open');
      setTimeout(() => el.remove(), 1400);
    };
    el.addEventListener('click', finish);
    requestAnimationFrame(() => { bar.style.width = '100%'; });
    lines.forEach((l, i) => t.push(setTimeout(() => {
      const row = D.createElement('div');
      row.innerHTML = l[0] + ' <span style="color:#3a4450">' + '.'.repeat(Math.max(2, 30 - l[0].length)) + '</span> <span class="ok">' + l[1] + '</span>';
      term.appendChild(row);
    }, 120 + i * 230)));
    t.push(setTimeout(() => logo.classList.add('on'), 1350));
    t.push(setTimeout(finish, 2250));
  }

  /* ---------------- live candle simulator ---------------- */
  function chart() {
    const cv = D.createElement('canvas');
    cv.id = 'smc-bg';
    D.body.prepend(cv);
    const ctx = cv.getContext('2d');
    let w = 0, h = 0;
    function size() {
      const dpr = Math.min(W.devicePixelRatio || 1, 2);
      w = W.innerWidth; h = W.innerHeight;
      cv.width = w * dpr; cv.height = h * dpr;
      cv.style.width = w + 'px'; cv.style.height = h + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    size(); W.addEventListener('resize', size);

    const CW = w < 700 ? 10 : 14, DUR = 620;
    let id = 0, trend = 1, candles = [], events = [], lastBos = -99;
    let lo = 90, hi = 110, tLo = 90, tHi = 110;

    function gen(o) {
      if (Math.random() < 0.045) trend *= -1;
      let body = (Math.random() - 0.5 + trend * 0.2) * 2.0;
      if (Math.random() < 0.08) body *= 3.4;            // impulse candle
      let c = o + body;
      let hh = Math.max(o, c) + Math.random() * 0.9;
      let ll = Math.min(o, c) - Math.random() * 0.9;
      const k = { id: id++, o, c, h: hh, l: ll };
      const n = candles.length;
      if (n > 14) {
        const recent = candles.slice(-14, -1);
        // liquidity grab: wick through equal lows / highs, close back inside
        if (Math.random() < 0.035 && !events.some(e => e.t === 'liq' && e.id > k.id - 12)) {
          if (trend > 0) {
            const L = Math.min(...recent.map(x => x.l));
            k.l = L - 0.9 - Math.random(); k.c = Math.max(k.o, L + 0.8 + Math.random()); k.h = Math.max(k.h, k.c + 0.3);
            events.push({ t: 'liq', id: k.id, y: L, dir: 1, born: performance.now() });
          } else {
            const H = Math.max(...recent.map(x => x.h));
            k.h = H + 0.9 + Math.random(); k.c = Math.min(k.o, H - 0.8 - Math.random()); k.l = Math.min(k.l, k.c - 0.3);
            events.push({ t: 'liq', id: k.id, y: H, dir: -1, born: performance.now() });
          }
        }
        // break of structure
        const swing = candles.slice(-26, -4);
        if (swing.length > 6 && k.id - lastBos > 14) {
          let sh = swing[0], sl = swing[0];
          swing.forEach(x => { if (x.h > sh.h) sh = x; if (x.l < sl.l) sl = x; });
          if (k.c > sh.h) { events.push({ t: 'bos', id: k.id, from: sh.id, y: sh.h, dir: 1, born: performance.now() }); lastBos = k.id; }
          else if (k.c < sl.l) { events.push({ t: 'bos', id: k.id, from: sl.id, y: sl.l, dir: -1, born: performance.now() }); lastBos = k.id; }
        }
        // fair value gap (3-candle imbalance)
        const a = candles[n - 2];
        const fvgs = events.filter(e => e.t === 'fvg' && e.id > k.id - 30).length;
        if (fvgs < 2 && a && a.h < k.l - 0.6) events.push({ t: 'fvg', id: a.id, top: k.l, bot: a.h, dir: 1, born: performance.now() });
        else if (fvgs < 2 && a && a.l > k.h + 0.6) events.push({ t: 'fvg', id: a.id, top: a.l, bot: k.h, dir: -1, born: performance.now() });
      }
      return k;
    }
    // seed history
    let p = 100;
    const slots = Math.ceil(W.innerWidth / CW) + 6;
    for (let i = 0; i < slots; i++) { const k = gen(p); candles.push(k); p = k.c; }
    events = events.filter(e => e.id > candles[0].id + 4).map(e => (e.born = -1e9, e));
    let forming = gen(p), t0 = performance.now();

    const ease = x => 1 - Math.pow(1 - x, 3);
    const Y = v => h * 0.16 + (hi - v) / (hi - lo) * h * 0.66;
    const rightPad = () => (w < 700 ? 40 : 150);

    function draw(now) {
      let prog = Math.min(1, (now - t0) / DUR);
      if (prog >= 1) {
        candles.push(forming);
        const maxN = Math.ceil(w / CW) + 8;
        while (candles.length > maxN) candles.shift();
        events = events.filter(e => e.id >= candles[0].id - 2);
        forming = gen(forming.c); t0 = now; prog = 0;
      }
      const e = ease(prog);
      const live = { ...forming };
      live.c = forming.o + (forming.c - forming.o) * e + Math.sin(now / 90) * 0.12 * (1 - e);
      live.h = Math.max(live.o, live.c) + (forming.h - Math.max(forming.o, forming.c)) * e;
      live.l = Math.min(live.o, live.c) - (Math.min(forming.o, forming.c) - forming.l) * e;
      const all = candles.concat([live]);

      // auto-range (smoothed)
      tLo = Math.min(...all.map(k => k.l)); tHi = Math.max(...all.map(k => k.h));
      const pad = (tHi - tLo) * 0.08 + 0.5;
      lo += ((tLo - pad) - lo) * 0.06; hi += ((tHi + pad) - hi) * 0.06;

      const n = all.length, xr = w - rightPad();
      const X = i => xr - (n - 1 - i) * CW - prog * CW;
      const idx = {}; all.forEach((k, i) => idx[k.id] = i);

      ctx.clearRect(0, 0, w, h);
      // grid
      ctx.strokeStyle = 'rgba(120,140,160,0.06)'; ctx.lineWidth = 1;
      for (let gx = (-(now / 40) % 80); gx < w; gx += 80) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke(); }
      for (let gy = 40; gy < h; gy += 80) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke(); }

      ctx.font = '600 10px JetBrains Mono, monospace';
      // events (behind candles)
      events.forEach(ev => {
        const i = idx[ev.id]; if (i === undefined) return;
        const age = Math.min(1, (now - ev.born) / 500);
        ctx.globalAlpha = Math.max(0.15, age);
        if (ev.t === 'fvg') {
          const x0 = X(i), y0 = Y(ev.top), y1 = Y(ev.bot), x1 = Math.min(xr + 10, x0 + CW * 18);
          const col = 'rgba(255,176,32,';
          ctx.fillStyle = col + '0.10)'; ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
          ctx.strokeStyle = col + '0.4)'; ctx.setLineDash([2, 3]); ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.setLineDash([]);
          ctx.fillStyle = C.amber; ctx.fillText('FVG', x0 + 3, y0 - 4);
        } else if (ev.t === 'bos') {
          const fi = idx[ev.from] !== undefined ? idx[ev.from] : 0;
          const xa = X(fi), xb = X(i) + CW * 2, y = Y(ev.y);
          const col = ev.dir > 0 ? C.up : C.dn;
          ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.setLineDash([6, 4]);
          ctx.beginPath(); ctx.moveTo(xa, y); ctx.lineTo(xa + (xb - xa) * age, y); ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = col; ctx.fillText(ev.dir > 0 ? 'BOS ▲' : 'BOS ▼', (xa + xb) / 2 - 14, y + (ev.dir > 0 ? -6 : 14));
        } else if (ev.t === 'liq') {
          const x1 = X(i), y = Y(ev.y);
          ctx.strokeStyle = C.cyan; ctx.lineWidth = 1; ctx.setLineDash([1, 3]);
          ctx.beginPath(); ctx.moveTo(x1 - CW * 12, y); ctx.lineTo(x1 + CW, y); ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = C.cyan; ctx.fillText('$$$ SWEPT', x1 - CW * 12, y + (ev.dir > 0 ? 14 : -6));
          const r = 4 + (1 - age) * 14; ctx.beginPath(); ctx.arc(x1 + CW * 0.3, ev.dir > 0 ? Y(candles[i] ? candles[i].l : ev.y) : Y(candles[i] ? candles[i].h : ev.y), r, 0, 7);
          ctx.strokeStyle = 'rgba(57,208,255,' + (0.8 - age * 0.5) + ')'; ctx.stroke();
        }
        ctx.globalAlpha = 1;
      });

      // candles
      all.forEach((k, i) => {
        const x = X(i), up = k.c >= k.o, col = up ? C.up : C.dn, isLive = i === n - 1;
        ctx.globalAlpha = isLive ? 1 : 0.62;
        ctx.strokeStyle = col; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x + CW * 0.3, Y(k.h)); ctx.lineTo(x + CW * 0.3, Y(k.l)); ctx.stroke();
        const yt = Y(Math.max(k.o, k.c)), yb = Y(Math.min(k.o, k.c));
        if (isLive) { ctx.shadowColor = col; ctx.shadowBlur = 18; }
        ctx.fillStyle = up ? 'rgba(0,229,155,0.85)' : 'rgba(255,59,92,0.85)';
        ctx.fillRect(x, yt, CW * 0.6, Math.max(1, yb - yt));
        ctx.shadowBlur = 0; ctx.globalAlpha = 1;
      });

      // live price line
      const yl = Y(live.c), colL = live.c >= live.o ? C.up : C.dn;
      ctx.strokeStyle = colL; ctx.globalAlpha = 0.5; ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(0, yl); ctx.lineTo(w, yl); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      if (w >= 700) {
        ctx.fillStyle = colL; ctx.fillRect(w - 92, yl - 10, 80, 20);
        ctx.fillStyle = '#05070a'; ctx.font = '700 10px JetBrains Mono, monospace'; ctx.fillText('SIM · LIVE', w - 86, yl + 4);
      }
      if (!reduce) W.requestAnimationFrame(draw);
    }
    W.requestAnimationFrame(draw);

    // subtle parallax
    if (!touch && !reduce) W.addEventListener('mousemove', ev => {
      const dx = (ev.clientX / w - 0.5) * -14, dy = (ev.clientY / h - 0.5) * -10;
      cv.style.transform = 'translate3d(' + dx + 'px,' + dy + 'px,0) scale(1.03)';
    });
  }

  /* ---------------- cursor glow + ring ---------------- */
  function cursor() {
    if (touch || reduce) return;
    const g = D.createElement('div'); g.id = 'smc-glow';
    const r = D.createElement('div'); r.id = 'smc-ring';
    D.body.append(g, r);
    let mx = -999, my = -999, gx = mx, gy = my, rx = mx, ry = my;
    W.addEventListener('mousemove', e => {
      mx = e.clientX; my = e.clientY; g.style.opacity = 1; r.style.opacity = 1;
      const hot = e.target.closest && e.target.closest('button, a, [role="button"], textarea, input');
      r.classList.toggle('hot', !!hot);
    });
    D.addEventListener('mouseleave', () => { g.style.opacity = 0; r.style.opacity = 0; });
    (function loop() {
      gx += (mx - gx) * 0.08; gy += (my - gy) * 0.08; rx += (mx - rx) * 0.28; ry += (my - ry) * 0.28;
      g.style.transform = 'translate3d(' + gx + 'px,' + gy + 'px,0)';
      r.style.transform = 'translate3d(' + rx + 'px,' + ry + 'px,0)';
      W.requestAnimationFrame(loop);
    })();
  }

  /* ---------------- per-render enhancements ---------------- */
  const SCR = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$#%▲▼◆';
  function scramble(el) {
    if (el.__busy) return; el.__busy = true;
    const final = el.dataset.text || el.textContent; el.dataset.text = final;
    let f = 0; const total = 18;
    (function tick() {
      el.textContent = final.split('').map((ch, i) => ch === ' ' ? ' ' :
        (i < (f / total) * final.length ? ch : SCR[(Math.random() * SCR.length) | 0])).join('');
      if (++f <= total) W.requestAnimationFrame(tick); else { el.textContent = final; el.__busy = false; }
    })();
  }

  let seenMsgs = 0;
  function enhance() {
    // glide to a newly asked question
    const msgs = D.querySelectorAll('[data-testid="stChatMessage"]');
    if (msgs.length > seenMsgs && msgs.length) {
      const target = msgs[Math.max(0, msgs.length - 2)];
      if (seenMsgs > 0 || msgs.length <= 2) target.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
    }
    seenMsgs = msgs.length;
    // count-up stats
    D.querySelectorAll('[data-count]:not([data-fx])').forEach(el => {
      el.dataset.fx = 1;
      const end = parseFloat(el.dataset.count), delay = parseFloat(el.dataset.delay || 0) * 1000, dur = 1400;
      if (reduce) { el.textContent = end; return; }
      el.textContent = '0';
      setTimeout(() => {
        const s = performance.now();
        (function step(t) {
          const x = Math.min(1, (t - s) / dur), v = end * (1 - Math.pow(1 - x, 4));
          el.textContent = Math.round(v).toLocaleString();
          if (x < 1) W.requestAnimationFrame(step);
        })(s);
      }, delay);
    });
    // tilt + spotlight on prompt cards
    if (!touch && !reduce) D.querySelectorAll('.stButton > button:not([data-fx])').forEach(b => {
      b.dataset.fx = 1;
      b.addEventListener('mousemove', e => {
        const r = b.getBoundingClientRect(), px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
        b.style.setProperty('--ry', ((px - 0.5) * 10).toFixed(2) + 'deg');
        b.style.setProperty('--rx', ((0.5 - py) * 10).toFixed(2) + 'deg');
        b.style.setProperty('--mx', (px * 100).toFixed(1) + '%');
        b.style.setProperty('--my', (py * 100).toFixed(1) + '%');
      });
      b.addEventListener('mouseleave', () => { b.style.setProperty('--rx', '0deg'); b.style.setProperty('--ry', '0deg'); });
    });
    // text scramble on hover
    D.querySelectorAll('.scramble:not([data-fx])').forEach(el => {
      el.dataset.fx = 1; el.addEventListener('mouseenter', () => scramble(el));
      if (!reduce) setTimeout(() => scramble(el), parseFloat(el.dataset.delay || 0) * 1000);
    });
  }
  W.__smcFxEnhance = enhance;

  // Streamlit's chat layout sticks to the bottom; on the landing view keep the hero in sight
  let userMoved = false;
  ['wheel', 'touchstart', 'keydown', 'mousedown'].forEach(ev => W.addEventListener(ev, () => { userMoved = true; }, { passive: true, capture: true }));
  const pinStart = performance.now();
  (function pinTop() {
    const m = D.querySelector('[data-testid="stAppScrollToBottomContainer"]') || D.querySelector('main');
    if (m && !userMoved && !D.querySelector('[data-testid="stChatMessage"]') && m.scrollTop > 0) m.scrollTop = 0;
    if (!userMoved && performance.now() - pinStart < 15000) W.requestAnimationFrame(pinTop);
  })();

  boot();
  chart();
  cursor();
  enhance();
  let pend = 0;
  new MutationObserver(() => { if (!pend) pend = setTimeout(() => { pend = 0; enhance(); }, 120); })
    .observe(D.body, { childList: true, subtree: true });
})();
