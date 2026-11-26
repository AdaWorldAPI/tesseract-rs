/* The archive's desktop view, modelled on report-rs's static/desktop.js. On the
   desktop (#desk + #task) every a[data-win] opens its page as a window (an iframe
   of the archive's own page); windows move by their title bar, resize from the
   corner, minimise to the taskbar and close. Inside a window (a folder page) an
   a[data-win] asks the desktop to open its page as another window. Without
   script every link is a plain link to the same page. */
(function () {
  'use strict';
  function plainClick(e) { return !(e.ctrlKey || e.metaKey || e.shiftKey || e.altKey || e.button !== 0); }
  function wire(open) {
    document.querySelectorAll('a[data-win]').forEach(function (a) {
      a.addEventListener('click', function (e) {
        if (!plainClick(e)) return;
        e.preventDefault();
        open(a.getAttribute('data-win'), a.getAttribute('data-title') || a.textContent, a.getAttribute('href'));
      });
    });
  }
  var desk = document.getElementById('desk'), task = document.getElementById('task');
  if (!desk || !task) {
    // A page inside a desktop window: hand window links to the desktop around it.
    var host = null;
    try { if (window.parent !== window && typeof window.parent.desktopOpen === 'function') host = window.parent; } catch (e) {}
    if (host) wire(function (id, title, src) { host.desktopOpen(id, title, src); });
    return;
  }
  var spacer = task.querySelector('.sp'), z = 10, wins = {};
  function focus(w) {
    w.el.classList.remove('min'); w.el.style.zIndex = ++z;
    Object.keys(wins).forEach(function (k) { wins[k].btn.classList.toggle('on', wins[k] === w); });
  }
  function close(id) {
    var w = wins[id]; if (!w) return;
    w.el.remove(); w.btn.remove(); delete wins[id];
  }
  function open(id, title, src) {
    if (wins[id]) { focus(wins[id]); return; }
    var n = Object.keys(wins).length % 8;
    var el = document.createElement('section'); el.className = 'win';
    el.setAttribute('data-win-id', id);
    // Never wider or taller than the desk, so the controls stay on screen; at most
    // 960 x 620 on a wide desk. The offset cascades only while the window still fits.
    var dw = desk.clientWidth, dh = desk.clientHeight;
    var ww = Math.min(960, Math.max(Math.min(360, dw), dw - 80));
    var wh = Math.min(620, Math.max(Math.min(240, dh), dh - 48));
    el.style.width = ww + 'px'; el.style.height = wh + 'px';
    el.style.left = Math.max(0, Math.min(40 + 28 * n, dw - ww)) + 'px';
    el.style.top = Math.max(0, Math.min(24 + 24 * n, dh - wh)) + 'px';
    el.setAttribute('role', 'dialog'); el.setAttribute('aria-label', title);
    var bar = document.createElement('div'); bar.className = 'bar';
    var t = document.createElement('b'); t.textContent = title; bar.appendChild(t);
    var ne = document.createElement('a'); ne.href = src; ne.textContent = '↗'; ne.title = 'Open as page';
    ne.setAttribute('aria-label', 'Open as page');
    var mn = document.createElement('button'); mn.type = 'button'; mn.textContent = '–'; mn.title = 'Minimise';
    mn.setAttribute('aria-label', 'Minimise');
    var cl = document.createElement('button'); cl.type = 'button'; cl.textContent = '×'; cl.title = 'Close';
    cl.setAttribute('aria-label', 'Close');
    bar.appendChild(ne); bar.appendChild(mn); bar.appendChild(cl);
    var fr = document.createElement('iframe'); fr.src = src; fr.title = title;
    el.appendChild(bar); el.appendChild(fr); desk.appendChild(el);
    var btn = document.createElement('button'); btn.type = 'button'; btn.textContent = title;
    task.insertBefore(btn, spacer);
    var w = wins[id] = { el: el, btn: btn };
    btn.onclick = function () {
      if (btn.classList.contains('on') && !el.classList.contains('min')) { el.classList.add('min'); btn.classList.remove('on'); }
      else focus(w);
    };
    mn.onclick = function () { el.classList.add('min'); btn.classList.remove('on'); };
    cl.onclick = function () { close(id); };
    el.addEventListener('pointerdown', function () { focus(w); });
    // A click inside the page never reaches this section (an iframe is its own
    // document); the pages are same-origin, so listen in each loaded page too.
    fr.addEventListener('load', function () {
      try { fr.contentDocument.addEventListener('pointerdown', function () { if (wins[id] === w) focus(w); }); } catch (e) {}
    });
    bar.addEventListener('pointerdown', function (e) {
      if (e.target.tagName === 'BUTTON' || e.target.tagName === 'A') return;
      var sx = e.clientX, sy = e.clientY, ox = el.offsetLeft, oy = el.offsetTop;
      fr.style.pointerEvents = 'none'; bar.setPointerCapture(e.pointerId);
      function mv(ev) { el.style.left = Math.max(0, ox + ev.clientX - sx) + 'px'; el.style.top = Math.max(0, oy + ev.clientY - sy) + 'px'; }
      // Ends the drag on release, and also when the browser cancels it or takes the
      // capture away; otherwise the page would stay unclickable. Safe to run twice.
      var ends = ['pointerup', 'pointercancel', 'lostpointercapture'];
      function up() {
        fr.style.pointerEvents = '';
        bar.removeEventListener('pointermove', mv);
        ends.forEach(function (t) { bar.removeEventListener(t, up); });
      }
      bar.addEventListener('pointermove', mv);
      ends.forEach(function (t) { bar.addEventListener(t, up); });
    });
    focus(w);
  }
  window.desktopOpen = open;
  wire(open);
  var clock = document.getElementById('clock');
  function tick() { clock.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
  if (clock) { tick(); setInterval(tick, 15000); }
})();
