"""
reporting/css_theme.py
Complete professional CSS stylesheet embedded into every HTML report.
Light theme (print-friendly) with dark-mode support via @media query.
Zero external dependencies — fully offline.
"""

CSS_THEME = """
/* ==========================================================================
   RISK MODELLING PIPELINE  —  PROFESSIONAL REPORT THEME  v1.0
   Light / Dark / Print
   ========================================================================== */

/* ── Google Fonts fallback stack (no CDN — system fonts only) ─────────────── */

/* ── CSS Custom Properties ─────────────────────────────────────────────────── */
:root {
  /* Brand palette */
  --navy:          #0f2d52;
  --navy-mid:      #1a3a5c;
  --navy-light:    #2a5298;
  --sky:           #0ea5e9;
  --sky-dark:      #0369a1;
  --emerald:       #059669;
  --amber:         #d97706;
  --rose:          #dc2626;
  --violet:        #7c3aed;
  --indigo:        #4f46e5;
  --teal:          #0891b2;

  /* Backgrounds */
  --bg-page:       #f0f4f8;
  --bg-surface:    #ffffff;
  --bg-subtle:     #f8fafc;
  --bg-muted:      #f1f5f9;

  /* Text */
  --text-base:     #0f172a;
  --text-secondary:#475569;
  --text-muted:    #94a3b8;

  /* Borders */
  --border:        #e2e8f0;
  --border-mid:    #cbd5e1;

  /* Shadows */
  --shadow-xs: 0 1px 2px rgba(0,0,0,.05);
  --shadow-sm: 0 2px 6px rgba(0,0,0,.07), 0 1px 3px rgba(0,0,0,.05);
  --shadow-md: 0 4px 16px rgba(0,0,0,.09), 0 2px 6px rgba(0,0,0,.05);
  --shadow-lg: 0 12px 32px rgba(0,0,0,.11), 0 4px 12px rgba(0,0,0,.06);

  /* Radius */
  --r-sm:  6px;
  --r-md:  10px;
  --r-lg:  16px;
  --r-xl:  22px;

  /* Typography */
  --font:  -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono:  "Cascadia Code", "Fira Code", "JetBrains Mono", "Courier New", monospace;

  /* Layout */
  --sidebar-w: 272px;
}

/* ── Reset ─────────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 16px; scroll-behavior: smooth; }
body {
  font-family: var(--font);
  background:  var(--bg-page);
  color:       var(--text-base);
  line-height: 1.7;
  min-height:  100vh;
}

/* ── Page layout ───────────────────────────────────────────────────────────── */
.page-wrapper { display: flex; min-height: 100vh; }

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
.sidebar {
  width:      var(--sidebar-w);
  min-width:  var(--sidebar-w);
  background: var(--navy);
  position:   sticky;
  top:        0;
  height:     100vh;
  overflow-y: auto;
  display:    flex;
  flex-direction: column;
  z-index:    200;
  flex-shrink: 0;
}
.sidebar::-webkit-scrollbar         { width: 4px; }
.sidebar::-webkit-scrollbar-track   { background: transparent; }
.sidebar::-webkit-scrollbar-thumb   { background: rgba(255,255,255,.18); border-radius: 2px; }

.sidebar-brand {
  padding: 22px 20px;
  border-bottom: 1px solid rgba(255,255,255,.10);
}
.sidebar-brand .proj-name {
  color: #fff;
  font-size: 14px;
  font-weight: 700;
  letter-spacing: .3px;
  line-height: 1.35;
}
.sidebar-brand .stage-pill {
  display:      inline-block;
  margin-top:   8px;
  padding:      3px 12px;
  background:   var(--sky);
  color:        #fff;
  border-radius:20px;
  font-size:    11px;
  font-weight:  700;
  letter-spacing:.4px;
}

.sidebar nav { padding: 14px 0; flex: 1; }

.nav-group-label {
  padding:        10px 20px 4px;
  font-size:      10px;
  font-weight:    700;
  letter-spacing: 1.8px;
  text-transform: uppercase;
  color:          rgba(255,255,255,.35);
}

.sidebar nav a {
  display:     flex;
  align-items: center;
  gap:         8px;
  padding:     8px 18px;
  color:       rgba(255,255,255,.72);
  text-decoration: none;
  font-size:   13px;
  font-weight: 500;
  border-left: 3px solid transparent;
  transition:  background .15s, color .15s, border-color .15s;
  line-height: 1.35;
}
.sidebar nav a:hover {
  background:        rgba(255,255,255,.07);
  color:             #fff;
  border-left-color: var(--sky);
}
.nav-num {
  display:         inline-flex;
  align-items:     center;
  justify-content: center;
  min-width:       22px;
  height:          22px;
  background:      rgba(255,255,255,.12);
  border-radius:   50%;
  font-size:       10.5px;
  font-weight:     700;
  flex-shrink:     0;
}

/* ── Main content ──────────────────────────────────────────────────────────── */
.main-content {
  flex:        1;
  min-width:   0;
  padding:     36px 44px;
  max-width:   calc(100vw - var(--sidebar-w));
}

/* ── Page header ───────────────────────────────────────────────────────────── */
.page-header {
  background: linear-gradient(130deg, var(--navy) 0%, var(--navy-light) 60%, #1e5799 100%);
  color:      white;
  padding:    48px 52px 42px;
  margin:     -36px -44px 44px;
  position:   relative;
  overflow:   hidden;
}
.page-header::before {
  content:''; position:absolute; top:-80px; right:-80px;
  width:320px; height:320px; border-radius:50%;
  background:rgba(255,255,255,.035);
}
.page-header::after {
  content:''; position:absolute; bottom:-100px; left:280px;
  width:280px; height:280px; border-radius:50%;
  background:rgba(14,165,233,.07);
}
.header-inner { position:relative; z-index:1; }

.report-eyebrow {
  font-size:      11px;
  font-weight:    700;
  letter-spacing: 2px;
  text-transform: uppercase;
  color:          rgba(255,255,255,.5);
  margin-bottom:  10px;
}
.page-header h1 {
  font-size:    clamp(24px, 3.5vw, 38px);
  font-weight:  800;
  letter-spacing:-.5px;
  line-height:  1.2;
  margin-bottom:10px;
}
.header-subtitle {
  font-size:  15px;
  color:      rgba(255,255,255,.72);
  margin-bottom: 24px;
}
.header-meta {
  display:    flex;
  flex-wrap:  wrap;
  gap:        20px 28px;
  padding-top:20px;
  border-top: 1px solid rgba(255,255,255,.13);
}
.meta-cell { display:flex; flex-direction:column; gap:2px; }
.meta-label {
  font-size:      10px;
  font-weight:    600;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color:          rgba(255,255,255,.45);
}
.meta-value {
  font-size:  13px;
  font-weight:600;
  color:      rgba(255,255,255,.9);
}

/* ── Sections ──────────────────────────────────────────────────────────────── */
section { margin-bottom: 44px; }

.section-title {
  display:       flex;
  align-items:   center;
  gap:           10px;
  font-size:     21px;
  font-weight:   700;
  color:         var(--navy);
  margin-bottom: 20px;
  padding-bottom:12px;
  border-bottom: 2px solid var(--border);
}
.sec-num {
  display:         inline-flex;
  align-items:     center;
  justify-content: center;
  width:           32px;
  height:          32px;
  background:      var(--navy);
  color:           #fff;
  border-radius:   8px;
  font-size:       13px;
  font-weight:     700;
  flex-shrink:     0;
}

.subsection-title {
  font-size:    16.5px;
  font-weight:  600;
  color:        var(--navy-light);
  margin:       28px 0 12px;
  padding-left: 14px;
  border-left:  4px solid var(--sky);
}

/* ── KPI Cards ─────────────────────────────────────────────────────────────── */
.kpi-grid {
  display:               grid;
  grid-template-columns: repeat(auto-fill, minmax(175px, 1fr));
  gap:                   16px;
  margin-bottom:         32px;
}
.kpi-card {
  background:    var(--bg-surface);
  border:        1px solid var(--border);
  border-radius: var(--r-md);
  padding:       20px 18px;
  box-shadow:    var(--shadow-sm);
  position:      relative;
  overflow:      hidden;
}
.kpi-card::before {
  content:''; position:absolute;
  top:0; left:0; right:0; height:4px;
  background: var(--sky);
}
.kpi-card.success::before { background: var(--emerald); }
.kpi-card.warning::before { background: var(--amber); }
.kpi-card.danger::before  { background: var(--rose); }
.kpi-card.info::before    { background: var(--violet); }

.kpi-label {
  font-size:      10.5px;
  font-weight:    700;
  letter-spacing: .8px;
  text-transform: uppercase;
  color:          var(--text-muted);
  margin-bottom:  8px;
}
.kpi-value {
  font-size:   30px;
  font-weight: 800;
  color:       var(--text-base);
  line-height: 1;
  margin-bottom:5px;
}
.kpi-sub {
  font-size:  12px;
  color:      var(--text-secondary);
  font-weight:500;
}

/* ── Callout boxes ─────────────────────────────────────────────────────────── */
.callout {
  border-radius: var(--r-md);
  padding:       16px 20px;
  margin:        18px 0;
  border-left:   5px solid;
  font-size:     14.5px;
  line-height:   1.7;
}
.callout-title {
  font-size:      11.5px;
  font-weight:    700;
  letter-spacing: .8px;
  text-transform: uppercase;
  margin-bottom:  6px;
}

.callout.note         { background:#eff6ff; border-color:#3b82f6; color:#1e3a8a; }
.callout.note         .callout-title { color:#1d4ed8; }
.callout.warning      { background:#fffbeb; border-color:#f59e0b; color:#78350f; }
.callout.warning      .callout-title { color:#b45309; }
.callout.danger       { background:#fef2f2; border-color:#ef4444; color:#7f1d1d; }
.callout.danger       .callout-title { color:#b91c1c; }
.callout.success      { background:#f0fdf4; border-color:#22c55e; color:#14532d; }
.callout.success      .callout-title { color:#166534; }
.callout.recommend    { background:#faf5ff; border-color:#9333ea; color:#3b0764; }
.callout.recommend    .callout-title { color:#7e22ce; }
.callout.insight      { background:#f0f9ff; border-color:#0ea5e9; color:#0c4a6e; }
.callout.insight      .callout-title { color:#0369a1; }

/* ── Card ──────────────────────────────────────────────────────────────────── */
.card {
  background:    var(--bg-surface);
  border:        1px solid var(--border);
  border-radius: var(--r-lg);
  padding:       24px 28px;
  box-shadow:    var(--shadow-sm);
  margin-bottom: 24px;
}
.card-title {
  font-size:      12px;
  font-weight:    700;
  letter-spacing: .6px;
  text-transform: uppercase;
  color:          var(--text-muted);
  margin-bottom:  16px;
}

/* ── Score card ────────────────────────────────────────────────────────────── */
.score-card {
  text-align:    center;
  padding:       32px;
  border-radius: var(--r-xl);
  background:    linear-gradient(135deg, var(--navy), var(--navy-light));
  color:         white;
  box-shadow:    var(--shadow-md);
  margin-bottom: 24px;
}
.score-value  { font-size:68px; font-weight:900; line-height:1; letter-spacing:-3px; }
.score-label  { font-size:13px; font-weight:600; letter-spacing:1px; text-transform:uppercase;
                color:rgba(255,255,255,.65); margin-top:10px; }
.score-grade  { display:inline-block; margin-top:14px; padding:6px 22px;
                background:rgba(255,255,255,.14); border-radius:20px;
                font-size:17px; font-weight:700; }

/* ── Tables ────────────────────────────────────────────────────────────────── */
.table-wrap {
  overflow-x:    auto;
  margin:        18px 0;
  border-radius: var(--r-md);
  border:        1px solid var(--border);
  box-shadow:    var(--shadow-xs);
}
table {
  width:           100%;
  border-collapse: collapse;
  font-size:       13.5px;
  background:      var(--bg-surface);
}
thead { background: var(--navy); color: #fff; }
thead th {
  padding:        11px 15px;
  text-align:     left;
  font-weight:    600;
  font-size:      12px;
  letter-spacing: .4px;
  white-space:    nowrap;
}
tbody tr:nth-child(even) { background: var(--bg-subtle); }
tbody tr:hover            { background: #e0f2fe; transition: background .1s; }
tbody td {
  padding:        9px 15px;
  border-bottom:  1px solid var(--border);
  vertical-align: middle;
}
tbody tr:last-child td { border-bottom: none; }

.tbl-caption {
  padding:        10px 15px;
  font-size:      12px;
  color:          var(--text-secondary);
  background:     var(--bg-muted);
  border-top:     1px solid var(--border);
  font-style:     italic;
}
.tbl-interpretation {
  padding:    10px 15px;
  font-size:  13px;
  color:      var(--text-base);
  line-height:1.65;
  background: var(--bg-subtle);
  border-top: 1px solid var(--border);
}

/* ── Badges ────────────────────────────────────────────────────────────────── */
.badge {
  display:       inline-block;
  padding:       2px 10px;
  border-radius: 12px;
  font-size:     11px;
  font-weight:   700;
  letter-spacing:.2px;
}
.b-numeric     { background:#dbeafe; color:#1d4ed8; }
.b-categorical { background:#fce7f3; color:#9d174d; }
.b-boolean     { background:#dcfce7; color:#166534; }
.b-datetime    { background:#fef9c3; color:#854d0e; }
.b-identifier  { background:#f3f4f6; color:#374151; }
.b-target      { background:#ede9fe; color:#5b21b6; }
.b-constant    { background:#fee2e2; color:#991b1b; }
.b-high        { background:#fee2e2; color:#991b1b; }
.b-medium      { background:#fef3c7; color:#92400e; }
.b-low         { background:#dcfce7; color:#166534; }
.b-none        { background:#f1f5f9; color:#64748b; }
.b-treebased   { background:#fef3c7; color:#92400e; }
.b-lrbased     { background:#dbeafe; color:#1e40af; }

/* ── Figures ───────────────────────────────────────────────────────────────── */
.figure-box {
  margin:        26px 0;
  background:    var(--bg-surface);
  border:        1px solid var(--border);
  border-radius: var(--r-md);
  overflow:      hidden;
  box-shadow:    var(--shadow-sm);
}
.figure-head {
  display:     flex;
  align-items: center;
  gap:         10px;
  padding:     11px 18px;
  background:  var(--bg-muted);
  border-bottom: 1px solid var(--border);
}
.fig-num   { font-size:11px; font-weight:700; color:var(--text-muted);
             background:var(--border-mid,var(--border)); padding:2px 9px;
             border-radius:4px; }
.fig-title { font-size:14px; font-weight:600; color:var(--text-base); }

.figure-body { padding: 16px; text-align: center; }
.figure-body img {
  display:   block;
  max-width: 100%;
  height:    auto;
  margin:    0 auto;
  border-radius: 6px;
}

.figure-foot {
  padding:    12px 18px;
  background: var(--bg-muted);
  border-top: 1px solid var(--border);
}
.fig-caption  { font-size:12.5px; color:var(--text-secondary); font-style:italic; margin-bottom:7px; }
.fig-interp   { font-size:13.5px; color:var(--text-base); line-height:1.65; }
.fig-business { font-size:13.5px; color:var(--navy-light); line-height:1.65; margin-top:6px; }

/* ── Collapsible ───────────────────────────────────────────────────────────── */
details {
  background:    var(--bg-surface);
  border:        1px solid var(--border);
  border-radius: var(--r-md);
  margin:        16px 0;
  overflow:      hidden;
}
summary {
  padding:        13px 18px;
  cursor:         pointer;
  font-weight:    600;
  font-size:      14px;
  color:          var(--navy);
  background:     var(--bg-muted);
  user-select:    none;
  list-style:     none;
  display:        flex;
  align-items:    center;
  gap:            8px;
  border-bottom:  1px solid transparent;
}
summary::-webkit-details-marker { display: none; }
summary::before {
  content:'▶'; font-size:9px; color:var(--sky);
  transition:transform .2s; flex-shrink:0;
}
details[open] > summary                 { border-bottom-color: var(--border); }
details[open] > summary::before         { transform: rotate(90deg); }
.details-body { padding: 20px 24px; }

/* ── Progress bar ──────────────────────────────────────────────────────────── */
.pbar-wrap {
  display:       inline-block;
  width:         110px;
  height:        7px;
  background:    var(--bg-muted);
  border-radius: 4px;
  overflow:      hidden;
  vertical-align:middle;
  margin-right:  6px;
}
.pbar {
  height:        100%;
  border-radius: 4px;
  background:    linear-gradient(90deg, var(--sky), var(--navy-light));
}
.pbar.danger  { background: var(--rose); }
.pbar.warning { background: var(--amber); }
.pbar.success { background: var(--emerald); }

/* ── Comparison table (tree vs LR) ────────────────────────────────────────── */
.compare-table thead { background: #1e3a5f; }
.compare-table .col-tree { background: rgba(217,119,6,.08); }
.compare-table .col-lr   { background: rgba(59,130,246,.08); }
.compare-table th.col-tree { border-top: 3px solid var(--amber); }
.compare-table th.col-lr   { border-top: 3px solid var(--sky); }

/* ── Typography ────────────────────────────────────────────────────────────── */
p { margin-bottom:13px; font-size:15px; line-height:1.75; }
strong { font-weight:700; }
ul, ol { padding-left:22px; margin:8px 0 16px; }
li     { font-size:14.5px; margin-bottom:6px; line-height:1.65; }
code   { font-family:var(--mono); font-size:12.5px; background:var(--bg-muted);
         padding:2px 7px; border-radius:4px; color:var(--navy-light); }
blockquote {
  border-left: 4px solid var(--sky);
  padding:     13px 20px;
  margin:      16px 0;
  background:  #f0f9ff;
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  font-style:  italic;
  color:       var(--text-secondary);
}
hr { border:none; height:1px; background:var(--border); margin:36px 0; }

/* ── Footer ────────────────────────────────────────────────────────────────── */
.report-footer {
  background:  var(--navy);
  color:       rgba(255,255,255,.65);
  padding:     24px 44px;
  margin:      60px -44px -36px;
  font-size:   12.5px;
  display:     grid;
  grid-template-columns: 1fr auto;
  gap:         16px;
  align-items: center;
}
.footer-info p  { margin:0; line-height:1.7; }
.footer-badge {
  padding:    8px 16px;
  background: rgba(255,255,255,.08);
  border-radius: var(--r-sm);
  font-size:  11px;
  font-weight:600;
  color:      rgba(255,255,255,.5);
  text-align: center;
  letter-spacing:.5px;
}

/* ── Responsive ────────────────────────────────────────────────────────────── */
@media (max-width: 1100px) {
  :root { --sidebar-w: 240px; }
  .main-content    { padding: 24px 28px; }
  .page-header     { padding: 32px 28px; margin: -24px -28px 32px; }
  .report-footer   { margin: 48px -28px -24px; padding: 20px 28px; }
}
@media (max-width: 768px) {
  .page-wrapper { flex-direction: column; }
  .sidebar      { width:100%; height:auto; position:relative; }
  .main-content { max-width:100%; padding: 20px; }
  .page-header  { margin: -20px -20px 24px; }
  .report-footer{ margin: 40px -20px -20px; grid-template-columns: 1fr; }
  .kpi-grid     { grid-template-columns: repeat(2,1fr); }
}

/* ── Print ─────────────────────────────────────────────────────────────────── */
@media print {
  .sidebar       { display:none !important; }
  .main-content  { padding:0; max-width:100%; }
  .page-header   { margin:0 0 24px; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
  .report-footer { margin:32px 0 0; }
  section        { page-break-before: always; }
  section:first-of-type { page-break-before: auto; }
  .card, .figure-box, .callout { break-inside:avoid; page-break-inside:avoid; }
  table  { font-size:11px; }
  thead  { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
  thead th, tbody td { padding:7px 10px; }
  a      { color:var(--text-base); text-decoration:none; }
  details[open] { break-inside: avoid; }
}

/* ── Dark mode ─────────────────────────────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
  :root {
    --bg-page:    #0b1120;
    --bg-surface: #131e30;
    --bg-subtle:  #1a2840;
    --bg-muted:   #1e2f48;
    --text-base:  #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border:     #1e3a5f;
    --border-mid: #2d5986;
  }
  tbody tr:nth-child(even) { background: #1a2840; }
  tbody tr:hover            { background: #1e3d6e; }
  .callout.note   { background:#0f2744; color:#bfdbfe; }
  .callout.warning{ background:#2d1f00; color:#fde68a; }
  .callout.danger { background:#2d0a0a; color:#fca5a5; }
  .callout.success{ background:#052e16; color:#bbf7d0; }
  .callout.recommend { background:#1e0f45; color:#e9d5ff; }
  code { background:#1e3a5f; color:#7dd3fc; }
  blockquote { background:#0f2744; }
}
"""
