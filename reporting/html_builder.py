"""
reporting/html_builder.py
HTML report assembly engine — builds complete, offline, self-contained HTML files.

Design
------
* No JavaScript, no external CDN, no network requests.
* Figures embedded as data:image/png;base64 URIs.
* Uses CSS-only sticky sidebar navigation.
* <details>/<summary> for collapsible appendix sections.
* Print-friendly via @media print in CSS.
"""
import datetime
import sys
from typing import Any, Dict, List, Optional, Union

import polars as pl

from reporting.css_theme import CSS_THEME


class HTMLReportBuilder:
    """
    Fluent builder for a self-contained HTML report.

    Typical usage
    -------------
    >>> b = HTMLReportBuilder(report_title="Data Overview", stage_number=1, config=cfg,
    ...                       n_rows=2_000_000, n_cols=4_000)
    >>> b.add_executive_summary(cards=[...], narrative="...")
    >>> b.add_section("Missing Values", content_html)
    >>> html = b.build()
    """

    def __init__(
        self,
        report_title: str,
        stage_number: int,
        config: dict,
        stage_subtitle: str = "",
        dataset_name: str = "",
        n_rows: int = 0,
        n_cols: int = 0,
    ):
        self.report_title   = report_title
        self.stage_number   = stage_number
        self.stage_subtitle = stage_subtitle
        self.config         = config
        self.n_rows         = n_rows
        self.n_cols         = n_cols
        self.dataset_name   = dataset_name or config.get("data", {}).get("input_path", "")

        proj = config.get("project", {})
        self.project_name = proj.get("name", "Risk Modelling Pipeline")
        self.author       = proj.get("author", "Risk Analytics Team")
        self.version      = proj.get("version", "1.0.0")

        self._sections:   List[Dict] = []
        self._toc:        List[Dict] = []
        self._sec_count   = 0
        self._fig_count   = 0
        self._tbl_count   = 0
        self._generated   = datetime.datetime.now()

    # ==================================================================
    # Section builders  (return self for chaining)
    # ==================================================================

    def add_executive_summary(
        self,
        cards: List[Dict],
        narrative: str = "",
    ) -> "HTMLReportBuilder":
        """
        Add executive summary with KPI cards and narrative paragraph(s).

        card dict keys
        --------------
        label  : str
        value  : str | int | float
        sub    : str  (optional sub-label)
        variant: str  (default | success | warning | danger | info)
        """
        body = self._kpi_grid(cards)
        if narrative:
            body += self._p_block(narrative)
        return self._push_section("Executive Summary", body, icon="📊", numbered=False)

    def add_section(
        self,
        title: str,
        content: str,
        icon: str = "📋",
    ) -> "HTMLReportBuilder":
        """Add a numbered main section."""
        return self._push_section(title, content, icon=icon)

    # ------------------------------------------------------------------
    # Component generators  (return HTML strings — compose manually)
    # ------------------------------------------------------------------

    def table(
        self,
        data: Union[pl.DataFrame, List[Dict]],
        caption: str = "",
        interpretation: str = "",
        max_rows: int = 300,
        extra_class: str = "",
    ) -> str:
        """Render *data* as a styled HTML table."""
        self._tbl_count += 1
        num = self._tbl_count

        if isinstance(data, pl.DataFrame):
            cols = data.columns
            rows = data.to_dicts()
        elif isinstance(data, list) and data:
            cols = list(data[0].keys())
            rows = data
        else:
            return "<p><em>No data available.</em></p>"

        rows = rows[:max_rows]
        th   = "".join(f"<th>{c}</th>" for c in cols)
        trs  = "\n".join(
            "<tr>" + "".join(f"<td>{self._fmt_cell(r.get(c, ''))}</td>" for c in cols) + "</tr>"
            for r in rows
        )

        cap_html = ""
        if caption:
            cap_html += f'<div class="tbl-caption">Table {num}: {caption}</div>'
        if interpretation:
            cap_html += f'<div class="tbl-interpretation">{interpretation}</div>'

        cls = f"compare-table {extra_class}".strip() if extra_class else ""
        tbl_class = f' class="{cls}"' if cls else ""

        return f"""
<div class="table-wrap">
  <table{tbl_class}>
    <thead><tr>{th}</tr></thead>
    <tbody>{trs}</tbody>
  </table>
  {cap_html}
</div>"""

    def figure(
        self,
        base64_png: str,
        title: str,
        description: str = "",
        interpretation: str = "",
        business_implication: str = "",
    ) -> str:
        """Embed a base64 PNG figure with caption."""
        self._fig_count += 1
        num = self._fig_count

        foot_parts = []
        if description:
            foot_parts.append(f'<div class="fig-caption">Figure {num}: {description}</div>')
        if interpretation:
            foot_parts.append(
                f'<div class="fig-interp"><strong>Interpretation:</strong> {interpretation}</div>'
            )
        if business_implication:
            foot_parts.append(
                f'<div class="fig-business"><strong>Business Implication:</strong> {business_implication}</div>'
            )
        foot = f'<div class="figure-foot">{"".join(foot_parts)}</div>' if foot_parts else ""

        return f"""
<div class="figure-box">
  <div class="figure-head">
    <span class="fig-num">Fig {num}</span>
    <span class="fig-title">{title}</span>
  </div>
  <div class="figure-body">
    <img src="data:image/png;base64,{base64_png}" alt="{title}" loading="lazy" />
  </div>
  {foot}
</div>"""

    def callout(
        self,
        text: str,
        kind: str = "note",
        title: str = "",
    ) -> str:
        """
        Render a callout box.
        kind: note | warning | danger | success | recommend | insight
        """
        defaults = {
            "note":      "ℹ Note",
            "warning":   "⚠ Warning",
            "danger":    "🚨 Critical Risk",
            "success":   "✓ Good",
            "recommend": "💡 Recommendation",
            "insight":   "🔍 Business Insight",
        }
        t = title or defaults.get(kind, "Note")
        return (
            f'<div class="callout {kind}">'
            f'<div class="callout-title">{t}</div>{text}</div>'
        )

    def score_card(
        self,
        score: Union[int, float],
        label: str = "Overall Score",
        grade: str = "",
        max_score: int = 100,
    ) -> str:
        """Render a large score hero card."""
        s = f"{score:.0f}" if isinstance(score, float) else str(score)
        g = f'<div class="score-grade">{grade}</div>' if grade else ""
        return (
            f'<div class="score-card">'
            f'<div class="score-value">{s}</div>'
            f'<div class="score-label">{label} / {max_score}</div>'
            f"{g}</div>"
        )

    def card(self, title: str, body: str) -> str:
        return (
            f'<div class="card">'
            f'<div class="card-title">{title}</div>{body}</div>'
        )

    def collapsible(self, summary: str, body: str, open_: bool = False) -> str:
        attr = " open" if open_ else ""
        return (
            f'<details{attr}><summary>{summary}</summary>'
            f'<div class="details-body">{body}</div></details>'
        )

    def subsection(self, title: str) -> str:
        return f'<h3 class="subsection-title">{title}</h3>'

    def progress_bar(self, value: float, max_val: float = 100.0, variant: str = "") -> str:
        pct = min(100.0, (value / max(max_val, 1e-9)) * 100)
        cls = f"pbar {variant}".strip() if variant else "pbar"
        return (
            f'<div class="pbar-wrap"><div class="{cls}" style="width:{pct:.1f}%"></div></div>'
        )

    def badge(self, text: str, kind: str = "") -> str:
        cls = f"badge b-{kind}" if kind else "badge"
        return f'<span class="{cls}">{text}</span>'

    def hr(self) -> str:
        return "<hr />"

    def p(self, text: str) -> str:
        return f"<p>{text}</p>"

    def h3(self, text: str) -> str:
        return f'<h3 class="subsection-title">{text}</h3>'

    # ==================================================================
    # Final build
    # ==================================================================

    def build(self) -> str:
        """Return the complete HTML document string."""
        sections_html = "\n".join(s["html"] for s in self._sections)
        sidebar_html  = self._sidebar()
        header_html   = self._header()
        footer_html   = self._footer()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="{self.report_title} — {self.project_name}" />
  <meta name="author" content="{self.author}" />
  <title>{self.report_title} — {self.project_name}</title>
  <style>
{CSS_THEME}
  </style>
</head>
<body>
<div class="page-wrapper">
  {sidebar_html}
  <div class="main-content">
    {header_html}
    <main id="report-body">
      {sections_html}
    </main>
    {footer_html}
  </div>
</div>
</body>
</html>"""

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _push_section(
        self,
        title: str,
        content: str,
        icon: str = "📋",
        numbered: bool = True,
    ) -> "HTMLReportBuilder":
        self._sec_count += 1
        sid  = f"sec-{self._sec_count}"
        num  = f'<span class="sec-num">{self._sec_count}</span> ' if numbered else ""
        head = f'<h2 class="section-title" id="{sid}">{num}{icon} {title}</h2>'
        self._sections.append({"html": f"<section id=\"{sid}\">{head}{content}</section>"})
        self._toc.append({"title": title, "id": sid, "num": self._sec_count, "numbered": numbered})
        return self

    def _kpi_grid(self, cards: List[Dict]) -> str:
        items = []
        for c in cards:
            var = c.get("variant", "")
            cls = f'kpi-card {var}' if var else "kpi-card"
            sub = f'<div class="kpi-sub">{c["sub"]}</div>' if c.get("sub") else ""
            items.append(
                f'<div class="{cls}">'
                f'<div class="kpi-label">{c["label"]}</div>'
                f'<div class="kpi-value">{c["value"]}</div>'
                f'{sub}</div>'
            )
        return f'<div class="kpi-grid">{"".join(items)}</div>'

    def _p_block(self, text: str) -> str:
        """Convert double-newline-delimited text into <p> tags."""
        parts = [t.strip() for t in text.split("\n\n") if t.strip()]
        html  = "".join(f"<p>{p}</p>" for p in parts) if parts else f"<p>{text}</p>"
        return f'<div class="card"><div class="details-body">{html}</div></div>'

    def _sidebar(self) -> str:
        links = []
        for item in self._toc:
            num_span = f'<span class="nav-num">{item["num"]}</span>' if item["numbered"] else ""
            links.append(f'<a href="#{item["id"]}">{num_span} {item["title"]}</a>')
        return f"""
<aside class="sidebar" role="navigation" aria-label="Report sections">
  <div class="sidebar-brand">
    <div class="proj-name">📊 {self.project_name}</div>
    <span class="stage-pill">Stage {self.stage_number:02d}</span>
  </div>
  <nav>
    <div class="nav-group-label">Contents</div>
    {"".join(links)}
  </nav>
</aside>"""

    def _header(self) -> str:
        data_cfg = self.config.get("data", {})
        domain   = self.config.get("domain", {})

        metas = [
            ("Project",    self.project_name),
            ("Author",     self.author),
            ("Domain",     domain.get("name", "Credit Risk")),
            ("Dataset",    str(self.dataset_name).split("/")[-1] or "N/A"),
            ("Records",    f"{self.n_rows:,}" if self.n_rows else "N/A"),
            ("Features",   f"{self.n_cols:,}" if self.n_cols else "N/A"),
            ("Target",     data_cfg.get("target_column", "target")),
            ("Generated",  self._generated.strftime("%d %b %Y  %H:%M")),
        ]
        meta_html = "".join(
            f'<div class="meta-cell">'
            f'<span class="meta-label">{k}</span>'
            f'<span class="meta-value">{v}</span>'
            f'</div>'
            for k, v in metas
        )
        sub_html = (
            f'<p class="header-subtitle">{self.stage_subtitle}</p>'
            if self.stage_subtitle else ""
        )
        return f"""
<header class="page-header">
  <div class="header-inner">
    <div class="report-eyebrow">Stage {self.stage_number:02d} · Risk Analytics Pipeline · {self.author}</div>
    <h1>{self.report_title}</h1>
    {sub_html}
    <div class="header-meta">{meta_html}</div>
  </div>
</header>"""

    def _footer(self) -> str:
        py_ver = sys.version.split()[0]
        libs   = {}
        for lib in ["duckdb", "polars", "sklearn", "numpy", "matplotlib"]:
            try:
                name = "scikit-learn" if lib == "sklearn" else lib
                mod  = __import__("sklearn" if lib == "sklearn" else lib)
                libs[name] = getattr(mod, "__version__", "?")
            except ImportError:
                pass
        lib_str = " · ".join(f"{k} {v}" for k, v in libs.items())
        return f"""
<footer class="report-footer">
  <div class="footer-info">
    <p><strong>{self.project_name}</strong> — {self.report_title}</p>
    <p>Generated: {self._generated.strftime('%Y-%m-%d %H:%M:%S')} · Python {py_ver} · {lib_str}</p>
    <p>Author: {self.author} · Version: {self.version} · Confidential &amp; Proprietary</p>
  </div>
  <div class="footer-badge">STATIC REPORT<br/>OFFLINE READY</div>
</footer>"""

    @staticmethod
    def _fmt_cell(val: Any) -> str:
        if val is None:
            return '<span style="color:#94a3b8;font-style:italic">null</span>'
        if isinstance(val, float):
            if val != val:  # NaN
                return '<span style="color:#94a3b8">NaN</span>'
            return f"{val:,.4f}" if abs(val) < 1_000_000 else f"{val:,.2f}"
        if isinstance(val, int):
            return f"{val:,}"
        s = str(val)
        # Truncate very long strings
        return s[:120] + "…" if len(s) > 120 else s
