"""
reporting/report_writer.py
Saves completed HTML reports and generates an index page.
"""
from pathlib import Path
from core.logger import get_logger


class ReportWriter:
    """
    Saves HTML report strings to the configured output directory
    and generates an index.html linking all reports.
    """

    REPORT_META = {
        "01_Data_Overview.html":          ("📊", "Data Overview",           "Dataset profiling, schema, and statistical summary"),
        "02_Data_Quality.html":           ("🔍", "Data Quality",            "Missing values, duplicates, constants, leakage risks"),
        "03_Exploratory_Data_Analysis.html": ("📈", "Exploratory Analysis", "Distributions, correlations, target relationships"),
        "04_Feature_Engineering.html":    ("⚙️",  "Feature Engineering",    "Transformations, encoding, scaling"),
        "05_Feature_Selection.html":      ("🎯", "Feature Selection",        "All methods ranked — tree-based vs LR-based comparison"),
        "06_Model_Preparation.html":      ("🗂️", "Model Preparation",       "Train/test split, CV strategy, pipeline design"),
        "07_Model_Comparison.html":       ("🏆", "Model Comparison",         "All models trained and ranked by primary metric"),
        "08_Best_Model_Report.html":      ("🥇", "Best Model Report",        "Deep analysis of the winning model"),
        "09_Feature_Importance.html":     ("📌", "Feature Importance",       "Coefficients, SHAP, permutation importance"),
        "10_Final_Executive_Report.html": ("📋", "Executive Report",         "Consolidated findings for senior stakeholders"),
    }

    def __init__(self, config: dict):
        self.config     = config
        self.output_dir = Path(config.get("reporting", {}).get("output_dir", "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger     = get_logger("ReportWriter",
                                     config.get("paths", {}).get("logs_dir", "logs"))

    def write(self, html: str, filename: str, overwrite: bool = True) -> Path:
        """Write *html* string to *output_dir/filename*. Returns the Path."""
        path = self.output_dir / filename
        if path.exists() and not overwrite:
            self.logger.warning(f"Report exists, skipping: {path}")
            return path
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        kb = path.stat().st_size / 1024
        self.logger.info(f"Report saved: {path.name}  ({kb:.1f} KB)")
        return path

    def write_index(self) -> Path:
        """Generate and write reports/index.html linking all saved reports."""
        html = self._build_index()
        return self.write(html, "index.html")

    # ------------------------------------------------------------------

    def _build_index(self) -> str:
        import datetime
        items = ""
        for fname, (icon, name, desc) in self.REPORT_META.items():
            path = self.output_dir / fname
            exists = path.exists()
            size   = f"{path.stat().st_size / 1024:.0f} KB" if exists else ""
            status = "available" if exists else "pending"
            status_col = "#059669" if exists else "#94a3b8"
            link_open  = f'<a href="{fname}" style="text-decoration:none;color:#0ea5e9;">' if exists else "<span>"
            link_close = "</a>" if exists else "</span>"
            items += f"""
<li style="display:flex;align-items:center;gap:16px;margin:10px 0;padding:16px 20px;
    background:#1e293b;border-radius:10px;border-left:4px solid {status_col};">
  <span style="font-size:24px">{icon}</span>
  <div style="flex:1;">
    {link_open}<strong style="font-size:15px;">{name}</strong>{link_close}<br/>
    <span style="font-size:13px;color:#94a3b8;">{desc}</span>
  </div>
  <div style="text-align:right;font-size:12px;color:{status_col};font-weight:600;">
    {status.upper()}<br/><span style="color:#64748b;font-weight:400;">{size}</span>
  </div>
</li>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Risk Modelling Pipeline — Reports</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
            background:#0f172a; color:#f1f5f9; min-height:100vh;
            display:flex; align-items:center; justify-content:center; }}
    .container {{ max-width:760px; width:100%; padding:40px 24px; }}
    h1 {{ font-size:28px; font-weight:800; color:#0ea5e9; margin-bottom:6px; }}
    .sub {{ font-size:14px; color:#94a3b8; margin-bottom:32px; }}
    ul {{ list-style:none; }}
    .footer {{ margin-top:32px; font-size:12px; color:#475569; text-align:center; }}
  </style>
</head>
<body>
<div class="container">
  <h1>📊 Risk Modelling Pipeline</h1>
  <p class="sub">Generated reports — click to open each report offline.</p>
  <ul>{items}</ul>
  <div class="footer">Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Confidential</div>
</div>
</body>
</html>"""
