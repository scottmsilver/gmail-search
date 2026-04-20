"""Render a before/after HTML mockup of the HEADLINE + DETAIL UI
change. Uses real bench rows + real summaries. Outputs to
/tmp/mockup_hd.html so you can open it in a browser (or screenshot
via $B).
"""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

OUT_HTML = Path("/tmp/mockup_hd.html")
DATA = json.loads(Path("/tmp/mockup_data.json").read_text())

_HD_RE = re.compile(r"HEADLINE:\s*(.+?)\s*DETAIL:\s*(.+)", re.IGNORECASE | re.DOTALL)


def split_hd(raw: str) -> tuple[str, str]:
    m = _HD_RE.search(raw or "")
    if not m:
        return (raw or "").strip(), ""
    return m.group(1).strip(), m.group(2).strip()


def clean_sender(s: str) -> str:
    m = re.match(r'^"?([^"<]+?)"?\s*<', s or "")
    return (m.group(1) if m else s).strip() or "(unknown)"


def short_names(s: str) -> str:
    # Very simple; real UI uses shortPeople
    return clean_sender(s).split("@")[0][:40]


def avatar(name: str) -> str:
    initial = (name.strip()[0] if name.strip() else "?").upper()
    # Simple hash→hue for stable colour per sender
    h = 0
    for c in name:
        h = (h * 31 + ord(c)) & 0xFFFFFF
    hue = h % 360
    return f'<span class="avatar" style="background:hsl({hue},55%,50%)">{escape(initial)}</span>'


def row_current(r: dict) -> str:
    names = short_names(r["from"])
    return f"""
    <div class="row">
      {avatar(names)}
      <div class="col names"><strong>{escape(names)}</strong></div>
      <div class="col content">
        <div class="subject">{escape(r['subject'])}</div>
        <div class="summary">{escape(r['summary_v5'][:400])}</div>
      </div>
      <div class="col date">{escape(r['date'] or 'Mar 11')}</div>
    </div>
    """


def row_headline_detail(r: dict) -> str:
    names = short_names(r["from"])
    headline, detail = split_hd(r["summary_hd"])
    return f"""
    <div class="row">
      {avatar(names)}
      <div class="col names"><strong>{escape(names)}</strong></div>
      <div class="col content">
        <div class="subject">{escape(r['subject'])}</div>
        <div class="headline"><strong>{escape(headline)}</strong></div>
        <div class="detail">{escape(detail)}</div>
      </div>
      <div class="col date">{escape(r['date'] or 'Mar 11')}</div>
    </div>
    """


CSS = """
body { font: 13px ui-sans-serif, system-ui; background:#fafafa; color:#111; margin:0 }
h1 { font-size: 20px; margin: 24px 20px 4px }
h2 { font-size: 13px; color:#666; margin: 0 20px 16px; font-weight:normal }
.panel { background: white; margin: 0 20px 28px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.06) }
.panel-title { padding: 10px 16px; background:#f4f4f4; font-weight:600; border-radius:8px 8px 0 0; border-bottom: 1px solid #eee }
.row { display:grid; grid-template-columns: 40px 200px 1fr 90px; gap: 12px; padding: 10px 16px; border-bottom: 1px solid #f0f0f0; align-items:flex-start }
.row:last-child { border-bottom: none }
.avatar { width:32px; height:32px; border-radius:50%; color:white; display:inline-flex; align-items:center; justify-content:center; font-weight:600; font-size:13px }
.col.date { color:#888; font-size:12px; text-align:right; padding-top: 2px }
.col.names { font-size:13px; padding-top:6px }
.subject { font-weight:600; color:#111; margin-bottom:2px; font-size: 13px }
.summary { color: #666; font-size: 13px; line-height: 1.45 }
.headline { color:#111; font-size: 13px; line-height: 1.35; margin-bottom: 3px }
.detail { color:#666; font-size: 12.5px; line-height: 1.45 }
.legend { margin: -16px 20px 20px; font-size:11px; color:#888 }
.compare { display:grid; grid-template-columns: 1fr 1fr; gap: 20px }
"""


def main() -> None:
    sections = []
    # Single panel per pick, with two columns: current vs new
    for r in DATA:
        sections.append(
            f"""
        <div class="compare">
          <div class="panel">
            <div class="panel-title">CURRENT — subject + v5 summary</div>
            {row_current(r)}
          </div>
          <div class="panel">
            <div class="panel-title">PROPOSED — subject + HEADLINE + DETAIL</div>
            {row_headline_detail(r)}
          </div>
        </div>
        """
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>HEADLINE+DETAIL mockup</title>
<style>{CSS}</style></head>
<body>
  <h1>Search result row: current vs proposed HEADLINE + DETAIL</h1>
  <h2>Real bench data. Left: current layout (subject + one wrapped summary paragraph). Right: structured HEADLINE (one line, bold) + DETAIL (muted, wrapped).</h2>
  {''.join(sections)}
</body></html>"""
    OUT_HTML.write_text(html)
    print(f"wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
