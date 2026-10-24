"""Small inline-SVG chart helpers for the supervisor report.

Charts are drawn to one scale per axis, coloured through CSS custom properties
(--s1..--s4 series, --ink-* text, --rule grid) so they follow the page's light and
dark themes. Every mark carries a data-tip attribute; the page's script shows it as
a tooltip on hover or focus.
"""
import html
import math

SLOTS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"]


def _e(s):
    return html.escape(str(s), quote=True)


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda s: abs(span / s - n))
    start = math.ceil(lo / step - 1e-9) * step
    out, v = [], start
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def _fmt(v, pct=False, dp=None):
    if pct:
        return f"{v * 100:.0f}%" if dp is None else f"{v * 100:.{dp}f}%"
    if dp is not None:
        return f"{v:.{dp}f}"
    return f"{v:g}"


def legend(names, kinds=None):
    kinds = kinds or ["box"] * len(names)
    items = []
    for i, (nm, k) in enumerate(zip(names, kinds)):
        sw = (f'<i style="background:{SLOTS[i]}"></i>' if k == "box" else
              f'<i style="background:{SLOTS[i]};height:3px;border-radius:2px"></i>')
        items.append(f"<span>{sw}{_e(nm)}</span>")
    return '<div class="legend">' + "".join(items) + "</div>"


def columns(groups, series, values, *, ylabel, ymax=None, ymin=0.0, pct=False, dp=2, width=720, height=300,
            ref=None, tips=None, label_tops=False):
    """Grouped columns. values[s][g]; ref = list of (y, label) horizontal reference lines."""
    ml, mr, mt, mb = 56, 12, 14, 46
    pw, ph = width - ml - mr, height - mt - mb
    vals = [v for row in values for v in row if v is not None and not math.isnan(v)]
    ymax = ymax if ymax is not None else max(vals) * 1.12
    ticks = nice_ticks(ymin, ymax)
    y = lambda v: mt + ph * (1 - (v - ymin) / (ymax - ymin))
    ng, ns = len(groups), len(series)
    band = pw / ng
    bw = min(24.0, (band * 0.72 - 2 * (ns - 1)) / ns)
    gw = ns * bw + 2 * (ns - 1)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    for t in ticks:
        out.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                   f'<text class="tick" x="{ml - 6}" y="{y(t) + 3.5:.1f}" text-anchor="end">{_fmt(t, pct)}</text>')
    out.append(f'<text class="axl" transform="translate(14,{mt + ph / 2:.0f}) rotate(-90)" text-anchor="middle">'
               f'{_e(ylabel)}</text>')
    for gi, g in enumerate(groups):
        x0 = ml + band * gi + (band - gw) / 2
        for si in range(ns):
            v = values[si][gi]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            x = x0 + si * (bw + 2)
            yt, yb = y(max(v, ymin)), y(ymin)
            h = max(yb - yt, 0.5)
            r = min(4, h, bw / 2)
            path = (f"M{x:.1f},{yb:.1f} V{yt + r:.1f} Q{x:.1f},{yt:.1f} {x + r:.1f},{yt:.1f} "
                    f"H{x + bw - r:.1f} Q{x + bw:.1f},{yt:.1f} {x + bw:.1f},{yt + r:.1f} V{yb:.1f} Z")
            tip = tips[si][gi] if tips else f"{series[si]} · {g}: {_fmt(v, pct, dp)}"
            out.append(f'<path d="{path}" fill="{SLOTS[si]}" tabindex="0" data-tip="{_e(tip)}"/>')
            if label_tops:
                out.append(f'<text class="val" x="{x + bw / 2:.1f}" y="{yt - 4:.1f}" text-anchor="middle">'
                           f'{_fmt(v, pct, dp)}</text>')
        out.append(f'<text class="tick" x="{ml + band * gi + band / 2:.1f}" y="{mt + ph + 16}" '
                   f'text-anchor="middle">{_e(g)}</text>')
    for yv, lab in (ref or []):
        out.append(f'<line class="refl" x1="{ml}" x2="{ml + pw}" y1="{y(yv):.1f}" y2="{y(yv):.1f}"/>'
                   f'<text class="lab3" x="{ml + pw - 4}" y="{y(yv) - 5:.1f}" text-anchor="end">{_e(lab)}</text>')
    out.append(f'<line class="axis" x1="{ml}" x2="{ml + pw}" y1="{y(ymin):.1f}" y2="{y(ymin):.1f}"/>')
    out.append("</svg>")
    return "".join(out)


def lines(series, *, xlabel, ylabel, xlog=False, xmin=None, xmax=None, ymin=None, ymax=None, pct=False,
          width=720, height=300, xticks=None, band=None, ref=None, dp=2, xfmt=None, end_labels=True):
    """series: list of dict(name, x, y, lo=None, hi=None, dash=False, slot=i)."""
    ml, mr, mt, mb = 56, 120 if end_labels else 16, 14, 46
    pw, ph = width - ml - mr, height - mt - mb
    xs = [v for s in series for v in s["x"]]
    ys = [v for s in series for v in list(s["y"]) + list(s.get("lo") or []) + list(s.get("hi") or [])
          if v is not None and not math.isnan(v)]
    xmin = min(xs) if xmin is None else xmin
    xmax = max(xs) if xmax is None else xmax
    ymin = min(0.0, min(ys)) if ymin is None else ymin
    ymax = max(ys) * 1.08 if ymax is None else ymax
    tx = (lambda v: math.log10(v)) if xlog else (lambda v: v)
    x = lambda v: ml + pw * (tx(v) - tx(xmin)) / (tx(xmax) - tx(xmin))
    y = lambda v: mt + ph * (1 - (v - ymin) / (ymax - ymin))
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    if band:
        out.append(f'<rect x="{ml}" width="{pw}" y="{y(band[1]):.1f}" height="{y(band[0]) - y(band[1]):.1f}" '
                   f'fill="var(--good)" opacity="0.10"/>')
    for t in nice_ticks(ymin, ymax):
        out.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                   f'<text class="tick" x="{ml - 6}" y="{y(t) + 3.5:.1f}" text-anchor="end">{_fmt(t, pct)}</text>')
    for t in (xticks or nice_ticks(xmin, xmax)):
        lab = xfmt(t) if xfmt else _fmt(t)
        out.append(f'<text class="tick" x="{x(t):.1f}" y="{mt + ph + 16}" text-anchor="middle">{_e(lab)}</text>'
                   f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{mt + ph}" y2="{mt + ph + 4}"/>')
    out.append(f'<text class="axl" x="{ml + pw / 2:.0f}" y="{height - 8}" text-anchor="middle">{_e(xlabel)}</text>')
    out.append(f'<text class="axl" transform="translate(14,{mt + ph / 2:.0f}) rotate(-90)" text-anchor="middle">'
               f'{_e(ylabel)}</text>')
    for yv, lab in (ref or []):
        out.append(f'<line class="refl" x1="{ml}" x2="{ml + pw}" y1="{y(yv):.1f}" y2="{y(yv):.1f}"/>'
                   f'<text class="lab3" x="{ml + 4}" y="{y(yv) - 5:.1f}">{_e(lab)}</text>')
    out.append(f'<line class="axis" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    ends = []
    for i, s in enumerate(series):
        col = SLOTS[s.get("slot", i)]
        pts = [(x(a), y(b)) for a, b in zip(s["x"], s["y"]) if b is not None and not math.isnan(b)]
        if s.get("lo") is not None:
            up = [(x(a), y(h)) for a, h in zip(s["x"], s["hi"])]
            dn = [(x(a), y(l)) for a, l in zip(s["x"], s["lo"])][::-1]
            out.append('<path d="M' + " L".join(f"{a:.1f},{b:.1f}" for a, b in up + dn) +
                       f' Z" fill="{col}" opacity="0.12"/>')
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        out.append('<path d="M' + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts) +
                   f'" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
        for (a, b), xv, yv in zip(pts, s["x"], s["y"]):
            tip = f'{s["name"]} · {xfmt(xv) if xfmt else _fmt(xv)} {xlabel.split(" (")[0].lower()}: {_fmt(yv, pct, dp)}'
            out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{col}" stroke="var(--panel)" stroke-width="2" '
                       f'tabindex="0" data-tip="{_e(tip)}"/>')
        if pts:
            ends.append([pts[-1][1], s["name"], pts[-1][0]])
    if end_labels and ends:
        ends.sort()
        prev = -1e9
        for e in ends:                                      # leader lines keep labels 13 px apart
            e.append(max(e[0], prev + 13))
            prev = e[-1]
        for yv, nm, xv, yl in ends:
            out.append(f'<line class="lead" x1="{xv + 6:.1f}" y1="{yv:.1f}" x2="{ml + pw + 10:.1f}" y2="{yl:.1f}"/>'
                       f'<text class="lab" x="{ml + pw + 13:.1f}" y="{yl + 3.5:.1f}">{_e(nm)}</text>')
    out.append("</svg>")
    return "".join(out)


def dots(rows, *, xlabel, xmin, xmax, band=None, pct=True, width=720, row_h=26, label_w=230, ref=None):
    """Horizontal dot-and-interval rows: dict(label, v, lo, hi, slot=0, group=None)."""
    mt, mb, mr = 10, 40, 16
    ml = label_w
    height = mt + mb + row_h * len(rows)
    pw = width - ml - mr
    x = lambda v: ml + pw * (v - xmin) / (xmax - xmin)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    if band:
        out.append(f'<rect x="{x(band[0]):.1f}" y="{mt}" width="{x(band[1]) - x(band[0]):.1f}" '
                   f'height="{row_h * len(rows)}" fill="var(--good)" opacity="0.10"/>')
    for t in nice_ticks(xmin, xmax):
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{mt}" y2="{mt + row_h * len(rows)}"/>'
                   f'<text class="tick" x="{x(t):.1f}" y="{mt + row_h * len(rows) + 16}" text-anchor="middle">'
                   f'{_fmt(t, pct)}</text>')
    for xv, lab in (ref or []):
        out.append(f'<line class="refl" x1="{x(xv):.1f}" x2="{x(xv):.1f}" y1="{mt}" y2="{mt + row_h * len(rows)}"/>')
    out.append(f'<text class="axl" x="{ml + pw / 2:.0f}" y="{height - 6}" text-anchor="middle">{_e(xlabel)}</text>')
    for i, r in enumerate(rows):
        cy = mt + row_h * i + row_h / 2
        col = SLOTS[r.get("slot", 0)]
        out.append(f'<text class="lab" x="{ml - 10}" y="{cy + 4:.1f}" text-anchor="end">{_e(r["label"])}</text>')
        if r.get("lo") is not None and not math.isnan(r["lo"]):
            lo, hi = max(r["lo"], xmin), min(r["hi"], xmax)
            out.append(f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{col}" '
                       f'stroke-width="2" stroke-linecap="round"/>')
        v = min(max(r["v"], xmin), xmax)
        tip = r.get("tip") or f'{r["label"]}: {_fmt(r["v"], pct, 1)}'
        out.append(f'<circle cx="{x(v):.1f}" cy="{cy:.1f}" r="5" fill="{col}" stroke="var(--panel)" '
                   f'stroke-width="2" tabindex="0" data-tip="{_e(tip)}"/>')
    out.append("</svg>")
    return "".join(out)
