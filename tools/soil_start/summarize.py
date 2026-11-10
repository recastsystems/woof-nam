"""Tables + a wedge/plains Td2 bias chart (analysis chart, not a weather field) from the soil-start arms' region.json."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(sys.argv[1]); arms = sys.argv[2].split(","); out = Path(sys.argv[3])
D = {a: json.loads((R / a / "region.json").read_text()) for a in arms if (R / a / "region.json").is_file()}
nam = next(iter(D.values()))["nam_archive"]
leads = sorted(next(iter(D.values()))["leads"], key=int)
lines = []
for reg in ("wedge", "wedge_urban", "wedge_nonurban", "plains", "conus"):
    for var in ("TD2", "T2"):
        if reg.startswith("wedge_") and var == "T2": continue
        lines.append(f"\n### {reg} {var} (degF) bias / MAE, n\n")
        lines.append("| lead | n | archived NAM | " + " | ".join(D) + " |")
        lines.append("|---|---|---|" + "---|" * len(D))
        for L in leads:
            def f(s): return f"{s['bias']:+.1f} / {s['mae']:.1f}" if s and s.get("n") else "-"
            n = D[next(iter(D))]["leads"][L][reg][var].get("n")
            nm = nam.get(L, {}).get(reg, {}).get(var) if reg in nam.get(L, {}) else None
            lines.append(f"| +{L} | {n} | {f(nm)} | " + " | ".join(f(D[a]['leads'][L][reg][var]) for a in D) + " |")
out.with_suffix(".md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
for k, reg in enumerate(("wedge", "plains", "conus")):
    x = [int(L) for L in leads]
    ax[k].plot(x, [nam[L][reg]["TD2"]["bias"] for L in leads], "k--", lw=2, label="archived NAM")
    for a in D: ax[k].plot(x, [D[a]["leads"][L][reg]["TD2"]["bias"] for L in leads], "-o", label=a)
    ax[k].axhline(0, color="0.6", lw=0.8); ax[k].set_title(f"{reg}: 2 m dewpoint bias vs ASOS (F)"); ax[k].set_xlabel("lead (h), GFS start 2026-01-23 12Z")
ax[0].legend(); fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=110)
