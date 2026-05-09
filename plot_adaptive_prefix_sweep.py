from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ASR over custom adaptive-prefix prefill length from eval_safety_tinker JSON results."
    )
    parser.add_argument(
        "--result_template",
        default="logs/adaptive_prefixes/eval_augmented_on_mined_prefixes_k{k}.json",
        help="Path template containing `{k}` for each prefix length.",
    )
    parser.add_argument("--ks", default="5,10,15,20", help="Comma-separated prefix lengths.")
    parser.add_argument(
        "--save_path",
        default="figures/adaptive_prefix_sweep.png",
        help="PNG output path. If matplotlib is unavailable, an SVG is written next to it.",
    )
    parser.add_argument(
        "--csv_path",
        default="logs/adaptive_prefixes/adaptive_prefix_sweep.csv",
        help="CSV summary path.",
    )
    parser.add_argument("--title", default="Adaptive Prefix Prefill Attack")
    return parser.parse_args()


def parse_ks(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def load_points(result_template: str, ks: list[int]) -> list[dict]:
    points = []
    for k in ks:
        path = Path(result_template.format(k=k))
        with path.open() as handle:
            result = json.load(handle)
        metrics = result.get("metrics") or {}
        points.append(
            {
                "k": k,
                "path": str(path),
                "asr": float(metrics.get("asr", 0.0)),
                "num_success": int(metrics.get("num_success", 0)),
                "num_tot": int(metrics.get("num_tot", 0)),
            }
        )
    return points


def write_csv(points: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "asr", "num_success", "num_tot", "path"])
        writer.writeheader()
        writer.writerows(points)


def write_svg(points: list[dict], save_path: Path, title: str) -> Path:
    svg_path = save_path.with_suffix(".svg")
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 720, 420
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 50, 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    ks = [point["k"] for point in points]
    asrs = [point["asr"] for point in points]
    x_min, x_max = min(ks), max(ks)
    y_max = max(0.1, min(1.0, max(asrs) * 1.2))

    def x_pos(k: int) -> float:
        if x_max == x_min:
            return margin_left + plot_w / 2
        return margin_left + (k - x_min) / (x_max - x_min) * plot_w

    def y_pos(asr: float) -> float:
        return margin_top + plot_h - (asr / y_max) * plot_h

    line_points = " ".join(f"{x_pos(k):.1f},{y_pos(asr):.1f}" for k, asr in zip(ks, asrs))
    circles = "\n".join(
        f'<circle cx="{x_pos(point["k"]):.1f}" cy="{y_pos(point["asr"]):.1f}" r="5" fill="#2563eb" />'
        for point in points
    )
    labels = "\n".join(
        f'<text x="{x_pos(point["k"]):.1f}" y="{height - 25}" text-anchor="middle" font-size="13">{point["k"]}</text>'
        for point in points
    )
    value_labels = "\n".join(
        f'<text x="{x_pos(point["k"]):.1f}" y="{y_pos(point["asr"]) - 10:.1f}" text-anchor="middle" font-size="12">{point["asr"]:.2f}</text>'
        for point in points
    )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2:.0f}" y="28" text-anchor="middle" font-size="18" font-family="sans-serif">{title}</text>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#111827" />
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#111827" />
  <text x="{width / 2:.0f}" y="{height - 5}" text-anchor="middle" font-size="14" font-family="sans-serif">Prefill prefix length k</text>
  <text x="18" y="{height / 2:.0f}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90 18 {height / 2:.0f})">ASR</text>
  <polyline points="{line_points}" fill="none" stroke="#2563eb" stroke-width="3" />
  {circles}
  {labels}
  {value_labels}
</svg>
'''
    svg_path.write_text(svg)
    return svg_path


def write_plot(points: list[dict], save_path: Path, title: str) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return write_svg(points, save_path, title)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    ks = [point["k"] for point in points]
    asrs = [point["asr"] for point in points]

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(ks, asrs, marker="o", linewidth=2.5)
    for k, asr in zip(ks, asrs):
        plt.annotate(f"{asr:.2f}", (k, asr), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.title(title)
    plt.xlabel("Prefill prefix length k")
    plt.ylabel("ASR")
    plt.ylim(0, min(1.0, max(0.1, max(asrs) * 1.2)))
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def main() -> None:
    args = parse_args()
    ks = parse_ks(args.ks)
    points = load_points(args.result_template, ks)
    write_csv(points, Path(args.csv_path))
    plot_path = write_plot(points, Path(args.save_path), args.title)
    print(json.dumps({"points": points, "plot_path": str(plot_path), "csv_path": args.csv_path}, indent=2))


if __name__ == "__main__":
    main()
