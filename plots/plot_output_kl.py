from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@dataclass
class DistributionSpec:
    path: Path
    probs: np.ndarray
    kind: str
    ids: np.ndarray | None = None

    @property
    def num_examples(self) -> int:
        return int(self.probs.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.probs.shape[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist_a", required=True, help="First distribution .npy path.")
    parser.add_argument("--dist_b", required=True, help="Second distribution .npy path.")
    parser.add_argument("--label_a", default="A", help="Legend label for the first distribution.")
    parser.add_argument("--label_b", default="B", help="Legend label for the second distribution.")
    parser.add_argument(
        "--direction",
        choices=["a_to_b", "b_to_a", "both"],
        default="both",
        help="Which KL direction(s) to plot.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output image path. If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Small value used to stabilize KL when q has zeros.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on number of examples used from each file.",
    )
    return parser.parse_args()


def _infer_sparse_ids_path(path: Path) -> Path | None:
    candidate_names = []
    if "token_probs" in path.name:
        candidate_names.append(path.name.replace("token_probs", "token_ids"))
    if "logprobs" in path.name:
        candidate_names.append(path.name.replace("logprobs", "token_ids"))

    for candidate_name in candidate_names:
        candidate = path.with_name(candidate_name)
        if candidate.exists():
            return candidate
    return None


def load_distribution(path_str: str) -> DistributionSpec:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Distribution file not found: {path}")

    probs = np.load(path)
    if probs.ndim == 2:
        raise ValueError(
            f"{path} has shape {probs.shape}. This looks like chosen-token scores, not a token "
            "distribution tensor. Use a full distribution file such as `token_distributions.npy` "
            "or `topk_token_probs.npy`."
        )
    if probs.ndim != 3:
        raise ValueError(f"{path} must be a rank-3 array, got shape {probs.shape}.")

    ids_path = _infer_sparse_ids_path(path)
    if ids_path is not None:
        ids = np.load(ids_path)
        if ids.shape != probs.shape:
            raise ValueError(
                f"Sparse ids/probs shape mismatch for {path} and {ids_path}: "
                f"{probs.shape} vs {ids.shape}"
            )
        return DistributionSpec(path=path, probs=probs, ids=ids, kind="sparse_topk")

    return DistributionSpec(path=path, probs=probs, kind="dense")


def align_distributions(
    dist_a: DistributionSpec,
    dist_b: DistributionSpec,
    max_examples: int | None,
) -> tuple[DistributionSpec, DistributionSpec]:
    num_examples = min(dist_a.num_examples, dist_b.num_examples)
    num_steps = min(dist_a.num_steps, dist_b.num_steps)

    if max_examples is not None:
        num_examples = min(num_examples, max_examples)

    if num_examples == 0 or num_steps == 0:
        raise ValueError("The aligned distributions have zero examples or zero steps.")

    if dist_a.num_examples != dist_b.num_examples or dist_a.num_steps != dist_b.num_steps:
        print(
            f"Aligning arrays to first {num_examples} examples and first {num_steps} steps "
            f"(A: {dist_a.probs.shape}, B: {dist_b.probs.shape})."
        )

    aligned_a = DistributionSpec(
        path=dist_a.path,
        probs=dist_a.probs[:num_examples, :num_steps],
        ids=None if dist_a.ids is None else dist_a.ids[:num_examples, :num_steps],
        kind=dist_a.kind,
    )
    aligned_b = DistributionSpec(
        path=dist_b.path,
        probs=dist_b.probs[:num_examples, :num_steps],
        ids=None if dist_b.ids is None else dist_b.ids[:num_examples, :num_steps],
        kind=dist_b.kind,
    )
    return aligned_a, aligned_b


def _normalize_dense_step(step_probs: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(step_probs)
    if not np.any(finite):
        return None
    normalized = np.where(finite, step_probs, 0.0).astype(np.float64, copy=False)
    total = normalized.sum()
    if total <= 0:
        return None
    return normalized / total


def _build_sparse_map(step_ids: np.ndarray, step_probs: np.ndarray) -> tuple[dict[int, float], float]:
    token_map: dict[int, float] = {}
    for token_id, prob in zip(step_ids, step_probs):
        if token_id < 0 or not np.isfinite(prob) or prob <= 0:
            continue
        token_map[int(token_id)] = float(prob)

    total = sum(token_map.values())
    tail_mass = max(0.0, 1.0 - total)
    return token_map, tail_mass


def _coarse_distribution_for_support(
    dist: DistributionSpec,
    example_idx: int,
    step_idx: int,
    support_tokens: list[int],
) -> np.ndarray | None:
    if dist.kind == "dense":
        dense_step = _normalize_dense_step(dist.probs[example_idx, step_idx])
        if dense_step is None:
            return None
        explicit_probs = np.array([dense_step[token_id] for token_id in support_tokens], dtype=np.float64)
        tail_mass = max(0.0, 1.0 - explicit_probs.sum())
        coarse = np.concatenate([explicit_probs, np.array([tail_mass], dtype=np.float64)])
    elif dist.kind == "sparse_topk":
        assert dist.ids is not None
        token_map, tail_mass = _build_sparse_map(
            dist.ids[example_idx, step_idx],
            dist.probs[example_idx, step_idx],
        )
        explicit_probs = np.array([token_map.get(token_id, 0.0) for token_id in support_tokens], dtype=np.float64)
        coarse = np.concatenate([explicit_probs, np.array([tail_mass], dtype=np.float64)])
    else:
        raise ValueError(f"Unsupported distribution kind: {dist.kind}")

    total = coarse.sum()
    if total <= 0:
        return None
    return coarse / total


def _step_support_tokens(
    dist_a: DistributionSpec,
    dist_b: DistributionSpec,
    example_idx: int,
    step_idx: int,
) -> list[int]:
    support_tokens: set[int] = set()

    if dist_a.kind == "sparse_topk":
        assert dist_a.ids is not None
        support_tokens.update(
            int(token_id)
            for token_id in dist_a.ids[example_idx, step_idx]
            if token_id >= 0
        )
    if dist_b.kind == "sparse_topk":
        assert dist_b.ids is not None
        support_tokens.update(
            int(token_id)
            for token_id in dist_b.ids[example_idx, step_idx]
            if token_id >= 0
        )

    return sorted(support_tokens)


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    valid = p > 0
    if not np.any(valid):
        return np.nan

    p_safe = np.clip(p[valid], eps, None)
    q_safe = np.clip(q[valid], eps, None)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


def compute_kl_per_step(
    dist_p: DistributionSpec,
    dist_q: DistributionSpec,
    eps: float,
) -> np.ndarray:
    num_examples, num_steps = dist_p.probs.shape[:2]
    step_values = np.full((num_examples, num_steps), np.nan, dtype=np.float64)

    if dist_p.kind == "dense" and dist_q.kind == "dense":
        for example_idx in range(num_examples):
            for step_idx in range(num_steps):
                p_step = _normalize_dense_step(dist_p.probs[example_idx, step_idx])
                q_step = _normalize_dense_step(dist_q.probs[example_idx, step_idx])
                if p_step is None or q_step is None:
                    continue
                step_values[example_idx, step_idx] = _kl_divergence(p_step, q_step, eps)
    else:
        for example_idx in range(num_examples):
            for step_idx in range(num_steps):
                support_tokens = _step_support_tokens(dist_p, dist_q, example_idx, step_idx)
                p_step = _coarse_distribution_for_support(dist_p, example_idx, step_idx, support_tokens)
                q_step = _coarse_distribution_for_support(dist_q, example_idx, step_idx, support_tokens)
                if p_step is None or q_step is None:
                    continue
                step_values[example_idx, step_idx] = _kl_divergence(p_step, q_step, eps)

    return np.nanmean(step_values, axis=0)


def build_default_title(args: argparse.Namespace, dist_a: DistributionSpec, dist_b: DistributionSpec) -> str:
    mode = {
        "a_to_b": f"KL({args.label_a} || {args.label_b})",
        "b_to_a": f"KL({args.label_b} || {args.label_a})",
        "both": f"KL Between {args.label_a} and {args.label_b}",
    }[args.direction]

    if dist_a.kind == "dense" and dist_b.kind == "dense":
        suffix = "exact full-vocab"
    else:
        suffix = "top-k / mixed approximate"
    return f"{mode} by Output Token Step ({suffix})"


def _svg_polyline_points(x_values: np.ndarray, y_values: np.ndarray, x0: float, y0: float, width: float, height: float, ymin: float, ymax: float) -> str:
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    x_range = max(x_max - x_min, 1.0)
    y_range = max(ymax - ymin, 1e-12)

    points = []
    for x_value, y_value in zip(x_values, y_values):
        px = x0 + width * ((float(x_value) - x_min) / x_range)
        py = y0 + height * (1.0 - ((float(y_value) - ymin) / y_range))
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def save_svg_plot(
    output_path: Path,
    token_steps: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    title: str,
) -> None:
    finite_values = np.concatenate([values[np.isfinite(values)] for _, values in series])
    if finite_values.size == 0:
        raise ValueError("All KL values are NaN; nothing to plot.")

    ymin = min(0.0, float(np.min(finite_values)))
    ymax = float(np.max(finite_values))
    if ymax <= ymin:
        ymax = ymin + 1.0

    width = 960
    height = 540
    margin_left = 90
    margin_right = 30
    margin_top = 50
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    y_ticks = np.linspace(ymin, ymax, 5)
    x_ticks = token_steps

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-size="20" font-family="sans-serif">{escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="black" />',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black" />',
    ]

    for y_tick in y_ticks:
        py = margin_top + plot_height * (1.0 - ((float(y_tick) - ymin) / max(ymax - ymin, 1e-12)))
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{py:.2f}" x2="{margin_left + plot_width}" y2="{py:.2f}" stroke="#dddddd" />'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{py + 4:.2f}" text-anchor="end" font-size="12" font-family="sans-serif">{y_tick:.4f}</text>'
        )

    for x_tick in x_ticks:
        px = margin_left + plot_width * ((float(x_tick) - float(token_steps[0])) / max(float(token_steps[-1] - token_steps[0]), 1.0))
        svg_parts.append(
            f'<line x1="{px:.2f}" y1="{margin_top}" x2="{px:.2f}" y2="{margin_top + plot_height}" stroke="#f0f0f0" />'
        )
        svg_parts.append(
            f'<text x="{px:.2f}" y="{margin_top + plot_height + 20}" text-anchor="middle" font-size="12" font-family="sans-serif">{int(x_tick)}</text>'
        )

    for idx, (label, values) in enumerate(series):
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            continue
        x_vals = token_steps[finite_mask]
        y_vals = values[finite_mask]
        points = _svg_polyline_points(
            x_vals,
            y_vals,
            x0=margin_left,
            y0=margin_top,
            width=plot_width,
            height=plot_height,
            ymin=ymin,
            ymax=ymax,
        )
        color = colors[idx % len(colors)]
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}" />'
        )
        for x_value, y_value in zip(x_vals, y_vals):
            px, py = _svg_polyline_points(
                np.array([x_value]),
                np.array([y_value]),
                x0=margin_left,
                y0=margin_top,
                width=plot_width,
                height=plot_height,
                ymin=ymin,
                ymax=ymax,
            ).split()[0].split(",")
            svg_parts.append(f'<circle cx="{px}" cy="{py}" r="3.5" fill="{color}" />')

    legend_x = margin_left + 12
    legend_y = margin_top + 12
    for idx, (label, _) in enumerate(series):
        color = colors[idx % len(colors)]
        y = legend_y + idx * 22
        svg_parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 18}" y2="{y}" stroke="{color}" stroke-width="3" />')
        svg_parts.append(
            f'<text x="{legend_x + 24}" y="{y + 4}" font-size="12" font-family="sans-serif">{escape(label)}</text>'
        )

    svg_parts.extend(
        [
            f'<text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="sans-serif">Token Step</text>',
            f'<text x="20" y="{height / 2:.0f}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90 20 {height / 2:.0f})">KL Divergence</text>',
            "</svg>",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg_parts))


def main() -> None:
    args = parse_args()

    dist_a = load_distribution(args.dist_a)
    dist_b = load_distribution(args.dist_b)
    dist_a, dist_b = align_distributions(dist_a, dist_b, args.max_examples)

    token_steps = np.arange(1, dist_a.num_steps + 1)

    series: list[tuple[str, np.ndarray]] = []

    if args.direction in {"a_to_b", "both"}:
        kl_a_to_b = compute_kl_per_step(dist_a, dist_b, args.eps)
        series.append((f"KL({args.label_a} || {args.label_b})", kl_a_to_b))

    if args.direction in {"b_to_a", "both"}:
        kl_b_to_a = compute_kl_per_step(dist_b, dist_a, args.eps)
        series.append((f"KL({args.label_b} || {args.label_a})", kl_b_to_a))

    title = args.title or build_default_title(args, dist_a, dist_b)

    if args.output:
        output_path = Path(args.output)
        if output_path.suffix.lower() == ".svg" or plt is None:
            save_svg_plot(output_path, token_steps, series, title)
        else:
            plt.figure(figsize=(10, 5))
            for label, values in series:
                plt.plot(token_steps, values, marker="o", label=label)
            plt.xlabel("Token Step")
            plt.ylabel("KL Divergence")
            plt.xticks(token_steps)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.title(title)
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")
    else:
        if plt is None:
            raise RuntimeError(
                "matplotlib is not installed in this environment. "
                "Pass `--output some_plot.svg` to use the built-in SVG renderer."
            )
        plt.figure(figsize=(10, 5))
        for label, values in series:
            plt.plot(token_steps, values, marker="o", label=label)
        plt.xlabel("Token Step")
        plt.ylabel("KL Divergence")
        plt.xticks(token_steps)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
