"""从 trainer_state.json 重绘 loss 曲线，一张图对比多组实验。

用法
  python -m scripts.plot_loss --root outputs/factory
  python -m scripts.plot_loss --root outputs/factory --metric eval_loss
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_curve(state_file: Path, metric: str) -> tuple:
    data = json.loads(state_file.read_text(encoding="utf-8"))
    points = data.get("log_history", [])
    xs = [p["step"] for p in points if metric in p]
    ys = [p[metric] for p in points if metric in p]
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/factory")
    parser.add_argument("--metric", default="loss")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.root)
    states = sorted(root.glob("*/trainer_state.json"))
    if not states:
        print(f"{root} 下没有 trainer_state.json，先跑训练")
        return

    plt.figure(figsize=(10, 6))
    for state in states:
        xs, ys = load_curve(state, args.metric)
        if xs:
            plt.plot(xs, ys, label=state.parent.name, linewidth=1.5)

    plt.xlabel("step")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric} 曲线对比")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = Path(args.output) if args.output else root / f"{args.metric}.png"
    plt.savefig(out, dpi=150)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
