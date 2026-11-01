"""检查模型是否完整，缺失则用 ModelScope 下载。可重复执行。

用法
  python scripts/download_models.py                      # 全部检查并补齐
  python scripts/download_models.py --only Qwen3-0.6B    # 只处理指定模型
  python scripts/download_models.py --root /root/autodl-tmp/models
"""
import argparse
import subprocess
import sys
from pathlib import Path

MODELS = {
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "bge-m3": "BAAI/bge-m3",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}


def is_complete(path: Path) -> bool:
    """config.json 在，且至少有一个权重分片。"""
    if not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/autodl-tmp/models")
    parser.add_argument("--only", action="append", default=[], help="只处理指定模型，可重复")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    targets = args.only or list(MODELS)
    for name in targets:
        if name not in MODELS:
            print(f"[skip] 未知模型 {name}")
            continue
        local = root / name
        if is_complete(local):
            print(f"[ok]   {name} 已完整，跳过")
            continue
        print(f"[down] {name} -> {local}")
        result = subprocess.run(
            [sys.executable, "-m", "modelscope", "download",
             "--model", MODELS[name], "--local_dir", str(local)],
            check=False,
        )
        if result.returncode != 0:
            print(f"[fail] {name} 下载失败，退出码 {result.returncode}")
            continue
        print(f"[done] {name} {'完整' if is_complete(local) else '仍不完整，请检查'}")


if __name__ == "__main__":
    main()
