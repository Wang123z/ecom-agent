"""给缺失操作路径的答案补上标准路径，提升 closure。

只补路径，不补时效——时效编错就是幻觉。
已在答案里出现【】的不动。

用法  python -m scripts.augment_closure
      python -m scripts.augment_closure --input data/train.jsonl data/val.jsonl
"""
import argparse
import json
from pathlib import Path

from app import config

# 品类 → 标准操作路径
PATHS = {
    "物流": "在【我的订单】→【查看物流】查看实时节点。",
    "发票": "在【我的订单】→【申请开票】提交申请。",
    "价保": "在【我的订单】→【申请价保】提交申请。",
    "优惠": "在【我的订单】结算页勾选优惠券。",
    "会员": "在【会员中心】查看与操作。",
    "支付": "在【我的订单】重新发起支付。",
    "订单": "在【我的订单】查看与操作。",
    "投诉": "在【我的订单】→【投诉建议】提交工单。",
    "安全": "在【设置】→【账号安全】处理。",
    "预售": "在【我的订单】查看发货进度。",
    "定制": "在【我的订单】→【申请售后】提交。",
}
DEFAULT_PATH = "在【我的订单】→【申请售后】提交。"


def augment(answer: str, category: str) -> str:
    if "【" in answer:
        return answer
    path = PATHS.get(category, DEFAULT_PATH)
    return answer.rstrip("。") + "。" + path


def process(path: Path) -> tuple:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    changed = 0
    for row in rows:
        new = augment(row["answer"], row.get("category", ""))
        if new != row["answer"]:
            row["answer"] = new
            changed += 1
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows), changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+",
                        default=[str(config.DATA_DIR / "train.jsonl"),
                                 str(config.DATA_DIR / "val.jsonl")])
    args = parser.parse_args()

    total = 0
    for name in args.input:
        path = Path(name)
        if not path.exists():
            continue
        n, changed = process(path)
        total += changed
        print(f"{path.name}: {n} 条，补路径 {changed} 条")
    print(f"合计补全 {total} 条")


if __name__ == "__main__":
    main()
