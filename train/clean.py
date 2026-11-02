"""数据清洗：标准化 → 精确去重 → 相似去重 → 低价值过滤 → 质量排序 → 8:2 切分。

输入  data/faq_raw.jsonl   每行 {"id","question","answer","category"}
输出  data/train.jsonl / data/val.jsonl
用法  python -m train.clean
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from app import config
from train.sim import cos, encode_all, jaccard_text

_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
_SPACE = re.compile(r"\s+")
_DATE_PATTERNS = [
    (re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日"), r"\1-\2-\3"),
    (re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})"), r"\1-\2-\3"),
    (re.compile(r"(\d{1,2})月(\d{1,2})日"), r"\1-\2"),
]
_CONDITION_WORDS = ("如果", "若", "超过", "以内", "需要", "可以", "不能")


def clean_text(s: str) -> str:
    """去 emoji、统一日期格式、压缩空白。

    不做全角转半角：中文标点与【】这类强调符是客服话术的正常写法，
    转成半角反而让训练数据不自然，也和推理时的输入格式不一致。
    """
    s = _EMOJI.sub("", s or "")
    for pattern, repl in _DATE_PATTERNS:
        s = pattern.sub(repl, s)
    return _SPACE.sub(" ", s).strip()


def dedup_key(s: str) -> str:
    """精确去重键：去掉所有标点空白后取 md5。"""
    bare = re.sub(r"[\W_]+", "", s.lower())
    return hashlib.md5(bare.encode("utf-8")).hexdigest()


def bigrams(s: str) -> set:
    s = re.sub(r"[\W_]+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def similarity(a: str, b: str) -> float:
    """字符 bigram 的 Jaccard 相似度，单对文本用。"""
    return jaccard_text(a, b)


def quality(row: dict) -> tuple:
    """质量分：答案长度 + 是否含具体时效 + 是否含条件判断 + 问题长度是否合理。"""
    answer, question = row["answer"], row["question"]
    score = len(answer)
    if re.search(r"\d", answer):
        score += 50
    if any(w in answer for w in _CONDITION_WORDS):
        score += 30
    if 6 <= len(question) <= 40:
        score += 20
    return (score, -len(answer))


def load_raw(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_qa(row: dict) -> dict:
    """兼容两种输入：本项目格式 {question,answer} 与 ModelScope 的 {messages}。"""
    if "messages" not in row:
        return row
    msgs = [m for m in row["messages"] if m.get("role") in ("user", "assistant")]
    if len(msgs) < 2:
        return {}
    return {"question": msgs[0].get("content", ""), "answer": msgs[-1].get("content", "")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.DATA_DIR / "faq_raw.jsonl"))
    parser.add_argument("--output-dir", default=str(config.DATA_DIR))
    parser.add_argument("--min-question", type=int, default=4,
                        help="中文短问句常见 5-6 字，阈值过大会误杀")
    parser.add_argument("--min-answer", type=int, default=15)
    parser.add_argument("--sim-threshold", type=float, default=0.9)
    parser.add_argument("--sim-backend", choices=["bigram", "bge"], default="bigram",
                        help="bigram 只能合并字面相近的；bge 需要 sentence-transformers")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--keywords", default="",
                        help="逗号分隔，命中任一才保留；用于从通用客服语料里筛售后子集")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="最终保留条数上限，0 表示不限制；用于对齐参考项目的数据规模")
    parser.add_argument("--skip-similar", action="store_true",
                        help="跳过 O(n²) 相似去重；数据量上万时必开，否则要跑几小时")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [to_qa(r) for r in load_raw(Path(args.input))]
    rows = [r for r in rows if r]
    raw_count = len(rows)
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

    # 1. 标准化 + 低价值过滤
    kept = []
    for row in rows:
        q = clean_text(row.get("question", ""))
        a = clean_text(row.get("answer", ""))
        if len(q) < args.min_question or len(a) < args.min_answer:
            continue
        if keywords and not any(k in q for k in keywords):
            continue
        kept.append({"id": row.get("id") or dedup_key(q)[:8], "question": q,
                     "answer": a, "category": row.get("category", "")})
    after_filter = len(kept)

    # 2. 精确去重
    seen, unique = set(), []
    for row in kept:
        key = dedup_key(row["question"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    after_exact = len(unique)

    # 3. 相似去重：保留质量分更高的那条
    unique.sort(key=quality, reverse=True)
    if args.skip_similar:
        final = unique
    else:
        vectors = encode_all([row["question"] for row in unique], args.sim_backend)
        final, kept_vectors = [], []
        for row, vec in zip(unique, vectors):
            if any(cos(vec, other, args.sim_backend) >= args.sim_threshold
                   for other in kept_vectors):
                continue
            final.append(row)
            kept_vectors.append(vec)
    after_similar = len(final)

    # 4. 截断到目标规模（按质量分保留最好的）
    if args.max_samples and len(final) > args.max_samples:
        final = final[:args.max_samples]

    # 5. 切分
    random.Random(args.seed).shuffle(final)
    cut = int(len(final) * (1 - args.val_ratio))
    train, val = final[:cut], final[cut:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train.jsonl", train), ("val.jsonl", val)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "raw": raw_count,
        "filtered_low_value": raw_count - after_filter,
        "dedup_exact": after_filter - after_exact,
        "dedup_similar": after_exact - after_similar,
        "train": len(train),
        "val": len(val),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
