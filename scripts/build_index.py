"""建索引：FAQ 精确匹配表 + 政策文档分块入库。

用法
  python -m scripts.build_index --dry-run     # 只校验数据并打印统计，不连 Milvus
  python -m scripts.build_index               # 真实入库
  python -m scripts.build_index --rebuild     # 先删集合再重建
"""
import argparse
import json
import re
from pathlib import Path

from app import config
from app.core.rag.retriever import normalize

MAX_CHARS = 400
OVERLAP = 50


def load_faq(path: Path) -> dict:
    """FAQ 精确匹配表：归一化问题 → 标准答案。"""
    index = {}
    if not path.exists():
        return index
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        index[normalize(row["question"])] = {
            "id": row["id"],
            "answer": row["answer"],
            "doc": "FAQ",
            "category": row.get("category", ""),
        }
    return index


def split_paragraphs(text: str) -> list:
    """按空行切段，合并到 MAX_CHARS 以内，段间留 OVERLAP 字符重叠。"""
    raw = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for para in raw:
        if len(buf) + len(para) <= MAX_CHARS:
            buf = f"{buf}\n{para}".strip()
            continue
        if buf:
            chunks.append(buf)
        buf = (buf[-OVERLAP:] + "\n" + para).strip() if buf else para
    if buf:
        chunks.append(buf)
    return chunks


def split_sections(text: str) -> list:
    """按 markdown 标题切节，返回 [(section, body)]。section 为空表示标题前的引言。"""
    sections, title, buf = [], "", []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf:
                sections.append((title, "\n".join(buf).strip()))
                buf = []
            title = line.lstrip("#").strip()
        else:
            buf.append(line)
    if buf:
        sections.append((title, "\n".join(buf).strip()))
    return [(t, b) for t, b in sections if b]


def load_policy(dir_path: Path) -> list:
    """政策文档 → 待入库行。先按标题切节，节内再按长度分块。"""
    rows = []
    if not dir_path.exists():
        return rows
    for md in sorted(dir_path.glob("*.md")):
        for section, body in split_sections(md.read_text(encoding="utf-8")):
            for chunk in split_paragraphs(body):
                rows.append({"id": f"{md.stem}_{len(rows):04d}", "text": chunk,
                             "doc": md.stem, "section": section or md.stem})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faq", default=str(config.DATA_DIR / "train.jsonl"))
    parser.add_argument("--policy-dir", default=str(config.DATA_DIR / "policy"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    faq_index = load_faq(Path(args.faq))
    faq_path = config.DATA_DIR / "faq_index.json"
    faq_path.parent.mkdir(parents=True, exist_ok=True)
    faq_path.write_text(json.dumps(faq_index, ensure_ascii=False, indent=2), encoding="utf-8")

    policy_rows = load_policy(Path(args.policy_dir))
    stats = {"faq_entries": len(faq_index), "policy_chunks": len(policy_rows),
             "policy_docs": len({r["doc"] for r in policy_rows})}

    if args.dry_run:
        print(json.dumps({**stats, "mode": "dry-run"}, ensure_ascii=False, indent=2))
        for row in policy_rows[:2]:
            print(f"\n[{row['id']}] {row['section']}\n{row['text'][:120]}...")
        return

    if not policy_rows:
        print(json.dumps({**stats, "error": "没有政策文档，先放 md 到 data/policy/"}, ensure_ascii=False))
        return

    from app.infra import embed, milvus

    if args.rebuild:
        milvus.drop_collection()
    milvus.ensure_collection()

    texts = [r["text"] for r in policy_rows]
    dense, sparse = embed.encode(texts, with_sparse=True)
    rows = [
        {"id": r["id"], "text": r["text"], "doc": r["doc"], "section": r["section"],
         "dense": dense[i], "sparse": sparse[i]}
        for i, r in enumerate(policy_rows)
    ]
    inserted = milvus.insert(rows)
    print(json.dumps({**stats, "inserted": inserted}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
