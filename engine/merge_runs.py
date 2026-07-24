"""Merge raw.jsonl from several result dirs into one new dir, so a newly-added model can be
judged head-to-head against an earlier run instead of re-generating everything.

Judging is comparative: judge.py groups candidates by (item_id, condition) and scores them
against each other. So adding a model means the WHOLE group must be re-judged — but only the
new model's generations need to be re-run. Merge first, then judge the merged dir.

Later dirs win on collision (same model_id + item_id + context), so re-running one model
replaces its old rows rather than duplicating them.

Usage: python engine/merge_runs.py <out_dir> <in_dir_1> <in_dir_2> [...]
"""
import json
import pathlib
import sys


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    out = pathlib.Path(sys.argv[1])
    ins = [pathlib.Path(p) for p in sys.argv[2:]]

    merged, order = {}, []
    for d in ins:
        raw = d / "raw.jsonl"
        if not raw.exists():
            print(f"!! no raw.jsonl in {d} — skipping")
            continue
        n = 0
        for line in raw.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r.get("model_id"), r.get("item_id"), r.get("context"))
            if key not in merged:
                order.append(key)
            merged[key] = r          # later dir wins
            n += 1
        print(f"  + {n:5d} rows from {d.name}")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "raw.jsonl", "w", encoding="utf-8") as fh:
        for key in order:
            fh.write(json.dumps(merged[key], ensure_ascii=False) + "\n")

    models = sorted({k[0] for k in merged})
    items = sorted({k[1] for k in merged})
    conds = sorted({k[2] for k in merged})
    (out / "meta.json").write_text(json.dumps({
        "task": "merged", "merged_from": [str(p) for p in ins], "models": models,
        "contexts": conds, "n_items": len(items), "n_tasks": len(merged)}, indent=2))
    print(f"\n{len(merged)} rows -> {out}\n  models: {models}\n  contexts: {conds}  items: {len(items)}")
    print(f"Next: python engine/judge.py <task_dir> {out}")


if __name__ == "__main__":
    main()
