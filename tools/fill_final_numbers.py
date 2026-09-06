"""Merge the human-language benchmark runs and fill the documentation numbers.

The benchmark can be run per level in parallel processes, each writing its own
JSON. This tool merges them into ``results/human_language_benchmark.json`` and
replaces the ``{{...}}`` placeholders in the README, the playbook, and the talk
track with the measured values so no document quotes a number that was not
produced by the tool.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCS = (
    "README.md",
    "docs/GRAND_FINAL_PLAYBOOK.md",
    "docs/GRAND_FINAL_TALK_TRACK_EN.md",
)

LEVEL_LABEL = {
    "canonical": "Organizer templates (control)",
    "natural": "Natural wording, attributes verbatim",
    "paraphrase": "Natural wording, attributes paraphrased",
}
LEVEL_LABEL_ZH = {
    "canonical": "官方模板（对照）",
    "natural": "口语化措辞，属性原话保留",
    "paraphrase": "口语化措辞，属性也用自己的话说",
}


def merge(paths: list[Path]) -> dict:
    merged: dict = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not merged:
            merged = {key: value for key, value in data.items() if key not in ("experiments", "sample_rewrites", "levels")}
            merged["levels"] = []
            merged["experiments"] = {}
            merged["sample_rewrites"] = {}
        merged["levels"].extend(level for level in data["levels"] if level not in merged["levels"])
        merged["experiments"].update(data["experiments"])
        merged["sample_rewrites"].update(data.get("sample_rewrites", {}))
    return merged


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def table(merged: dict, zh: bool = False) -> str:
    labels = LEVEL_LABEL_ZH if zh else LEVEL_LABEL
    head = (
        "| 顾客措辞 | 臂 | Hit@10 | MRR | MTTC | TechnicalScore | 模型调用 | tokens |\n"
        if zh else
        "| Shopper wording | Arm | Hit@10 | MRR | MTTC | TechnicalScore | Model calls | Tokens |\n"
    )
    lines = [head, "|---|---|---:|---:|---:|---:|---:|---:|\n"]
    for level in merged["levels"]:
        for arm in merged["arms"]:
            row = merged["experiments"].get(f"{level}__{arm}")
            if not row:
                continue
            arm_label = ("deterministic (frozen submission)" if arm == "deterministic" else "hybrid (LLM grounding)")
            if zh:
                arm_label = "deterministic（提交冻结版）" if arm == "deterministic" else "hybrid（LLM grounding）"
            calls = row["grounding"]["model_calls"]
            tokens = row["reported_token_usage"]["total_tokens"]
            bold = arm == "hybrid" and level != "canonical"
            score = fmt(row["technical_score"], 6)
            lines.append(
                f"| {labels[level]} | {arm_label} | {fmt(row['hit_rate_at_10'])} | {fmt(row['mrr'])} | "
                f"{row['mttc']:.3f} | {'**' + score + '**' if bold else score} | {calls} | {tokens:,} |\n"
            )
    return "".join(lines)


def notes(merged: dict, zh: bool = False) -> str:
    n = merged["sample_count"]
    hyb = {level: merged["experiments"].get(f"{level}__hybrid") for level in merged["levels"]}
    det = {level: merged["experiments"].get(f"{level}__deterministic") for level in merged["levels"]}
    per_session = []
    for level, row in hyb.items():
        if row and level != "canonical":
            per_session.append((level, row["grounding"]["model_calls"] / n, row["reported_token_usage"]["total_tokens"] / n, row["grounding"]["mean_model_latency_ms_per_turn"]))
    gm = merged["grounding_model"]["model"]
    cm = merged["customer_model"]["model"]
    rewrites = merged.get("sample_rewrites", {})
    example = ""
    for level in ("paraphrase", "natural"):
        rows = rewrites.get(level) or []
        if rows:
            example = rows[0]
            break
    if zh:
        text = (
            f"- {n} 个公开 session；顾客改写模型 `{cm}`，grounding 模型 `{gm}`（同一本地 vLLM 服务；改写只看模板消息，不看目录）。\n"
            f"- 官方模板下 hybrid 与 deterministic 逐字节同分、0 次调用：协议句式从不到达模型。\n"
        )
        for level, calls, toks, lat in per_session:
            text += f"- {LEVEL_LABEL_ZH[level]}：hybrid 每 session 平均 {calls:.2f} 次调用、{toks:,.0f} tokens，单次平均 {lat:.0f} ms。\n"
        if example:
            text += f"- 改写样例：模拟器「{example['simulator'][:90]}」→ 顾客「{example['human'][:90]}」。\n"
        text += "- 这是研究诊断，不是主办方分数；顾客仍是模型扮演的。\n"
        return text
    text = (
        f"- {n} public sessions; customer rewrites by `{cm}`, grounding by `{gm}` (the same local vLLM service; the rewriter sees only the template message, never the catalog).\n"
        f"- On the organizer templates the hybrid arm makes zero model calls and reproduces the deterministic score exactly: protocol wording never reaches the model.\n"
    )
    for level, calls, toks, lat in per_session:
        text += f"- {LEVEL_LABEL[level]}: the hybrid arm averages {calls:.2f} model calls and {toks:,.0f} tokens per session, {lat:.0f} ms per call.\n"
    if example:
        text += f"- Example rewrite: simulator “{example['simulator'][:90]}” → shopper “{example['human'][:90]}”.\n"
    text += "- These are research diagnostics with a model playing the shopper, not organizer scores.\n"
    return text


def html_blocks(merged: dict) -> dict[str, str]:
    """HTML fragments for the playbook page: bars, table, notes."""
    import html as html_module

    esc = html_module.escape
    ex = merged["experiments"]
    n = merged["sample_count"]
    labels = LEVEL_LABEL_ZH
    bars = ['<div class="bars">']
    for level in merged["levels"]:
        if level == "canonical":
            continue
        for arm in ("deterministic", "hybrid"):
            row = ex.get(f"{level}__{arm}")
            if not row:
                continue
            width = max(3, round(row["technical_score"] * 100))
            label = f"{labels[level]} · {'deterministic' if arm == 'deterministic' else 'hybrid'}"
            bars.append(
                f'<div class="bar {"det" if arm == "deterministic" else ""}"><span class="lbl">{esc(label)}</span>'
                f'<div class="track"><div class="fill" style="width:{width}%"></div></div>'
                f'<span class="val">{row["technical_score"]:.4f}</span></div>'
            )
    bars.append("</div>")
    rows = ['<table><thead><tr><th>顾客措辞</th><th>臂</th><th class="num">Hit@10</th><th class="num">MRR</th>'
            '<th class="num">MTTC</th><th class="num">TechnicalScore</th><th class="num">模型调用</th><th class="num">tokens</th></tr></thead><tbody>']
    for level in merged["levels"]:
        for arm in merged["arms"]:
            row = ex.get(f"{level}__{arm}")
            if not row:
                continue
            arm_label = "deterministic（提交冻结版）" if arm == "deterministic" else "hybrid（LLM grounding）"
            bold = arm == "hybrid" and level != "canonical"
            score = f"{row['technical_score']:.6f}"
            rows.append(
                f"<tr><td>{esc(labels[level])}</td><td>{esc(arm_label)}</td><td class=\"num\">{row['hit_rate_at_10']:.4f}</td>"
                f"<td class=\"num\">{row['mrr']:.4f}</td><td class=\"num\">{row['mttc']:.3f}</td>"
                f"<td class=\"num\">{'<b>' + score + '</b>' if bold else score}</td>"
                f"<td class=\"num\">{row['grounding']['model_calls']}</td><td class=\"num\">{row['reported_token_usage']['total_tokens']:,}</td></tr>"
            )
    rows.append("</tbody></table>")
    note_lines = notes(merged, zh=True).strip().splitlines()
    notes_html = "<ul>" + "".join(f"<li>{esc(line.lstrip('- '))}</li>" for line in note_lines) + "</ul>"
    return {"<!--HL_BARS-->": "".join(bars), "<!--HL_TABLE-->": "".join(rows), "<!--HL_NOTES-->": notes_html}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=[
        "results/human_language_benchmark_natural.json",
        "results/human_language_benchmark_paraphrase.json",
    ])
    parser.add_argument("--output", default="results/human_language_benchmark.json")
    parser.add_argument("--no-docs", action="store_true")
    parser.add_argument("--html", help="playbook HTML page whose <!--HL_*--> markers are filled")
    args = parser.parse_args()

    merged = merge([ROOT / path for path in args.inputs])
    order = [level for level in ("canonical", "natural", "paraphrase") if level in merged["levels"]]
    merged["levels"] = order
    (ROOT / args.output).write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(table(merged))
    if args.html:
        page = Path(args.html)
        text = page.read_text(encoding="utf-8")
        for marker, fragment in html_blocks(merged).items():
            text = text.replace(marker, fragment)
        page.write_text(text, encoding="utf-8")
        print(f"filled {page}")
    if args.no_docs:
        return
    ex = merged["experiments"]
    n = merged["sample_count"]
    hyb_para = ex.get("paraphrase__hybrid") or ex.get("natural__hybrid")
    values = {
        "{{HL_TABLE}}": table(merged),
        "{{HL_NOTES}}": notes(merged),
        "{{DET_NATURAL}}": fmt(ex["natural__deterministic"]["technical_score"], 3),
        "{{HYB_NATURAL}}": fmt(ex["natural__hybrid"]["technical_score"], 3),
        "{{DET_PARA}}": fmt(ex["paraphrase__deterministic"]["technical_score"], 3) if "paraphrase__deterministic" in ex else "n/a",
        "{{HYB_PARA}}": fmt(ex["paraphrase__hybrid"]["technical_score"], 3) if "paraphrase__hybrid" in ex else "n/a",
        "{{CALLS_PER_SESSION}}": f"{hyb_para['grounding']['model_calls'] / n:.1f}",
        "{{TOKENS_PER_SESSION}}": f"{hyb_para['reported_token_usage']['total_tokens'] / n:,.0f}",
    }
    for doc in DOCS:
        path = ROOT / doc
        text = path.read_text(encoding="utf-8")
        if doc.endswith("PLAYBOOK.md"):
            local = dict(values)
            local["{{HL_TABLE}}"] = table(merged, zh=True)
            local["{{HL_NOTES}}"] = notes(merged, zh=True)
        else:
            local = values
        for key, value in local.items():
            text = text.replace(key, value)
        path.write_text(text, encoding="utf-8")
        remaining = [line for line in text.splitlines() if "{{" in line and "}}" in line and "COMPETITOR_TABLE" not in line]
        print(f"filled {doc}; unresolved placeholders: {len(remaining)}")


if __name__ == "__main__":
    main()
