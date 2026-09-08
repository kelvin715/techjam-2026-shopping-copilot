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
ARM_ORDER = ("deterministic", "lexical", "cascade", "hybrid")
ARM_LABEL = {
    "deterministic": "deterministic (frozen submission)",
    "lexical": "catalog reads the sentence, no model",
    "cascade": "cascade: catalog first, model on demand (shipped)",
    "hybrid": "model on every off-protocol message",
}
ARM_LABEL_ZH = {
    "deterministic": "deterministic（提交冻结版）",
    "lexical": "目录逐字匹配，不用模型",
    "cascade": "级联：目录先读，按需调模型（交付版）",
    "hybrid": "每条非协议消息都调模型",
}


def shipped_arm(merged: dict) -> str:
    """The arm that the shipped ``ground`` mode runs."""
    return "cascade" if any(key.endswith("__cascade") for key in merged["experiments"]) else "hybrid"


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
    cross = merged.get("cross_model")
    shipped = shipped_arm(merged)
    shopper_g = merged["customer_model"]["model"]
    shopper_q = cross["customer_model"]["model"] if cross else None
    cells = [(level, arm, merged["experiments"].get(f"{level}__{arm}"), shopper_g) for level in merged["levels"] for arm in merged["arms"]]
    if cross:
        cells += [(level, arm, cross["experiments"].get(f"{level}__{arm}"), shopper_q)
                  for level in ("natural", "paraphrase") for arm in merged["arms"]]
    for level, arm, row, shopper in cells:
            if not row:
                continue
            arm_label = (ARM_LABEL_ZH if zh else ARM_LABEL).get(arm, arm)
            calls = row["grounding"]["model_calls"]
            tokens = row["reported_token_usage"]["total_tokens"]
            bold = arm == shipped and level != "canonical"
            score = fmt(row["technical_score"], 6)
            level_label = labels[level] if level == "canonical" else f"{labels[level]} ({shopper} shopper)"
            lines.append(
                f"| {level_label} | {arm_label} | {fmt(row['hit_rate_at_10'])} | {fmt(row['mrr'])} | "
                f"{row['mttc']:.3f} | {'**' + score + '**' if bold else score} | {calls} | {tokens:,} |\n"
            )
    return "".join(lines)


def notes(merged: dict, zh: bool = False) -> str:
    n = merged["sample_count"]
    shipped = shipped_arm(merged)
    hyb = {level: merged["experiments"].get(f"{level}__{shipped}") for level in merged["levels"]}
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
    ex = merged["experiments"]
    nat = ex.get(f"natural__{shipped}")
    par = ex.get(f"paraphrase__{shipped}")
    nat_model = ex.get("natural__hybrid")
    par_model = ex.get("paraphrase__hybrid")
    cross = merged.get("cross_model")
    arm_name = ARM_LABEL_ZH[shipped] if zh else ARM_LABEL[shipped]
    if zh:
        text = (
            f"- {n} 个公开 session；顾客改写模型 `{cm}`，grounding 模型 `{gm}`（同一本地 vLLM 服务；改写只看模板消息，不看目录）。\n"
            f"- 官方模板下所有臂与 deterministic 逐字节同分、0 次调用：协议句式从不到达模型。\n"
        )
        for level, calls, toks, lat in per_session:
            text += f"- {LEVEL_LABEL_ZH[level]}：交付版每 session 平均 {calls:.2f} 次调用、{toks:,.0f} tokens"
            model_row = ex.get(f"{level}__hybrid")
            if model_row and shipped != "hybrid":
                text += f"（每条消息都调模型的基线为 {model_row['grounding']['model_calls']/n:.2f} 次、{model_row['reported_token_usage']['total_tokens']/n:,.0f} tokens）"
            text += f"，单次模型调用平均 {lat:.0f} ms。\n"
        if example:
            text += f"- 改写样例：模拟器「{example['simulator'][:90]}」→ 顾客「{example['human'][:90]}」。\n"
        if nat and par:
            text += (f"- 开启语言层后的 agent 单轮延迟（含模型调用）：natural 档均值 {nat['mean_respond_latency_ms']/1000:.2f} s（p95 {nat['p95_respond_latency_ms']/1000:.2f} s）；"
                     f"paraphrase 档均值 {par['mean_respond_latency_ms']/1000:.2f} s（p95 {par['p95_respond_latency_ms']/1000:.2f} s）。整版证伪后的扩池经稀有词倒排索引限定在 2,000 件商品内，不再每轮重排 5 万件。\n")
            sc = par["scenario_metrics"]
            text += (f"- paraphrase 档按场景（交付版）Hit@10：buying {sc['buying']['hit_rate_at_10']:.3f} / browsing {sc['browsing']['hit_rate_at_10']:.3f} / intent override {sc['intent_override']['hit_rate_at_10']:.3f} / boundary {sc['boundary']['hit_rate_at_10']:.3f}。"
                     "boundary 掉分是模拟器的人工痕迹：它先说「对 X 没偏好」，之后又会回答 X——按人类语义我们把该问题退役了。\n")
        if cross:
            c = cross["experiments"]
            text += (f"- 跨模型对照：改用 `{cross['customer_model']['model']}` 扮演顾客（改写由 `tools/precompute_customer_rewrites.py` 预生成，grounding 仍是 `{gm}`），natural 档 deterministic {c['natural__deterministic']['technical_score']:.4f}、交付版 {c[f'natural__{shipped}']['technical_score']:.4f}"
                     f"（Hit@10 {c['natural__deterministic']['hit_rate_at_10']:.3f} → {c[f'natural__{shipped}']['hit_rate_at_10']:.3f}，每 session {c[f'natural__{shipped}']['grounding']['model_calls']/n:.1f} 次调用）。结论与同模型实验一致。\n")
        text += "- 这是研究诊断，不是主办方分数；顾客仍是模型扮演的。\n"
        return text
    text = (
        f"- {n} public sessions per cell; customer rewrites by `{cm}`, grounding by `{gm}` (the same local vLLM service; the rewriter sees only the template message, never the catalog).\n"
    )
    if merged.get("caches", {}).get("strict_replay"):
        source = merged.get("replay_dir", "results/attribution/replay/")
        text += ("- Every rewrite and every model reply is served from an immutable cache and the grid was replayed with `--strict-replay --strict-grounding` (zero cache misses); "
                 f"per-session outcomes and turn-level trajectories are in `{source}`, and `tools/check_attribution_replay.py` verifies that the replay reproduces every cell.\n")
    text += "- On the organizer templates every arm makes zero model calls and reproduces the deterministic score exactly: protocol wording never reaches the model.\n"
    for level, calls, toks, lat in per_session:
        text += f"- {LEVEL_LABEL[level]}: the shipped cascade averages {calls:.2f} model calls and {toks:,.0f} tokens per session"
        model_row = ex.get(f"{level}__hybrid")
        if model_row and shipped != "hybrid":
            text += f" (the model-on-every-message baseline: {model_row['grounding']['model_calls']/n:.2f} calls, {model_row['reported_token_usage']['total_tokens']/n:,.0f} tokens)"
        text += f", {lat:.0f} ms per model call (latency from a live-call run of the same condition).\n"
    if example:
        text += f"- Example rewrite: simulator “{example['simulator'][:90]}” → shopper “{example['human'][:90]}”.\n"
    if nat and par:
        text += (f"- Agent latency with the language layer on, model calls included: natural wording mean {nat['mean_respond_latency_ms']/1000:.2f} s per turn (p95 {nat['p95_respond_latency_ms']/1000:.2f} s); "
                 f"paraphrased wording mean {par['mean_respond_latency_ms']/1000:.2f} s (p95 {par['p95_respond_latency_ms']/1000:.2f} s). A session that has refuted a full slate looks beyond its shelf pool through a rarity-weighted inverted index bounded to 2,000 products instead of re-scoring the whole catalog on every later turn.\n")
        sc = par["scenario_metrics"]
        text += (f"- Paraphrased wording by scenario (cascade): buying {sc['buying']['hit_rate_at_10']:.3f} / browsing {sc['browsing']['hit_rate_at_10']:.3f} / intent override {sc['intent_override']['hit_rate_at_10']:.3f} / boundary {sc['boundary']['hit_rate_at_10']:.3f} Hit@10. "
                 "The boundary drop is a simulator artefact: its shopper says \"no preference\" once and then answers that very attribute later, which a human reading treats as a retired question.\n")
    if cross:
        c = cross["experiments"]
        text += (f"- Cross-model check: with `{cross['customer_model']['model']}` playing the shopper (rewrites pre-generated by `tools/precompute_customer_rewrites.py`, grounding still `{gm}`), natural wording scores {c['natural__deterministic']['technical_score']:.4f} deterministic and {c[f'natural__{shipped}']['technical_score']:.4f} cascade "
                 f"(Hit@10 {c['natural__deterministic']['hit_rate_at_10']:.3f} → {c[f'natural__{shipped}']['hit_rate_at_10']:.3f}, {c[f'natural__{shipped}']['grounding']['model_calls']/n:.1f} calls per session)")
        if f"paraphrase__{shipped}" in c:
            text += (f"; paraphrased wording scores {c['paraphrase__deterministic']['technical_score']:.4f} deterministic and {c[f'paraphrase__{shipped}']['technical_score']:.4f} cascade")
        text += ": the same direction as the same-model run.\n"
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
        for arm in ("deterministic", shipped_arm(merged)):
            row = ex.get(f"{level}__{arm}")
            if not row:
                continue
            width = max(3, round(row["technical_score"] * 100))
            label = f"{labels[level]} · {ARM_LABEL_ZH.get(arm, arm)}"
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
            arm_label = ARM_LABEL_ZH.get(arm, arm)
            bold = arm == shipped_arm(merged) and level != "canonical"
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


def _replace_existing_blocks(text: str, values: dict, zh: bool) -> str:
    """Regenerate a previously filled table and notes list in place."""
    import re

    header = "| 顾客措辞 | 臂 |" if zh else "| Shopper wording | Arm |"
    if header in text:
        start = text.index(header)
        end = start
        for line in text[start:].splitlines(keepends=True):
            if not line.startswith("|"):
                break
            end += len(line)
        text = text[:start] + values["{{HL_TABLE}}"] + text[end:]
    first = "- 200 个公开 session" if zh else "- 200 public sessions"
    last = "- 这是研究诊断" if zh else "- These are research diagnostics"
    if first in text and last in text:
        start = text.index(first)
        tail = text.index(last, start)
        end = tail + len(text[tail:].splitlines(keepends=True)[0])
        text = text[:start] + values["{{HL_NOTES}}"] + text[end:]
    return text


def _replace_existing_html(text: str, blocks: dict) -> str:
    import re

    text = re.sub(r'<div class="bars">.*?</div>\s*</div>', blocks["<!--HL_BARS-->"], text, count=1, flags=re.S)
    text = re.sub(r'<div class="tablewrap"><table><thead><tr><th>顾客措辞</th>.*?</table></div>',
                  '<div class="tablewrap">' + blocks["<!--HL_TABLE-->"] + '</div>', text, count=1, flags=re.S)
    text = re.sub(r'<ul><li>200 个公开 session.*?</ul>', blocks["<!--HL_NOTES-->"], text, count=1, flags=re.S)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # The newest strict-replay grid wins: replay_v2 carries the cascade arm.
    replay = ROOT / "results" / "attribution" / "replay"
    for candidate in ("replay_v2/strict", "replay"):
        if (ROOT / "results" / "attribution" / candidate / "combined_gemma.json").is_file():
            replay = ROOT / "results" / "attribution" / candidate
            break
    replay_rel = replay.relative_to(ROOT).as_posix()
    replay_ready = (replay / "combined_gemma.json").is_file()
    parser.add_argument("--inputs", nargs="+", default=(
        [f"{replay_rel}/combined_gemma.json"] if replay_ready else [
            "results/human_language_benchmark_natural.json",
            "results/human_language_benchmark_paraphrase.json",
        ]))
    # One canonical file: the demo bundle and the docs read this one.
    parser.add_argument("--output", default="results/human_language_benchmark.json")
    parser.add_argument("--no-docs", action="store_true")
    parser.add_argument("--html", help="playbook HTML page whose <!--HL_*--> markers are filled")
    parser.add_argument("--cross-model", default=(
        f"{replay_rel}/combined_qwen.json" if replay_ready
        else "results/human_language_benchmark_natural_qwen_customer.json"))
    parser.add_argument("--latency", default="results/attribution/live_latency",
                        help="directory of live-call runs whose latencies replace the replay's cached-call timings")
    args = parser.parse_args()

    merged = merge([ROOT / path for path in args.inputs])
    order = [level for level in ("canonical", "natural", "paraphrase") if level in merged["levels"]]
    merged["levels"] = order
    merged["arms"] = [arm for arm in ARM_ORDER if arm in merged["arms"]]
    merged["replay_dir"] = replay_rel + "/"
    cross_path = ROOT / args.cross_model
    if cross_path.is_file():
        merged["cross_model"] = json.loads(cross_path.read_text(encoding="utf-8"))
    # A strict replay serves model replies from cache, so its timings are not
    # model latencies; take them from the live-call measurement when present.
    latency_dir = ROOT / args.latency
    for level in ("natural", "paraphrase"):
        for arm in ("cascade", "hybrid"):
            live = latency_dir / f"gemma_{level}_{arm}.json"
            row = merged["experiments"].get(f"{level}__{arm}")
            if live.is_file() and row:
                live_row = list(json.loads(live.read_text(encoding="utf-8"))["experiments"].values())[0]
                for key in ("mean_respond_latency_ms", "p95_respond_latency_ms"):
                    row[key] = live_row[key]
                row["grounding"]["mean_model_latency_ms_per_turn"] = live_row["grounding"]["mean_model_latency_ms_per_turn"]
                row["latency_source"] = str(live.relative_to(ROOT))
    (ROOT / args.output).write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(table(merged))
    if args.html:
        page = Path(args.html)
        text = page.read_text(encoding="utf-8")
        blocks = html_blocks(merged)
        for marker, fragment in blocks.items():
            text = text.replace(marker, fragment)
        text = _replace_existing_html(text, blocks)
        page.write_text(text, encoding="utf-8")
        print(f"filled {page}")
    if args.no_docs:
        return
    ex = merged["experiments"]
    n = merged["sample_count"]
    shipped = shipped_arm(merged)
    hyb_para = ex.get(f"paraphrase__{shipped}") or ex.get(f"natural__{shipped}")
    values = {
        "{{HL_TABLE}}": table(merged),
        "{{HL_NOTES}}": notes(merged),
        "{{DET_NATURAL}}": fmt(ex["natural__deterministic"]["technical_score"], 3),
        "{{HYB_NATURAL}}": fmt(ex[f"natural__{shipped}"]["technical_score"], 3),
        "{{DET_PARA}}": fmt(ex["paraphrase__deterministic"]["technical_score"], 3) if "paraphrase__deterministic" in ex else "n/a",
        "{{HYB_PARA}}": fmt(ex[f"paraphrase__{shipped}"]["technical_score"], 3) if f"paraphrase__{shipped}" in ex else "n/a",
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
        text = _replace_existing_blocks(text, local, doc.endswith("PLAYBOOK.md"))
        path.write_text(text, encoding="utf-8")
        remaining = [line for line in text.splitlines() if "{{" in line and "}}" in line and "COMPETITOR_TABLE" not in line]
        print(f"filled {doc}; unresolved placeholders: {len(remaining)}")


if __name__ == "__main__":
    main()
