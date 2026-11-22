"""Text2DSL：把自然语言问题翻译成 DSL（唯一调用 LLM 的环节）。

流程：问句 → 向量召回 TopK 指标 → 把召回指标的完整档案写进 system prompt
      → 模型产出 DSL → dsl2sql 编译成 SQL。
LLM 只被允许做两件事：判断问题类型、产出 DSL JSON；
聚合表达式、表名、时间字段一律由 metrics.yaml 决定，LLM 碰不到。
"""
import json
import os
import re
import sys
from typing import Optional

from openai import OpenAI
from pydantic import ValidationError

from dsl2sql import DSL, compile_sql
from metricstore import MetricIndex, MetricStore

BASE = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = os.getenv("GLM_MODEL", "glm-4-flash")
KEY = os.getenv("GLM_API_KEY","2d875673d4834b3a9868ae3c0d77334a.kboCOVWGKHb99Xw0")
_client: Optional[OpenAI] = None


def chat(messages: list, temperature: float = 0.0) -> str:
    global _client
    if _client is None:
        _client = OpenAI(api_key=KEY, base_url=BASE)
    resp = _client.chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content or ""


INTENTS = ("definition", "query", "attribution")


def classify_intent(question: str) -> str:
    prompt = f"""把用户问题归到下面三类之一，只输出类别英文单词，不要解释：
definition：问指标口径/怎么算/定义
query：要一个数或一张表
attribution：问为什么涨/为什么跌/原因/归因
用户问题：{question}"""
    try:
        label = chat([{"role": "user", "content": prompt}]).strip().lower()
    except Exception:
        label = ""
    for k in INTENTS:
        if k in label:
            return k
    # 兜底规则，保证没有 API 时链路也能判断
    if any(w in question for w in ("为什么", "原因", "下滑", "下降", "涨", "跌")):
        return "attribution"
    if any(w in question for w in ("口径", "怎么算", "定义", "含义")):
        return "definition"
    return "query"


def typical_questions(m: dict) -> str:
    dims = m["dimensions"]
    return "；".join(
        [
            f"{m['label']}是多少",
            f"按 {dims[0]} 拆分的{m['label']}" if dims else f"{m['label']}趋势",
            f"上个月{m['label']}环比",
        ]
    )


def profile_block(metrics: list) -> str:
    """召回指标的完整档案，直接嵌进 system prompt。"""
    blocks = []
    for i, m in enumerate(metrics, 1):
        blocks.append(
            f"""【指标 {i}】
名称：{m['label']}（{m['name']}）
别名（用户常这么说）：{'、'.join(m.get('aliases', []))}
口径定义：{m['description']}
计算式：{m['expr']}
事实表：{m['table']}　时间字段：{m['time_field']}（格式 YYYY-MM，月度粒度）
可用维度：{'、'.join(m['dimensions'])}
单位：{m['unit']}
典型问法：{typical_questions(m)}
口径注意：聚合只能用上面的计算式；不要自造字段、不要跨表 JOIN、不要用日粒度。"""
        )
    return "\n\n".join(blocks)


SYSTEM_TMPL = """你是指标查询 DSL 生成器。只能输出一个 JSON 对象，不要 Markdown、不要解释。

本次只允许使用下面这批指标（向量召回结果，按相关度排序）：
{profiles}

DSL 字段：
- metric: string，必须取上面某个指标的 name
- time: {{"start": "YYYY-MM", "end": "YYYY-MM"}}，闭区间
- group_by: string[]，只能取该指标的可用维度
- filters: [{{"field": 维度名, "op": "="|"in", "value": 值}}]
- compare: "mom" | "yoy" | null（问“为什么/下滑/原因”时必须给）

今天的日期是 {today}。用户说“上个月”取上个自然月，“3月”默认今年 3 月。
如果上面没有任何指标能回答这个问题，输出 {{"metric": null, "reason": "原因"}}，不要硬选。
"""


def build_system_prompt(candidates: list, today: str = "2026-04") -> str:
    return SYSTEM_TMPL.format(profiles=profile_block(candidates), today=today)


def recall_candidates(question: str, store: MetricStore, top_k: int = 2) -> list:
    idx = MetricIndex(store).build()
    scored = idx.recall(question, top_k=top_k)
    if not scored:
        return []
    return [m for m in (store.get(name) for name, _ in scored) if m]


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?\s*|```$", "", text, flags=re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("模型没有返回 JSON")
    return json.loads(text[start : end + 1])


def text_to_dsl(
    question: str,
    store: MetricStore,
    today: str = "2026-04",
    top_k: int = 2,
    retries: int = 1,
) -> DSL:
    """召回 → 全量档案 prompt → DSL，失败时把报错回灌给模型重试一次。"""
    candidates = recall_candidates(question, store, top_k=top_k)
    if not candidates:
        raise RuntimeError("没有指标越过相似度阈值，建议换个说法或补充指标定义")
    allowed = {m["name"] for m in candidates}
    messages = [
        {"role": "system", "content": build_system_prompt(candidates, today)},
        {"role": "user", "content": question},
    ]
    last_err = None
    for _ in range(retries + 1):
        raw = chat(messages)
        try:
            payload = _extract_json(raw)
            if payload.get("metric") is None:
                raise ValueError(f"判定无法回答：{payload.get('reason', '')}")
            dsl = DSL(**payload)
            if dsl.metric not in allowed:
                raise ValueError(f"指标不在召回集合内：{dsl.metric}")
            return dsl
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last_err = e
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"上一条不合法：{e}。请只输出修正后的 JSON。"},
            ]
    raise RuntimeError(f"DSL 生成失败：{last_err}")


def answer_definition(metric_name: str, store: MetricStore) -> str:
    m = store.get(metric_name) or store.metrics[0]
    return (
        f'{m["label"]}（{m["name"]}）口径：{m["description"]}'
        f'计算式 {m["expr"]}，事实表 {m["table"]}，时间字段 {m["time_field"]}，'
        f'可用维度 {",".join(m["dimensions"])}，单位 {m["unit"]}。'
    )


def match_metric(question: str, store: MetricStore) -> str:
    """口径问答用：先别名精确命中，命中不了再召回 Top1。"""
    hit = MetricIndex(store).build().lexical_hits(question)
    if hit:
        return hit[0]
    cands = recall_candidates(question, store, top_k=1)
    if cands:
        return cands[0]["name"]
    return store.metrics[0]["name"]


def main() -> None:
    """命令行自测：意图 -> 召回 -> DSL -> SQL，四步都打印出来。"""
    q = " ".join(sys.argv[1:]) or "2026年3月各区域线上销售额环比"
    store = MetricStore()
    intent = classify_intent(q)
    print(f"意图：{intent}")
    if intent == "definition":
        print(answer_definition(match_metric(q, store), store))
        return
    cands = recall_candidates(q, store)
    labels = "、".join(f"{m['label']}({m['name']})" for m in cands) or "无"
    print(f"召回指标：{labels}")
    print("\n===== 系统提示词（截前 800 字）=====")
    print(build_system_prompt(cands)[:800])
    dsl = text_to_dsl(q, store)
    print("\n===== DSL =====")
    print(json.dumps(dsl.model_dump(), ensure_ascii=False, indent=2))
    print("\n===== SQL =====")
    print(compile_sql(dsl, store))


if __name__ == "__main__":
    main()