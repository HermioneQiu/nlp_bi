"""阶段 4 总装：LangGraph 编排「口径问答 / 指标取数 / 异动归因」三条分支。

运行：
  export GLM_API_KEY="id.secret"
  python init_db.py
  python graph.py "2026年3月各区域线上销售额环比"
  python graph.py "销售额的口径是什么"
  python graph.py "为什么3月线上销售额下滑了"
"""
import sqlite3
import sys
from typing import Any, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

import text2dsl
from attribution import (conclusion, conclusion_rate, decompose, decompose_rate,
                        should_drill, should_drill_rate)
from dsl2sql import DSL, compile_rate_sql, compile_sql
from metric_store import MetricStore

STORE = MetricStore()
DB_PATH = "metric.db"


class State(TypedDict, total=False):
    question: str
    intent: str
    metric: str
    dsl: dict
    sql: str
    rows: Any
    answer: str
    trace: list


def _log(state: State, node: str, **fields) -> list:
    trace = list(state.get("trace", []))
    changed = ", ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    trace.append(f"[{node}] {changed}" if changed else f"[{node}]")
    return trace


def classify(state: State) -> dict:
    intent = text2dsl.classify_intent(state["question"])
    return {"intent": intent, "trace": _log(state, "classify_intent", intent=intent)}


def route_intent(state: State) -> str:
    return "answer_definition" if state["intent"] == "definition" else "build_dsl"


def answer_definition(state: State) -> dict:
    m = text2dsl.match_metric(state["question"], STORE)
    return {
        "metric": m,
        "answer": text2dsl.answer_definition(m, STORE),
        "trace": _log(state, "answer_definition", metric=m),
    }


def build_dsl(state: State) -> dict:
    dsl = text2dsl.text_to_dsl(state["question"], STORE)
    return {
        "metric": dsl.metric,
        "dsl": dsl.model_dump(),
        "trace": _log(state, "build_dsl", metric=dsl.metric, compare=dsl.compare),
    }


def compile_sql_node(state: State) -> dict:
    dsl = DSL(**state["dsl"])
    metric = STORE.get(dsl.metric) or {}
    if metric.get("type") == "ratio":
        sql = compile_rate_sql(dsl, STORE)     # 比率：取回分子/分母，不先算成率
    else:
        sql = compile_sql(dsl, STORE)
    return {"sql": sql, "trace": _log(state, "compile_sql", type=metric.get("type", "additive"))}


def run_sql(state: State) -> dict:
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = pd.read_sql_query(state["sql"], conn)
    finally:
        conn.close()
    return {
        "rows": rows,
        "trace": _log(state, "run_sql", rows=len(rows)),
    }


def route_after_run(state: State) -> str:
    if state["intent"] == "attribution":
        return "decompose"
    return "summarize"


def decompose_node(state: State) -> dict:
    rows, dsl = state["rows"], DSL(**state["dsl"])
    dims = dsl.group_by or ["region"]
    dim = dims[0]
    label = STORE.get(dsl.metric)["label"]
    period = dsl.time.end

    if STORE.get(dsl.metric).get("type") == "ratio":
        if not should_drill_rate(rows):
            return {
                "answer": "整体转化率相对变动未超过 5%，判定为正常波动，不做下钻。",
                "trace": _log(state, "decompose", drill=False, type="ratio"),
            }
        detail, rate0, rate1 = decompose_rate(rows, dim)
        answer = conclusion_rate(detail, rate0, rate1, dim, label, period)
        return {
            "rows": detail,
            "answer": answer,
            "trace": _log(state, "decompose", drill=True, dim=dim, type="ratio"),
        }

    if not should_drill(rows):
        return {
            "answer": "总变化未超过 5%，判定为正常波动，不做下钻。",
            "trace": _log(state, "decompose", drill=False),
        }
    detail = decompose(rows, dim)
    answer = conclusion(detail, dim, label, period)
    return {
        "rows": detail,
        "answer": answer,
        "trace": _log(state, "decompose", drill=True, dim=dim),
    }


def summarize(state: State) -> dict:
    rows, dsl = state["rows"], DSL(**state["dsl"])
    label = STORE.get(dsl.metric)["label"]

    if STORE.get(dsl.metric).get("type") == "ratio":
        den1, num1 = float(rows["den1"].sum()), float(rows["num1"].sum())
        rate1 = num1 / den1 * 100 if den1 else 0.0
        if dsl.compare:
            den0, num0 = float(rows["den0"].sum()), float(rows["num0"].sum())
            rate0 = num0 / den0 * 100 if den0 else 0.0
            answer = (f"{dsl.time.end} {label}为 {rate1:.2f}%，"
                      f"{dsl.compare.upper()} 基期 {rate0:.2f}%，变动 {rate1 - rate0:+.2f} 个百分点。")
        else:
            answer = f"{dsl.time.end} {label}为 {rate1:.2f}%。"
        return {"answer": answer, "trace": _log(state, "summarize", type="ratio")}

    if dsl.compare:
        cur, base = float(rows["cur_v"].sum()), float(rows["base_v"].sum())
        pct = (cur - base) / base * 100 if base else 0.0
        answer = (
            f"{dsl.time.end} {label}为 {cur:,.2f}，"
            f"{dsl.compare.upper()} 基期 {base:,.2f}，变动 {pct:+.1f}%。"
        )
    else:
        answer = f"{dsl.time.end} {label}合计 {float(rows['v'].sum()):,.2f}。"
    return {"answer": answer, "trace": _log(state, "summarize")}


def build_graph():
    g = StateGraph(State)
    for name, fn in [
        ("classify", classify),
        ("answer_definition", answer_definition),
        ("build_dsl", build_dsl),
        ("compile_sql", compile_sql_node),
        ("run_sql", run_sql),
        ("decompose", decompose_node),
        ("summarize", summarize),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("classify")
    g.add_conditional_edges(
        "classify",
        route_intent,
        {"answer_definition": "answer_definition", "build_dsl": "build_dsl"},
    )
    g.add_edge("answer_definition", END)
    g.add_edge("build_dsl", "compile_sql")
    g.add_edge("compile_sql", "run_sql")
    g.add_conditional_edges(
        "run_sql", route_after_run, {"decompose": "decompose", "summarize": "summarize"}
    )
    g.add_edge("decompose", END)
    g.add_edge("summarize", END)
    return g.compile()


def ask(question: str) -> State:
    result = build_graph().invoke({"question": question, "trace": []})
    print("\n".join(result["trace"]))
    print("\n问：", question)
    print("答：", result.get("answer"))
    if isinstance(result.get("rows"), pd.DataFrame):
        print(result["rows"].to_string(index=False))
    return result


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "2026年3月各区域线上销售额环比"
    ask(q)