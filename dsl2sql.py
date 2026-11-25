"""Text2DSL2SQL 的第 2 步：DSL -> SQL 的确定性编译器（不经过 LLM）。

原则：聚合表达式、表名、时间字段全部来自 metrics.yaml；
LLM 只能在 DSL 里选指标、选维度、给时间窗和过滤条件。
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, field_validator
from metric_store import MetricStore


class TimeWindow(BaseModel):
    start: str  # 形如 2026-03
    end: str


class Filter(BaseModel):
    field: str
    op: Literal["=", "!=", "in"] = "="
    value: object


class DSL(BaseModel):
    metric: str
    time: TimeWindow
    group_by: List[str] = []
    filters: List[Filter] = []
    compare: Optional[Literal["mom", "yoy"]] = None  # 环比 / 同比

    @field_validator("group_by")
    @classmethod
    def _no_duplicate_dim(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("group_by 不能重复")
        return v


def _shift_month(month: str, mode: str) -> str:
    y, m = (int(x) for x in month.split("-"))
    if mode == "yoy":
        return f"{y - 1:04d}-{m:02d}"
    m -= 1
    if m == 0:
        y, m = y - 1, 12
    return f"{y:04d}-{m:02d}"


def _where(metric: dict, dsl: DSL, start: str, end: str) -> str:
    parts = [f"{metric['time_field']} BETWEEN '{start}' AND '{end}'"]
    for f in dsl.filters:
        if f.field not in metric["dimensions"]:
            raise ValueError(f"不允许的过滤字段：{f.field}")
        if f.op == "in":
            vals = ", ".join(repr(str(v)) for v in f.value)
            parts.append(f"{f.field} IN ({vals})")
        else:
            parts.append(f"{f.field} {f.op} {repr(str(f.value))}")
    return " AND ".join(parts)


def compile_sql(dsl: DSL, store: MetricStore) -> str:
    metric = store.get(dsl.metric)
    if metric is None:
        raise ValueError(f"未知指标：{dsl.metric}")
    for d in dsl.group_by:
        if d not in metric["dimensions"]:
            raise ValueError(f"指标 {metric['name']} 不支持维度：{d}")

    expr, table = metric["expr"], metric["table"]
    dims = dsl.group_by
    select_dims = ", ".join(dims)
    group_clause = f" GROUP BY {select_dims}" if dims else ""
    where_cur = _where(metric, dsl, dsl.time.start, dsl.time.end)

    if not dims:
        sql = (
            f"SELECT {expr} AS v FROM {table} WHERE {where_cur}"
        )
        if dsl.compare:
            base_start = _shift_month(dsl.time.start, dsl.compare)
            base_end = _shift_month(dsl.time.end, dsl.compare)
            where_base = _where(metric, dsl, base_start, base_end)
            sql = (
                f"WITH cur AS (SELECT {expr} AS v FROM {table} WHERE {where_cur}),\n"
                f"     base AS (SELECT {expr} AS v FROM {table} WHERE {where_base})\n"
                f"SELECT cur.v AS cur_v, base.v AS base_v, "
                f"ROUND(cur.v - base.v, 2) AS delta FROM cur, base"
            )
        return sql

    sql = (
        f"SELECT {select_dims}, {expr} AS v FROM {table} WHERE {where_cur}{group_clause}"
    )
    if not dsl.compare:
        return sql

    base_start = _shift_month(dsl.time.start, dsl.compare)
    base_end = _shift_month(dsl.time.end, dsl.compare)
    where_base = _where(metric, dsl, base_start, base_end)
    on = " AND ".join(f"c.{d} = b.{d}" for d in dims)
    return (
        f"WITH cur AS (SELECT {select_dims}, {expr} AS v FROM {table} "
        f"WHERE {where_cur}{group_clause}),\n"
        f"     base AS (SELECT {select_dims}, {expr} AS v FROM {table} "
        f"WHERE {where_base}{group_clause})\n"
        f"SELECT {', '.join(f'c.{d}' for d in dims)}, c.v AS cur_v, "
        f"COALESCE(b.v, 0) AS base_v, ROUND(c.v - COALESCE(b.v, 0), 2) AS delta\n"
        f"FROM cur c LEFT JOIN base b ON {on}"
    )


def compile_rate_sql(dsl: DSL, store: MetricStore) -> str:
    """比率指标专用：不出 SUM 后的比值，而是把分子/分母各期原样取回来。

    原因：比率要按维度还原成 w×r，必须先拿到每个维度的 num 与 den；
    如果在 SQL 里先算成 rate，权重信息就丢了，归因无法做结构/效率分解。
    输出列：无 compare → dims + num_v + den_v；有 compare → dims + num1/den1/num0/den0。
    """
    metric = store.get(dsl.metric)
    if metric is None:
        raise ValueError(f"未知指标：{dsl.metric}")
    if metric.get("type") != "ratio":
        raise ValueError(f"{dsl.metric} 不是比率指标，请用 compile_sql")
    dims = dsl.group_by
    for d in dims:
        if d not in metric["dimensions"]:
            raise ValueError(f"指标 {metric['name']} 不支持维度：{d}")

    num, den, table = metric["numerator"], metric["denominator"], metric["table"]
    cols = ", ".join(dims)
    prefix = f"{cols}, " if dims else ""
    group = f" GROUP BY {cols}" if dims else ""
    where_cur = _where(metric, dsl, dsl.time.start, dsl.time.end)

    if not dsl.compare:
        return (f"SELECT {prefix}{num} AS num_v, {den} AS den_v "
                f"FROM {table} WHERE {where_cur}{group}")

    base_start = _shift_month(dsl.time.start, dsl.compare)
    base_end = _shift_month(dsl.time.end, dsl.compare)
    where_base = _where(metric, dsl, base_start, base_end)
    select_cur = f"SELECT {prefix}{num} AS num, {den} AS den FROM {table} WHERE {where_cur}{group}"
    select_base = f"SELECT {prefix}{num} AS num, {den} AS den FROM {table} WHERE {where_base}{group}"
    if dims:
        on = " AND ".join(f"c.{d} = b.{d}" for d in dims)
        head = f"{', '.join(f'c.{d}' for d in dims)}, "
        join = f"cur c LEFT JOIN base b ON {on}"
    else:
        head, join = "", "cur c LEFT JOIN base b"
    return (
        f"WITH cur AS ({select_cur}),\n"
        f"     base AS ({select_base})\n"
        f"SELECT {head}c.num AS num1, c.den AS den1, "
        f"COALESCE(b.num, 0) AS num0, COALESCE(b.den, 0) AS den0\n"
        f"FROM {join}"
    )


if __name__ == "__main__":
    store = MetricStore()
    demo = DSL(
        metric="sales_amount",
        time=TimeWindow(start="2026-03", end="2026-03"),
        group_by=["region"],
        filters=[Filter(field="channel", op="=", value="线上")],
        compare="mom",
    )
    print(compile_sql(demo, store))