"""Text2DSL2SQL 的第 2 步：DSL -> SQL 的确定性编译器（不经过 LLM）。

原则：聚合表达式、表名、时间字段全部来自 metrics.yaml；
LLM 只能在 DSL 里选指标、选维度、给时间窗和过滤条件。
"""
from typing import Any, List, Optional, Literal
from pydantic import BaseModel, field_validator
from metricstore import MetricStore

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
        if len(set[Any](v)) != len(v):
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

# 生成除了group by之外的 where 条件：时间 & 其他过滤条件
def _where(metric: dict, dsl: DSL, start: str, end: str) -> str:
    # 1. 获取时间筛选代码
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
    # 1. 根dsl中的指标名称，判断指标是否在store
    # ？？？调用store的get函数，这个应该改成用向量匹配
    metric = store.get(dsl.metric)
    if metric is None:
        raise ValueError(f"未知指标：{dsl.metric}")
    # 2. 判断维度是否在store
    #？？？为啥不校验filter、time
    for d in dsl.group_by:
        if d not in metric["dimensions"]:
            raise ValueError(f"指标 {metric['name']} 不支持维度：{d}")

    # 1. 获取聚合公式和表名
    expr, table = metric["expr"], metric["table"]
    dims = dsl.group_by
    select_dims = ", ".join(dims)
    group_clause = f" GROUP BY {select_dims}" if dims else ""
    where_cur = _where(metric, dsl, dsl.time.start, dsl.time.end)

    # 1 没有group
    if not dims:
        sql = (
            f"SELECT {expr} AS v FROM {table} WHERE {where_cur}"
        )
        # 2. 没有group，需要环比
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

    # 3, 有group，不需要环比
    sql = (
        f"SELECT {select_dims}, {expr} AS v FROM {table} WHERE {where_cur}{group_clause}"
    )
    if not dsl.compare:
        return sql

    # 4, 有group，需要环比
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

# ========== 测试主函数 ==========
if __name__ == "__main__":
    store = MetricStore()
    # 这里的DSL是直接生成的，为了测试DSL 2 SQL的链路；和query、LLM生成DSL没关系
    demo = DSL(
        metric="sales_amount",
        time=TimeWindow(start="2026-03", end="2026-03"),
        group_by=["region"],
        filters=[Filter(field="channel", op="=", value="线上")],
        compare="mom",
    )
    print(compile_sql(demo, store))