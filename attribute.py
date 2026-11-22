"""维度拆解与贡献度（对应 MetricBot 的指标归因）。

输入是 DSL 带 compare 时产出的 DataFrame：维度、cur_v、base_v、delta。
输出增加 contribution_pct（对总变化的贡献占比）与 rank。
"""
import pandas as pd


def decompose(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    out = df.copy()
    total_delta = out["delta"].sum()
    out["contribution_pct"] = (
        (out["delta"] / total_delta * 100).round(2) if total_delta else 0.0
    )
    out["rank"] = out["delta"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("delta", ascending=False).reset_index(drop=True)


def should_drill(rows: pd.DataFrame, threshold: float = 0.05) -> bool:
    """总变化超过阈值才下钻，避免对噪声做归因。"""
    total_cur = rows["cur_v"].sum()
    total_base = rows["base_v"].sum()
    if total_base == 0:
        return False
    return abs(total_cur - total_base) / total_base >= threshold


def conclusion(df: pd.DataFrame, dim: str, metric_label: str, period: str) -> str:
    total_delta = df["delta"].sum()
    top = df.loc[df["delta"].abs().idxmax()]  # 影响最大的那一项
    if total_delta >= 0:
        return (
            f"{period} {metric_label}环比增长 {total_delta:,.2f}；"
            f"主要拉动是「{top[dim]}」（{top['delta']:+,.2f}，"
            f"贡献 {top['contribution_pct']:+.1f}%）。"
        )
    return (
        f"{period} {metric_label}环比下滑 {abs(total_delta):,.2f}；"
        f"主要拖累是「{top[dim]}」（{top['delta']:+,.2f}，"
        f"贡献 {top['contribution_pct']:+.1f}% 的降幅）。"
    )


if __name__ == "__main__":
    import sqlite3
    from metricstore import MetricStore
    from dsl2sql import DSL, TimeWindow, Filter, compile_sql

    store, conn = MetricStore(), sqlite3.connect("metric.db")
    dsl = DSL(
        metric="sales_amount",
        time=TimeWindow(start="2026-03", end="2026-03"),
        group_by=["region"],
        filters=[Filter(field="channel", value="线上")],
        compare="mom",
    )
    rows = pd.read_sql_query(compile_sql(dsl, store), conn)
    result = decompose(rows, "region")
    print(result.to_string(index=False))
    print()
    print("是否需要下钻：", should_drill(rows))
    print(conclusion(result, "region", "销售额", "2026-03"))