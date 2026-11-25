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


def should_drill_rate(df: pd.DataFrame, threshold: float = 0.05) -> bool:
    """比率指标的“该不该下钻”：按整体率算相对变化，不看单维度波动。"""
    den0, den1 = float(df["den0"].sum()), float(df["den1"].sum())
    if den0 == 0 or den1 == 0:
        return False
    r0 = float(df["num0"].sum()) / den0
    if r0 == 0:
        return False
    r1 = float(df["num1"].sum()) / den1
    return abs(r1 - r0) / r0 >= threshold


def decompose_rate(df: pd.DataFrame, dim: str):
    """比率指标归因：rate = Σ(w_i × r_i)，拆成结构效应 + 效率效应。

    - 结构效应：流量占比变化（w 变了），用基期转化率计价
    - 效率效应：各维度自身转化率变化（r 变了），用基期占比计价
    - 交叉项：Δw × Δr，用 Shapley 对称平分给两项，保证两项之和 = 总变化
    返回 (明细表, 基期整体率, 本期整体率)，率都是 0~1 的小数。
    """
    out = df.copy()
    den0, den1 = float(out["den0"].sum()), float(out["den1"].sum())
    if den0 == 0 or den1 == 0:
        raise ValueError("基期或本期访问量为 0，无法计算转化率")
    rate0 = float(out["num0"].sum()) / den0
    rate1 = float(out["num1"].sum()) / den1

    recs = []
    for _, r in out.iterrows():
        w0 = float(r["den0"]) / den0
        w1 = float(r["den1"]) / den1
        rr0 = float(r["num0"]) / float(r["den0"]) if r["den0"] else 0.0
        rr1 = float(r["num1"]) / float(r["den1"]) if r["den1"] else 0.0
        dw, dr = w1 - w0, rr1 - rr0
        struct, eff = rr0 * dw, w0 * dr
        cross = dw * dr
        recs.append({
            dim: r[dim],
            "w0": round(w0 * 100, 2), "w1": round(w1 * 100, 2),
            "rate0": round(rr0 * 100, 2), "rate1": round(rr1 * 100, 2),
            "struct_pp": round((struct + cross / 2) * 100, 3),
            "eff_pp": round((eff + cross / 2) * 100, 3),
            "total_pp": round((struct + eff + cross) * 100, 3),
        })
    return pd.DataFrame(recs).sort_values("total_pp").reset_index(drop=True), rate0, rate1


def conclusion_rate(
    detail: pd.DataFrame, rate0: float, rate1: float, dim: str, metric_label: str, period: str
) -> str:
    """比率指标的结论：单位是“个百分点”，并且必须分开报结构与效率。"""
    total = (rate1 - rate0) * 100
    struct = float(detail["struct_pp"].sum())
    eff = float(detail["eff_pp"].sum())
    worst_struct = detail.loc[detail["struct_pp"].abs().idxmax()]
    worst_eff = detail.loc[detail["eff_pp"].abs().idxmax()]
    trend = "下滑" if total < 0 else "上升"
    return (
        f"{period} {metric_label}从 {rate0 * 100:.2f}% {trend}到 {rate1 * 100:.2f}%，"
        f"变动 {total:+.2f} 个百分点；结构效应 {struct:+.2f}pp、效率效应 {eff:+.2f}pp。"
        f"结构上最大影响是「{worst_struct[dim]}」（{worst_struct['struct_pp']:+.2f}pp，"
        f"流量占比 {worst_struct['w0']:.0f}%→{worst_struct['w1']:.0f}%），"
        f"效率上最大影响是「{worst_eff[dim]}」（{worst_eff['eff_pp']:+.2f}pp，"
        f"转化率 {worst_eff['rate0']:.2f}%→{worst_eff['rate1']:.2f}%）。"
    )


if __name__ == "__main__":
    import sqlite3
    from metric_store import MetricStore
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