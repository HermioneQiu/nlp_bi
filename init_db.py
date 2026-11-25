"""MetricBot 复刻用的事实表：metric.db。

sales   一行 = 一笔订单（可加指标：销售额 / 订单量）
traffic 一行 = 一个「月份 × 区域 × 渠道」的流量与成交（比率指标：转化率）

month 用 TEXT 存 YYYY-MM，方便同比/环比；
2026-03 华东线上被刻意压低，用来演示归因；
流量从华东迁往华南、华北，用来演示「结构效应 vs 效率效应」。
"""
import sqlite3
from itertools import product

REGIONS, CHANNELS = ["华东", "华南", "华北"], ["线上", "线下"]
BASE = {("华东", "线上"): 128000, ("华东", "线下"): 96000,
        ("华南", "线上"): 101000, ("华南", "线下"): 87000,
        ("华北", "线上"): 92000, ("华北", "线下"): 78000}
M25, M26 = [f"2025-0{i}" for i in range(1, 5)], [f"2026-0{i}" for i in range(1, 5)]
GROWTH = {"2026-01": 1.10, "2026-02": 1.15, "2026-03": 1.20, "2026-04": 1.22}
SPECIAL = {("2026-03", "华东", "线上"): 0.60}

DDL = """DROP TABLE IF EXISTS sales;
CREATE TABLE sales (id INTEGER PRIMARY KEY AUTOINCREMENT, month TEXT,
                    region TEXT, channel TEXT, amount REAL);
CREATE INDEX idx_sales_month ON sales(month);
DROP TABLE IF EXISTS traffic;
CREATE TABLE traffic (id INTEGER PRIMARY KEY AUTOINCREMENT, month TEXT, region TEXT,
                      channel TEXT, uv INTEGER, orders INTEGER);
CREATE INDEX idx_traffic_month ON traffic(month);"""

# (uv, orders)：2026-02 总 40,000 访问 / 1,700 单（4.25%）→ 2026-03 总 40,000 / 1,570（3.93%）
TRAFFIC = {
    ("2026-02", "华东", "线上"): (12000, 620), ("2026-02", "华东", "线下"): (8000, 380),
    ("2026-02", "华南", "线上"): (6000, 240), ("2026-02", "华南", "线下"): (4000, 160),
    ("2026-02", "华北", "线上"): (6000, 180), ("2026-02", "华北", "线下"): (4000, 120),
    ("2026-03", "华东", "线上"): (5000, 210), ("2026-03", "华东", "线下"): (3000, 110),
    ("2026-03", "华南", "线上"): (12000, 620), ("2026-03", "华南", "线下"): (6000, 280),
    ("2026-03", "华北", "线上"): (9000, 230), ("2026-03", "华北", "线下"): (5000, 120),
}


def rows_for(months, growth):
    return [(m, r, c, round(BASE[(r, c)] * growth.get(m, 1.0) * SPECIAL.get((m, r, c), 1.0), 2))
            for m, (r, c) in product(months, BASE)]


def traffic_rows():
    return [(m, r, c, uv, od) for (m, r, c), (uv, od) in TRAFFIC.items()]


if __name__ == "__main__":
    conn = sqlite3.connect("metric.db")
    conn.executescript(DDL)
    conn.executemany("INSERT INTO sales(month,region,channel,amount) VALUES(?,?,?,?)",
                     rows_for(M25, {}) + rows_for(M26, GROWTH))
    conn.executemany("INSERT INTO traffic(month,region,channel,uv,orders) VALUES(?,?,?,?,?)",
                     traffic_rows())
    conn.commit()
    print("metric.db 已生成：", conn.execute("SELECT COUNT(*), ROUND(SUM(amount),2) FROM sales").fetchone())
    print("traffic 已生成：", conn.execute("SELECT COUNT(*), SUM(uv), SUM(orders) FROM traffic").fetchone())
    conn.close()