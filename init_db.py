"""MetricBot 复刻用的事实表：metric.db（sales）。一行 = 一笔订单。

month 用 TEXT 存 YYYY-MM，方便同比/环比；
2026-03 华东线上被刻意压低，用来演示归因。
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
CREATE INDEX idx_sales_month ON sales(month);"""

def rows_for(months, growth):
    return [(m, r, c, round(BASE[(r, c)] * growth.get(m, 1.0) * SPECIAL.get((m, r, c), 1.0), 2))
            for m, (r, c) in product(months, BASE)]

if __name__ == "__main__":
    conn = sqlite3.connect("metric.db")
    conn.executescript(DDL)
    conn.executemany("INSERT INTO sales(month,region,channel,amount) VALUES(?,?,?,?)",
                     rows_for(M25, {}) + rows_for(M26, GROWTH))
    conn.commit()
    print("metric.db 已生成：", conn.execute("SELECT COUNT(*), ROUND(SUM(amount),2) FROM sales").fetchone())
    print("metric.db 已生成：", conn.execute("SELECT * FROM sales limit 1").fetchone())

    conn.close()
