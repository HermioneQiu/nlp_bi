import json
import os
import re
from openai import OpenAI
from typing import Optional
from pydantic import ValidationError

from text2dsl import text_to_dsl
from dsl2sql import DSL,compile_sql
from metricstore import MetricStore

if __name__ == "__main__":
    q = "2026年3月各区域线上销售额环比"
    store = MetricStore()
    # 1. 确认意图
    intent = classify_intent(q)
    print(f"意图：{intent}")
    # 2.1 查询定义
    if intent == "definition":
        print(answer_definition(match_metric(q, store), store))
    # 2.2 生成dsl，再进行进一步的操作
    else:
        dsl = text_to_dsl(q, store)
        print("DSL：", json.dumps(dsl.model_dump(), ensure_ascii=False, indent=2))
        sql = compile_sql(dsl, store)
        print("SQL:", sql)