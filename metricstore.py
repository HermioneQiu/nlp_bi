"""指标语义层 + 指标向量召回。

上半部分：加载 metrics.yaml，按 name/别名取指标，压缩成给 LLM 的提示词。
下半部分：把每个指标拼成「档案文本」并向量化，供问句做相似度召回。

缓存键是档案文本的哈希：改了 metrics.yaml 会自动重建，不会读到旧向量。
"""
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import yaml
from openai import OpenAI

BASE = "https://open.bigmodel.cn/api/paas/v4/"
EMBED_MODEL = os.getenv("GLM_EMBED_MODEL", "embedding-3")
KEY = os.getenv("GLM_API_KEY","2d875673d4834b3a9868ae3c0d77334a.kboCOVWGKHb99Xw0")

class MetricStore:
    def __init__(self, path: str = "metrics.yaml"):
        with open(path, encoding="utf-8") as f:
            self.metrics = yaml.safe_load(f)["metrics"]

    def get(self, name_or_alias: str):
        for m in self.metrics:
            if name_or_alias == m["name"] or name_or_alias in m.get("aliases", []):
                return m
        return None

    def all_labels(self):
        return [f'{m["label"]}({m["name"]})' for m in self.metrics]

    def prompt_block(self) -> str:
        """给 LLM 的指标清单：只暴露 label / name / 可用维度 / 单位。"""
        lines = []
        for m in self.metrics:
            lines.append(
                f'- {m["label"]} | name={m["name"]} | 维度={",".join(m["dimensions"])} '
                f'| 单位={m["unit"]} | {m["description"]}'
            )
        return "\n".join(lines)


def doc_text(m: dict) -> str:
    """一个指标的检索文本：越贴业务说法，越容易被自然语言命中。"""
    return (
        f"指标：{m['label']}（{m['name']}）\n"
        f"别名：{'、'.join(m.get('aliases', []))}\n"
        f"口径：{m['description']}\n"
        f"计算式：{m['expr']}，来源表 {m['table']}，时间字段 {m['time_field']}\n"
        f"可用维度：{'、'.join(m['dimensions'])}\n"
        f"单位：{m['unit']}"
    )


def glm_embed(texts: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=KEY, base_url=BASE)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _cos(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class MetricIndex:
    def __init__(
        self,
        store: MetricStore,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        cache_path: str = "metric_emb_cache.json",
    ):
        self.store = store
        self.embed_fn = embed_fn or glm_embed
        self.cache_path = Path(cache_path)
        self.names: List[str] = []
        self.vectors: List[List[float]] = []

    def _docs(self) -> Tuple[List[str], List[str]]:
        names, docs = [], []
        for m in self.store.metrics:
            names.append(m["name"])
            docs.append(doc_text(m))
        return names, docs

    def build(self, force: bool = False) -> "MetricIndex":
        names, docs = self._docs()
        fingerprint = hashlib.md5("\n".join(docs).encode()).hexdigest()
        if not force and self.cache_path.exists():
            cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                self.names, self.vectors = cached["names"], cached["vectors"]
                return self
        self.names = names
        self.vectors = self.embed_fn(docs)
        self.cache_path.write_text(
            json.dumps({"fingerprint": fingerprint, "names": names, "vectors": self.vectors}),
            encoding="utf-8",
        )
        return self

    def lexical_hits(self, query: str) -> List[str]:
        """别名/指标名直接出现在问句里的，算强命中，排在最前面。"""
        hits = []
        for m in self.store.metrics:
            keys = [m["name"], m["label"]] + m.get("aliases", [])
            if any(k in query for k in keys):
                hits.append(m["name"])
        return hits

    def recall(
        self, query: str, top_k: int = 2, min_score: float = 0.25
    ) -> List[Tuple[str, float]]:
        """混合召回：先放别名强命中，再用向量相似度补足 TopK。"""
        qv = self.embed_fn([query])[0]
        scored = {n: _cos(qv, v) for n, v in zip(self.names, self.vectors)}
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        result: List[Tuple[str, float]] = [(n, 1.0) for n in self.lexical_hits(query)]
        for n, s in ranked:
            if len(result) >= top_k:
                break
            if n in {x[0] for x in result} or s < min_score:
                continue
            result.append((n, round(s, 4)))
        return result[:top_k]


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "成交额为什么掉了"
    idx = MetricIndex(MetricStore()).build()
    print(f"问题：{q}")
    for name, score in idx.recall(q):
        print(f"  {name:<14} {score:.4f}")