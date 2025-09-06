from __future__ import annotations
import os, re, logging
import random
import time
from typing import Any

import httpx
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from mcp.server.fastmcp import FastMCP

from utils.utils import compress_corpus, _truncate, _search_with_provider, _http, _extract_article

try:
    from ddgs import DDGS
except Exception:
    DDGS = None
logging.getLogger("httpx").disabled = True
logging.getLogger("httpcore").disabled = True
logging.getLogger("ddgs").disabled = True
# 日志（输出到 stderr，不污染 STDIO 协议）
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

mcp = FastMCP("CSDNHelper")

SEARCH_API  = os.getenv("SEARCH_API", "").lower()  # 'searchapi' | 'serpapi' | 'bing'
SEARCH_KEY  = os.getenv("SEARCH_API_KEY", "")
BING_KEY    = os.getenv("BING_KEY", "")
RSSHUB_BASE = os.getenv("RSSHUB_BASE", "https://rsshub.app")

DEFAULT_SITES = ["csdn.net"]  # 可加: 'juejin.cn', 'segmentfault.com', 'zhihu.com'
UA = "CSDN-MCP/0.1 (+https://example.local)"

# --------------------------- MCP 工具 ---------------------------

@mcp.tool()
def search_csdn(query: str, limit: int = 8, include_sites: list[str] | None = None) -> list[str]:
    """
    站内检索：默认 site:csdn.net；可通过 include_sites 扩展站群（如 'juejin.cn'）。
    返回: [[content]]
    """
    sites = include_sites or DEFAULT_SITES
    articles =  _search_with_provider(query, sites, limit)
    key = " ".join([ln.strip() for ln in query.splitlines() if ln.strip()][:3])
    from rapidfuzz import fuzz
    scored = []
    corpus = []
    with _http() as client:
        for i, art in enumerate(articles):
            time.sleep(random.randint(1, 5))
            url = art.get('url')
            r = client.get(url)
            r.raise_for_status()
            content = _extract_article(r.text, url)
            art["_content"] = content['excerpt']
            art["_score"] = fuzz.token_set_ratio(key, content['title'])
            scored.append(art)
    scored.sort(key=lambda x: x["_score"], reverse=True)
    for i, it in enumerate(scored[:int(0.8*len(scored))], 1):
        corpus.append(it['_content'])
    return corpus



@mcp.tool()
def compress_corpus_tool(
    corpus: list[str],
    question: str,
    per_doc_tokens: int = 360,
    per_doc_max_sents: int = 8,
    fuse_total_sents: int = 20,
    preview_chars: int = 2000,  # 为防止 MCP 报文过大，对句子做截断预览
) -> dict[str, Any]:
    """
    对传入的整文语料进行两级压缩：
      - 文内抽取式摘要（TextRank×0.6 + TF-IDF相关性×0.4，MMR 去冗）
      - 跨文档融合（共识/差异/补充，带来源引用）

    返回:
    {
      "per_doc": [{"id":"A1","summary":[...],"orig_len":N}, ...],
      "fused": {
         "consensus":[{"text": "...", "citations":["A3","A7"]}],
         "alternatives":[...],
         "supplements":[...]
      }
    }
    """
    if compress_corpus is None:
        raise RuntimeError("corpus_summarizer 未导入成功，请确认 corpus_summarizer.py 在同目录且依赖已安装（scikit-learn、numpy、networkx可选）")

    # 运行压缩
    res = compress_corpus(
        corpus=corpus,
        question=question,
        per_doc_tokens=per_doc_tokens,
        per_doc_max_sents=per_doc_max_sents,
        fuse_total_sents=fuse_total_sents,
    )

    for d in res.get("per_doc", []):
        d["summary"] = [_truncate(s, preview_chars) for s in d.get("summary", [])]
    for bucket in ("consensus", "alternatives", "supplements"):
        lst = res.get("fused", {}).get(bucket, [])
        for item in lst:
            item["text"] = _truncate(item.get("text", ""), preview_chars)

    return res

# --------------------------- MCP 资源 / Prompt ---------------------------

@mcp.resource("rss://csdn/{user}")
def csdn_user_feed(user: str) -> dict[str, str]:
    """返回 CSDN 某个用户的 RSS 条目（通过 RSSHub）。"""
    url = f"{RSSHUB_BASE}/csdn/blog/{user}"
    with _http() as client:
        r = client.get(url)
        r.raise_for_status()
        return {"feed_url": url, "xml": r.text}

@mcp.prompt()
def triage_prompt() -> str:
    """给客户端 LLM 用的模板：把抓到的帖子做要点对比和可操作步骤。"""
    return (
        "你是一名Bug分诊助手。给定若干检索到的帖子（含标题/摘要/链接），"
        "请：1) 总结可行修复步骤；2) 标注版本/平台前提；3) 输出一个最可能的 Root Cause。"
        "用中文，列表输出，最后附上引用的链接。"
    )

def main():
    mcp.run()

if __name__ == "__main__":
    main()
