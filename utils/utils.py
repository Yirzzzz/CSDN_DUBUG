
# corpus_summarizer.py
from __future__ import annotations
import re, math, unicodedata
from typing import List, Dict, Any, Tuple, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

try:
    import networkx as nx
except Exception:
    nx = None  # 没有 networkx 时会用简单中心性

# ---------- 基础工具 ----------
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?…]|[.?!])\s+|\n+")

def split_sentences(text: str, max_sents: int = 80) -> List[str]:
    text = unicodedata.normalize("NFKC", (text or "")).strip()
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    # 过短/过长过滤
    cleaned = [s for s in sents if 6 <= len(s) <= 400]
    return cleaned[:max_sents] if cleaned else (sents[:max_sents] if sents else [])

def est_tokens(s: str) -> int:
    # 粗略 token 估计：英文按词，中文按字符/2，混合取较大者
    words = len(re.findall(r"\w+", s))
    chars = len(re.sub(r"\s+", "", s))
    return max(words, int(chars / 2)) or 1

def tfidf_vectors(texts: List[str], analyzer: str = "char_wb", ngram_range=(2,5)) -> Tuple[np.ndarray, TfidfVectorizer]:
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, lowercase=True)
    X = vec.fit_transform(texts)
    return X, vec

def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    Xn = normalize(X, norm="l2", axis=1)
    return (Xn @ Xn.T).toarray()

# ---------- 文内摘要（整篇） ----------
def textrank_centrality(X_sent: np.ndarray) -> np.ndarray:
    sims = cosine_similarity_matrix(X_sent)
    np.fill_diagonal(sims, 0.0)
    if nx is None:
        # 退化：每句中心性=相似度和
        cen = sims.sum(axis=1)
        return (cen / (cen.max() + 1e-9))
    G = nx.from_numpy_array(sims)
    pr = nx.pagerank(G, weight="weight", max_iter=200, tol=1e-6)
    v = np.array([pr[i] for i in range(len(pr))], dtype=float)
    return v / (v.max() + 1e-9)

def mmr_select(
    cand_vecs: np.ndarray,
    query_vec: np.ndarray,
    scores: np.ndarray,
    max_sents: int = 8,
    token_budget: int = 360,
    lambda_mult: float = 0.7,
) -> List[int]:
    """在文内做 MMR 选择，兼顾相关性与去冗，并控制 token/句子上限。"""
    if cand_vecs.shape[0] == 0:
        return []
    C = normalize(cand_vecs, axis=1)
    q = normalize(query_vec, axis=1)
    rel = (C @ q.T).ravel()
    # 归一化再与“打分”融合
    if rel.max() > 0: rel = rel / rel.max()
    w = 0.5 * (rel + (scores / (scores.max() + 1e-9)))

    selected: List[int] = []
    budget = 0
    avail: Set[int] = set(range(C.shape[0]))

    # 先选一个最相关的
    i0 = int(np.argmax(w))
    selected.append(i0)
    budget += est_tokens_sent_index(i0)
    avail.remove(i0)

    def redundancy(idx: int) -> float:
        if not selected: return 0.0
        return float((C[idx] @ C[selected].T).max())

    while avail and len(selected) < max_sents and budget < token_budget:
        best_j, best_score = None, -1e9
        for j in list(avail):
            div = redundancy(j)
            mmr = lambda_mult * w[j] - (1 - lambda_mult) * div
            if mmr > best_score:
                best_score, best_j = mmr, j
        if best_j is None:
            break
        # 预算检查
        if budget + est_tokens_sent_index(best_j) > token_budget:
            avail.remove(best_j)  # 放弃这句，尝试下一句
            continue
        selected.append(best_j)
        budget += est_tokens_sent_index(best_j)
        avail.remove(best_j)
    return selected

# 为 mmr_select 提供一个可替换的 token 估计（通过闭包注入）
_SENT_TOKEN_CACHE: Dict[int,int] = {}
def est_tokens_sent_index(i: int) -> int:
    return _SENT_TOKEN_CACHE.get(i, 20)

def summarize_document(
    doc_text: str,
    question: str,
    max_sents: int = 8,
    token_budget: int = 360,
    char_ngram: Tuple[int,int] = (2,5),
) -> List[str]:
    """
    返回：该文的抽取式摘要句子列表（≤max_sents，≈token_budget）
    """
    sents = split_sentences(doc_text, max_sents=120)
    if not sents:
        return []
    # 向量化（句子+问题）
    X_all, vec = tfidf_vectors(sents + [question], analyzer="char_wb", ngram_range=char_ngram)
    X_sent = X_all[:-1]
    X_q = X_all[-1:]

    # TextRank 中心性
    cen = textrank_centrality(X_sent)

    # 与问题相关性
    Xn = normalize(X_all, axis=1)
    rel = (Xn[:-1] @ Xn[-1:].T).ravel()
    if rel.max() > 0: rel = rel / rel.max()

    # 综合分
    score = 0.6 * cen + 0.4 * rel

    # 为预算估计注入句子 token
    _SENT_TOKEN_CACHE.clear()
    for i, s in enumerate(sents):
        _SENT_TOKEN_CACHE[i] = est_tokens(s)

    # MMR 选择
    idx = mmr_select(X_sent, X_q, score, max_sents=max_sents, token_budget=token_budget, lambda_mult=0.7)
    idx_sorted = sorted(idx)  # 保持原文顺序更可读
    return [sents[i] for i in idx_sorted]

# ---------- 跨文档融合 ----------
def cluster_sentences(
    sentences: List[str],
    doc_ids: List[int],
    sim_thresh: float = 0.75,
    char_ngram: Tuple[int,int] = (2,5),
) -> List[Dict[str,Any]]:
    """
    贪心聚类：返回簇列表，每簇含 sentences / doc_set / rep_idx 等
    """
    if not sentences:
        return []
    X, vec = tfidf_vectors(sentences, analyzer="char_wb", ngram_range=char_ngram)
    Xn = normalize(X, axis=1)

    clusters: List[Dict[str,Any]] = []
    centroids: List[np.ndarray] = []  # 直接存平均向量

    for i, v in enumerate(Xn):
        placed = False
        for ci, cen in enumerate(centroids):
            sim = float(v @ cen)
            if sim >= sim_thresh:
                # 入簇
                clusters[ci]["indices"].append(i)
                clusters[ci]["doc_set"].add(doc_ids[i])
                # 更新质心
                k = len(clusters[ci]["indices"])
                centroids[ci] = (cen * (k-1) + v) / k
                placed = True
                break
        if not placed:
            clusters.append({"indices":[i], "doc_set": set([doc_ids[i]]), "centroid": v})
            centroids.append(v)

    # 选代表句：支持度优先，其次与簇内平均相似度
    sims = (Xn @ Xn.T).toarray()
    for c in clusters:
        idxs = c["indices"]
        sup = len(c["doc_set"])
        # 平均相似度
        avg_sim = []
        for j in idxs:
            if len(idxs) == 1:
                avg_sim.append(0.0)
            else:
                avg_sim.append(float(np.mean([sims[j,k] for k in idxs if k != j])))
        # 代表
        best = max(range(len(idxs)), key=lambda t: (sup, avg_sim[t]))
        c["rep_idx"] = idxs[best]
        c["support"] = sup
        c["avg_sim"] = float(avg_sim[best] if avg_sim else 0.0)
        c["sentences"] = [sentences[j] for j in idxs]
        c["docs"] = sorted({f"A{d+1}" for d in c["doc_set"]})
    return clusters

def fuse_across_docs(
    per_doc_summaries: List[List[str]],
    max_total_sentences: int = 20,
    sim_thresh: float = 0.75,
) -> Dict[str, List[Dict[str,Any]]]:
    """
    返回三组句子：consensus / alternatives / supplements
    每条带 citations（如 ['A3','A7']）
    """
    # 展开为句子列表
    all_sents, doc_ids = [], []
    for di, sents in enumerate(per_doc_summaries):
        for s in sents:
            all_sents.append(s)
            doc_ids.append(di)
    clusters = cluster_sentences(all_sents, doc_ids, sim_thresh=sim_thresh)

    # 分桶：共识(支持>=3)、差异(支持=2)、补充(支持=1)
    cons = [c for c in clusters if c["support"] >= 3]
    alt  = [c for c in clusters if c["support"] == 2]
    supp = [c for c in clusters if c["support"] == 1]

    # 排序规则
    cons.sort(key=lambda c: (c["support"], c["avg_sim"]), reverse=True)
    alt.sort(key=lambda c: (c["avg_sim"], len(c["docs"])), reverse=True)
    supp.sort(key=lambda c: c["avg_sim"], reverse=True)

    # 配额（60/25/15）
    n1 = int(max_total_sentences * 0.60)
    n2 = int(max_total_sentences * 0.25)
    n3 = max_total_sentences - n1 - n2

    def pick(clist, n):
        picked = []
        for c in clist[:n]:
            picked.append({
                "text": all_sents[c["rep_idx"]],
                "citations": c["docs"],
            })
        return picked

    return {
        "consensus":   pick(cons, n1),
        "alternatives":pick(alt,  n2),
        "supplements": pick(supp, n3),
    }

# ---------- 一体化入口 ----------
def compress_corpus(
    corpus: List[str],
    question: str,
    per_doc_tokens: int = 360,
    per_doc_max_sents: int = 8,
    fuse_total_sents: int = 20,
) -> Dict[str, Any]:
    """
    输入：
      corpus: List[str]  # 每个元素=整篇文章字符串
      question: str      # 问题（用于相关性）
    输出：
    {
      "per_doc": [ {"id":"A1","summary":[...],"orig_len":N}, ... ],
      "fused": { "consensus":[{text,citations}], "alternatives":[...], "supplements":[...] }
    }
    """
    per_doc = []
    for i, doc in enumerate(corpus):
        summary = summarize_document(
            doc_text=doc,
            question=question,
            max_sents=per_doc_max_sents,
            token_budget=per_doc_tokens,
        )
        per_doc.append({
            "id": f"A{i+1}",
            "summary": summary,
            "orig_len": len(doc),
        })

    fused = fuse_across_docs([x["summary"] for x in per_doc], max_total_sentences=fuse_total_sents)
    return {"per_doc": per_doc, "fused": fused}

def _truncate(s: str, max_chars: int = 2000) -> str:
    s = s or ""
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"

def _shorten(s: str, n: int = 300) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"

def _http():
    # 不标注返回类型，避免第三方类型在注解求值时被解析
    return httpx.Client(
        timeout=20,
        headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
        follow_redirects=True,
    )

def _search_with_provider(
    query: str,
    sites: list[str],
    limit: int,
    *,
    freshness: str | None = None,   # "Day" | "Week" | "Month" | "Year"
    mkt: str = "zh-CN",
    provider: str | None = None     # 可显式指定: "bing" | "ddg"
) -> list[dict[str, Any]]:
    """
    统一检索入口：
    - 优先走 Bing（若提供 BING_KEY 或 provider="bing"）
    - 否则走 duckduckgo_search（无钥，类似 websearch 的用法）
    返回: [{title, url, snippet, source}]
    """
    provider = (provider or os.getenv("SEARCH_PROVIDER") or ("bing" if BING_KEY else "ddg")).lower()
    q = f"{query} " + " ".join(f"site:{s}" for s in sites if s)
    results: list[dict[str, Any]] = []

    # -------------------- 分支 A：Bing Web Search API --------------------
    if provider == "bing":
        if not BING_KEY:
            raise RuntimeError("BING_KEY 未设置；若要无钥检索，请把 provider 改为 'ddg' 或不传。")
        count = max(1, min(int(limit or 1), 50))
        with _http() as client:
            params = {
                "q": q,
                "count": count,
                "mkt": mkt,
                "responseFilter": "Webpages",
                "safeSearch": "Off",
            }
            if freshness:
                params["freshness"] = freshness  # Day/Week/Month
            r = client.get(
                "https://api.bing.microsoft.com/v7.0/search",
                params=params,
                headers={"Ocp-Apim-Subscription-Key": BING_KEY},
            )
            r.raise_for_status()
            data = r.json()
            for item in (data.get("webPages", {}).get("value") or []):
                results.append({
                    "title": item.get("name", "") or "",
                    "url": item.get("url", "") or "",
                    "snippet": item.get("snippet", "") or "",
                    "source": "bing",
                })

    # -------------------- 分支 B：duckduckgo_search（无钥） --------------------
    else:
        if DDGS is None:
            raise RuntimeError("未安装 duckduckgo_search，请先执行: pip install duckduckgo_search")
        # freshness 映射到 DDG 的 timelimit：d/w/m/y
        tl_map = {"Day": "d", "Week": "w", "Month": "m", "Year": "y"}
        timelimit = tl_map.get(freshness or "", None)
        # 地区用 'wt-wt' 更通用；safesearch 可选: "off"/"moderate"/"strict"
        max_results = max(1, int(limit or 1)) * 2  # 稍多取一些，便于去重
        with DDGS(timeout=15) as ddgs:
            for item in ddgs.text(q, region="wt-wt", safesearch="moderate",
                                  timelimit=timelimit, max_results=max_results):
                # item: {'title':..., 'href':..., 'body':...}
                url = item.get("href") or ""
                if not url:
                    continue
                results.append({
                    "title": item.get("title", "") or "",
                    "url": url,
                    "snippet": item.get("body", "") or "",
                    "source": "ddg",
                })

    # -------------------- 去重 & 截断 --------------------
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for it in results:
        u = (it.get("url") or "").split("#")[0]
        if u and u not in seen:
            seen.add(u)
            deduped.append(it)
        if len(deduped) >= limit:
            break
    return deduped



def _extract_article(html_text: str, url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html_text, "lxml")
    title_node = soup.select_one("h1, .title-article, .article-title, title")
    title = title_node.get_text(" ", strip=True) if title_node else ""

    author_node = soup.select_one('meta[name="author"]')
    author = author_node["content"] if (author_node and author_node.has_attr("content")) else ""

    date_node = soup.select_one('meta[property="article:published_time"], time, .time, .date')
    published = (
        date_node["content"] if (date_node and date_node.has_attr("content"))
        else (date_node.get_text(strip=True) if date_node else "")
    )

    content_node = soup.select_one("#content_views, .blog-content-box, article, .markdown-body")
    text = ""
    if content_node:
        for bad in content_node.select(".hide-article-box, .csdn-toolbar, script, style"):
            bad.decompose()
        text = content_node.get_text("\n", strip=True)

    return {
        "url": url,
        "title": title,
        "author": author,
        "published": published,
        "excerpt": _shorten(text, 800),
    }