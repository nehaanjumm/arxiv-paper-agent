import re
import json
from pathlib import Path
from . import llm, arxiv_client, pdf_parser, vectorstore as vs

ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
MAX_RETRIES = 2
MAX_DIST = 0.7   # cosine distance; above this = "not in paper". Tune this!


# ---------- 1. Query understanding ----------
def understand_node(state):
    text = state["user_input"].strip()
    m = ID_RE.search(text)
    if m:
        return {"intent": "paper_lookup", "arxiv_id": m.group(1), "retries": 0}
    out = llm.chat_json([
        {"role": "system", "content": "Extract 2-4 concise search keywords/phrases for an arXiv "
         'search from the user request. Reply JSON: {"keywords": ["..."]}'},
        {"role": "user", "content": text}])
    kws = [k for k in out.get("keywords", []) if k] or [text]
    by_date = any(w in text.lower() for w in ("recent", "latest", "new"))
    return {"intent": "topic_search", "keywords": kws,
            "sort_by": "date" if by_date else "relevance", "retries": 0}


# ---------- 2. Retrieval ----------
def retrieve_node(state):
    try:
        if state["intent"] == "paper_lookup":
            cands = arxiv_client.get_by_id(state["arxiv_id"])
        else:
            op = " AND " if state.get("retries", 0) == 0 else " OR "   # broaden on retry
            query = op.join(f'all:"{k}"' for k in state["keywords"])
            cands = arxiv_client.search(query, 10, by_date=state.get("sort_by") == "date")
    except Exception as e:
        return {"candidates": [], "error": f"arXiv request failed: {e}"}
    return {"candidates": cands}


def route_after_retrieve(state):
    n = len(state.get("candidates", []))
    if n == 0:
        if state["intent"] == "topic_search" and state.get("retries", 0) < MAX_RETRIES:
            return "broaden"
        return "fail"
    return "fetch" if n == 1 else "select"


def broaden_node(state):
    print("⚠️  No results, broadening query...")
    return {"retries": state.get("retries", 0) + 1}


def fail_node(state):
    msg = state.get("error") or "No papers found. Try a different/broader topic or a direct arXiv ID."
    print(f"❌ {msg}")
    return {"error": msg, "done": True}


# ---------- 3. Selection ----------
def select_node(state):
    cands = state["candidates"][:8]
    listing = "\n".join(f"[{i}] {c['title']} ({c['published']}): {c['abstract'][:400]}"
                        for i, c in enumerate(cands))
    idx = 0
    try:
        out = llm.chat_json([
            {"role": "system", "content": "Pick the paper that best matches the user's topic. "
             'Reply JSON: {"index": <int>, "reason": "<short>"}'},
            {"role": "user", "content": f"Topic: {state['user_input']}\n\n{listing}"}])
        idx = int(out["index"])
        print(f"📌 Selected [{idx}]: {out.get('reason', '')}")
    except Exception:
        print("⚠️  Ranking failed, using top search result.")
    idx = idx if 0 <= idx < len(cands) else 0
    return {"selected": cands[idx]}


# ---------- 4. Fetch & parse ----------
def fetch_parse_node(state):
    paper = state.get("selected") or state["candidates"][0]
    print(f"📄 {paper['title']}  ({paper['arxiv_id']})")
    try:
        sections, quality = pdf_parser.parse_pdf(paper["pdf_url"])
    except Exception as e:
        print(f"⚠️  PDF parse failed ({e}). Falling back to abstract-only mode.")
        sections, quality = {}, "abstract_only"
    if quality == "abstract_only":
        print("⚠️  Answers will be limited to the abstract.")
    return {"selected": paper, "sections": sections, "parse_quality": quality}


# ---------- 5. Chunk & embed ----------
def chunk_embed_node(state):
    name = vs.index_paper(state["selected"], state["sections"])
    return {"collection_name": name}


# ---------- 6. Summarize ----------
GROUPS = {
    "intro": (["introduction", "background"], 2500),
    "method": (["method", "methods", "methodology", "approach"], 3000),
    "results": (["experiments", "results", "evaluation"], 3000),
    "end": (["limitations", "discussion", "conclusion", "conclusions"], 2500),
}


def _build_context(sections: dict) -> str:
    parts = []
    for label, (names, budget) in GROUPS.items():
        txt = " ".join(sections.get(n, "") for n in names).strip()
        if txt:
            parts.append(f"## {label.upper()}\n{txt[:budget]}")
    if not parts and sections.get("body"):       # heading detection failed
        b = sections["body"]
        parts.append(b[:6000] + "\n...\n" + b[-4000:])
    return "\n\n".join(parts)


BRIEF_PROMPT = """You are a careful research analyst. Using ONLY the paper text provided, produce JSON with keys:
"why_it_matters": one plain-English paragraph,
"problem": string,
"method": list of bullet strings,
"key_results": list of strings (include numbers ONLY if present in the text),
"limitations": list of strings. This MUST be non-empty. If the paper doesn't state limitations,
   infer them from the text and prefix each with "(inferred) ",
"followup_questions": list of 4-5 questions a reader might ask.
Do not invent facts. If something is unknown, say so."""


def summarize_node(state):
    paper = state["selected"]
    context = f"## ABSTRACT\n{paper['abstract']}\n\n" + _build_context(state["sections"])
    body = llm.chat_json([{"role": "system", "content": BRIEF_PROMPT},
                          {"role": "user", "content": f"Title: {paper['title']}\n\n{context}"}])
    briefing = {   # metadata comes from arXiv, NOT the LLM (avoids hallucinated fields)
        "title": paper["title"], "authors": paper["authors"], "arxiv_id": paper["arxiv_id"],
        "published": paper["published"], "link": paper["link"],
        "parse_quality": state["parse_quality"], **body}
    md = to_markdown(briefing)
    Path("outputs").mkdir(exist_ok=True)
    Path(f"outputs/{paper['arxiv_id']}.md").write_text(md, encoding="utf-8")
    Path(f"outputs/{paper['arxiv_id']}.json").write_text(json.dumps(briefing, indent=2), encoding="utf-8")
    print("\n" + md)
    return {"briefing": briefing}


def to_markdown(b: dict) -> str:
    bullets = lambda xs: "\n".join(f"- {x}" for x in xs)
    note = "\n> ⚠️ Abstract-only mode (PDF could not be parsed).\n" if b["parse_quality"] == "abstract_only" else ""
    return f"""# {b['title']}
**Authors:** {', '.join(b['authors'])}
**arXiv:** {b['arxiv_id']} | **Published:** {b['published']} | {b['link']}
{note}
## Why it matters
{b['why_it_matters']}

## Problem
{b['problem']}

## Method
{bullets(b['method'])}

## Key results
{bullets(b['key_results'])}

## Limitations
{bullets(b['limitations'])}

## Suggested follow-up questions
{bullets(b['followup_questions'])}
"""


# ---------- 7. QA loop (RAG) ----------
QA_PROMPT = """Answer the question using ONLY the numbered context excerpts from the paper.
Cite excerpts like [1], [2]. If the excerpts do not contain the answer, reply exactly:
"I couldn't find that in the paper." Do not use outside knowledge."""


def qa_node(state):
    q = input("\n❓ Ask about the paper (or 'exit'): ").strip()
    if q.lower() in {"exit", "quit", "q", ""}:
        print("👋 Bye!")
        return {"done": True}
    history = state.get("chat_history", [])

    standalone = q   # make follow-ups self-contained for better retrieval
    if history:
        standalone = llm.chat([
            {"role": "system", "content": "Rewrite the last question as a standalone question "
             "using the chat history. Output only the question."},
            {"role": "user", "content": json.dumps(history[-4:]) + "\nLast question: " + q}
        ], temperature=0).strip()

    hits = [h for h in vs.search(state["collection_name"], standalone, k=5)
            if h["distance"] <= MAX_DIST]
    if not hits:
        answer = "I couldn't find that in the paper."
    else:
        ctx = "\n\n".join(f"[{i+1}] (section: {h['section']}) {h['text']}" for i, h in enumerate(hits))
        answer = llm.chat([{"role": "system", "content": QA_PROMPT},
                           {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {standalone}"}],
                          temperature=0)
        sources = ", ".join(sorted({h["section"] for h in hits}))
        answer += f"\n\n_Sources: sections → {sources}_"
    print(f"\n🤖 {answer}")
    history = history + [{"q": q, "a": answer}]
    return {"chat_history": history}


def route_after_qa(state):
    return "end" if state.get("done") else "qa"