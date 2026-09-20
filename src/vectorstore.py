import re
import chromadb
from chromadb.utils import embedding_functions

_client = chromadb.PersistentClient(path="chroma_db")
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def collection_name(arxiv_id: str) -> str:
    return "paper_" + re.sub(r"[^a-zA-Z0-9]", "_", arxiv_id)


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Sliding window that prefers to break at sentence ends."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def index_paper(paper: dict, sections: dict) -> str:
    name = collection_name(paper["arxiv_id"])
    try:
        _client.delete_collection(name)
    except Exception:
        pass
    col = _client.create_collection(name, embedding_function=_ef,
                                    metadata={"hnsw:space": "cosine"})
    docs, metas, ids = [], [], []
    all_sections = {"abstract": paper["abstract"], **{k: v for k, v in sections.items()
                                                       if k not in ("references", "abstract")}}
    for sec, text in all_sections.items():
        for c in chunk_text(text):
            docs.append(c)
            metas.append({"section": sec})
            ids.append(f"{paper['arxiv_id']}-{len(ids)}")
    for i in range(0, len(docs), 100):
        col.add(documents=docs[i:i+100], metadatas=metas[i:i+100], ids=ids[i:i+100])
    return name


def search(name: str, query: str, k: int = 5) -> list[dict]:
    col = _client.get_collection(name, embedding_function=_ef)
    res = col.query(query_texts=[query], n_results=k)
    return [{"id": i, "text": d, "section": m["section"], "distance": dist}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                     res["metadatas"][0], res["distances"][0])]