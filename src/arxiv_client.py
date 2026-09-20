import arxiv

_client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=3)


def _to_dict(r: arxiv.Result) -> dict:
    return {
        "arxiv_id": r.get_short_id().split("v")[0],
        "title": r.title.strip().replace("\n", " "),
        "authors": [a.name for a in r.authors],
        "abstract": r.summary.strip().replace("\n", " "),
        "pdf_url": r.pdf_url,
        "link": r.entry_id,
        "categories": r.categories,
        "published": r.published.date().isoformat(),
    }


def get_by_id(arxiv_id: str) -> list[dict]:
    return [_to_dict(r) for r in _client.results(arxiv.Search(id_list=[arxiv_id]))]


def search(query: str, max_results: int = 10, by_date: bool = False) -> list[dict]:
    sort = arxiv.SortCriterion.SubmittedDate if by_date else arxiv.SortCriterion.Relevance
    s = arxiv.Search(query=query, max_results=max_results, sort_by=sort)
    return [_to_dict(r) for r in _client.results(s)]