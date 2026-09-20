from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    user_input: str
    intent: str
    arxiv_id: Optional[str]
    keywords: list[str]
    sort_by: str
    retries: int
    candidates: list[dict]
    selected: dict
    sections: dict
    parse_quality: str
    collection_name: str
    briefing: dict
    chat_history: list[dict]
    error: str
    done: bool