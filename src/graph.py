from langgraph.graph import StateGraph, START, END
from .state import AgentState
from . import nodes as n


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("understand", n.understand_node)
    g.add_node("retrieve", n.retrieve_node)
    g.add_node("broaden", n.broaden_node)
    g.add_node("select", n.select_node)
    g.add_node("fail", n.fail_node)
    g.add_node("fetch_parse", n.fetch_parse_node)
    g.add_node("chunk_embed", n.chunk_embed_node)
    g.add_node("summarize", n.summarize_node)
    g.add_node("qa", n.qa_node)

    g.add_edge(START, "understand")
    g.add_edge("understand", "retrieve")
    g.add_conditional_edges("retrieve", n.route_after_retrieve,
                            {"broaden": "broaden", "fail": "fail",
                             "select": "select", "fetch": "fetch_parse"})
    g.add_edge("broaden", "retrieve")
    g.add_edge("select", "fetch_parse")
    g.add_edge("fetch_parse", "chunk_embed")
    g.add_edge("chunk_embed", "summarize")
    g.add_edge("summarize", "qa")
    g.add_conditional_edges("qa", n.route_after_qa, {"qa": "qa", "end": END})
    g.add_edge("fail", END)
    return g.compile()