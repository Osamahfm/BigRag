import json
import re
from html import escape

import pandas as pd
import streamlit as st

from rag.generation import citation
from rag.text import tokens


def highlighted_excerpt(text, query):
    words = sorted(set(tokens(query)), key=len, reverse=True)[:16]
    if not words:
        return escape(text)
    pattern = re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)
    result, previous = [], 0
    for match in pattern.finditer(text):
        result.extend([escape(text[previous:match.start()]), f"<mark>{escape(match[0])}</mark>"])
        previous = match.end()
    result.append(escape(text[previous:]))
    return "".join(result)


def evidence_view(sources, cited_ids=None, query=""):
    cited_ids = set(cited_ids or [])
    for source in sources:
        location = f"Page {source['page']}" if source.get("page") else source.get("section", "Document")
        with st.container(border=True):
            columns = st.columns([4, 1])
            columns[0].markdown(f"**{source['source']}**")
            columns[0].caption(location)
            columns[1].caption("CITED" if source["id"] in cited_ids else "RETRIEVED")
            html = highlighted_excerpt(source["text"], query)
            direction = "rtl" if re.search(r"[\u0600-\u06ff]", source["text"]) else "ltr"
            st.markdown(f'<div dir="{direction}" style="white-space:pre-wrap;font-size:13px;line-height:1.7;color:#b8c9df">{html}</div>', unsafe_allow_html=True)
            scores = f"{source.get('score_type', 'rank')} {source.get('score', 0):.4f}"
            if "dense_score" in source:
                scores += f" · cosine {source['dense_score']:.3f}"
            if "sparse_score" in source:
                scores += f" · BM25 {source['sparse_score']:.3f}"
            if "rerank_score" in source:
                scores += f" · rerank {source['rerank_score']:.3f}"
            if "mmr_score" in source:
                scores += f" · diversity {source['mmr_score']:.3f}"
            st.caption(scores)
    st.caption("Ranking scores are not confidence probabilities. Citation checks verify source identity, not factual correctness.")


def ranking_table(candidates):
    return [{"Rank": i, "Document": c["source"],
             "Location": f"Page {c['page']}" if c.get("page") else c.get("section", ""),
             "Score": round(c.get("score", 0), 4), "Cosine": c.get("dense_score"),
             "BM25": c.get("sparse_score"), "Rerank": c.get("rerank_score"),
             "MMR": c.get("mmr_score"), "Chunk ID": c["id"][:12]}
            for i, c in enumerate(candidates, 1)]


def trace_view(debug):
    stages = debug.get("timings", {})
    if stages:
        st.bar_chart(pd.DataFrame([{"Stage": k.replace("_ms", "").replace("_", " "), "ms": v}
                                   for k, v in stages.items()]).set_index("Stage"))
    st.caption(f"Query: {debug.get('rewritten_query', '')}")
    st.write("Search variants", debug.get("expanded_queries", []))
    before, after, config = st.tabs(["Candidate ranking", "Selected evidence", "Configuration & tools"])
    with before:
        st.dataframe(ranking_table(debug.get("candidates", [])), width="stretch", hide_index=True)
    with after:
        st.dataframe(ranking_table(debug.get("diversified", debug.get("reranked", []))), width="stretch", hide_index=True)
        st.json(debug.get("context_stats", {}))
    with config:
        st.json({"method": debug.get("retrieval_method"), "diversity": debug.get("diversity_method"),
                 "agent_decisions": debug.get("agent_decisions", []), "configuration": debug.get("configuration", {})})


def message_view(message, debug=False, store=None):
    with st.chat_message(message["role"], avatar=":material/hexagon:" if message["role"] == "assistant" else ":material/person:"):
        st.markdown(message["content"])
        if message.get("source_notice"):
            st.caption(message["source_notice"])
        if message["role"] != "assistant":
            return
        cited = {c["id"] for c in message.get("cited_sources", [])}
        m = message.get("metrics", {})
        if m:
            st.caption(f"{m['total_ms']/1000:.2f}s · {m['sources']} passages · "
                       f"{len(cited)} citations · {m.get('mode', 'RAG')} · {m.get('backend', '')}")
        for warning in message.get("debug", {}).get("warnings", []):
            st.warning(warning)
        if message.get("sources"):
            with st.expander(f"Evidence · {len(cited)} cited / {len(message['sources'])} retrieved"):
                evidence_view(message["sources"], cited, message.get("query", ""))
        if debug and message.get("debug"):
            with st.expander("Query trace"):
                trace_view(message["debug"])
        if message.get("verification"):
            with st.expander("Answer verification"):
                st.json(message["verification"])
        message_id = message.get("message_id")
        if store and message_id:
            feedback = store.feedback().get(message_id, {"rating": 0, "note": "", "saved": 0})
            actions = st.columns([1, 1.4, 1.3, 2])
            if actions[0].button("Helpful" if feedback["rating"] != 1 else "✓ Helpful", key=f"helpful_{message_id}"):
                store.set_feedback(message_id, 1, feedback["note"], feedback["saved"])
                st.rerun()
            if actions[1].button("Needs review" if feedback["rating"] != -1 else "✓ Review", key=f"review_{message_id}"):
                store.set_feedback(message_id, -1, feedback["note"], feedback["saved"])
                st.rerun()
            if actions[2].button("Saved" if feedback["saved"] else "Save answer", key=f"save_{message_id}"):
                store.set_feedback(message_id, feedback["rating"], feedback["note"], not feedback["saved"])
                st.rerun()
            actions[3].download_button("Download answer", message["content"] + "\n\n" + "\n".join(citation(s) for s in message.get("sources", [])),
                file_name="atlas-answer.md", mime="text/markdown", key=f"answer_download_{message_id}")
            if feedback["rating"] == -1:
                with st.expander("Review note"):
                    note = st.text_area("What should be improved?", feedback["note"], key=f"review_note_{message_id}")
                    if st.button("Save review note", key=f"save_note_{message_id}"):
                        store.set_feedback(message_id, feedback["rating"], note, feedback["saved"])
                        st.rerun()
            if debug:
                st.download_button("Export query trace", json.dumps(message, ensure_ascii=False, indent=2),
                    "atlas-trace.json", "application/json", key=f"trace_download_{message_id}")
