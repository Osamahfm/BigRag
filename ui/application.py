"""Application shell: navigation, conversation scope and service configuration."""
import json
import os
from dataclasses import asdict, replace
from types import SimpleNamespace

import streamlit as st
from dotenv import load_dotenv

from config import ROOT, Settings
from rag.embeddings import DemoService, GeminiService
from rag.errors import RAGError
from rag.jobs import IngestionJobs
from rag.library import Library
from rag.pipeline import MODES
from rag.storage import Store
from ui import workspace
from ui.theme import brand, header, install_theme


@st.cache_resource
def job_manager(root):
    return IngestionJobs(Store(root))


def main(root):
    st.set_page_config(page_title="Atlas · Knowledge Workspace", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    load_dotenv(ROOT/".env")
    install_theme()
    store = Store(root)
    experiment = st.session_state.get("search_experiment")
    if experiment and experiment.get("revision") != store.revision():
        st.session_state.pop("search_experiment", None)
    known_fields = asdict(Settings())
    try:
        persisted = store.preference("settings")
        base = Settings(**{key: value for key, value in persisted.items() if key in known_fields})
    except (ValueError, TypeError, AttributeError):
        base = Settings()
        st.warning("Saved settings could not be read; using validated defaults.")
    jobs = job_manager(str(store.root))
    with st.sidebar:
        brand()
        page = st.radio("Navigate", ["Workspace", "Documents", "Search studio", "Evaluation", "Activity", "Settings"],
                        key="navigation", label_visibility="collapsed")
        st.divider()
        st.caption("CONVERSATIONS")
        sessions = store.sessions()
        if not sessions:
            store.new_session()
            sessions = store.sessions()
        session_map = {s["id"]: s["title"] for s in sessions}
        next_session = st.session_state.pop("next_session", None)
        if next_session:
            st.session_state["session_id"] = next_session
        session_id = st.selectbox("Chat session", list(session_map), format_func=session_map.get, key="session_id", label_visibility="collapsed")
        if st.button("New conversation", width="stretch"):
            st.session_state["next_session"] = store.new_session()
            st.rerun()
        session_scope = "conversation:"+session_id
        preference = store.preference(session_scope)
        documents = store.documents()
        doc_map = {d["id"]: d for d in documents}
        if st.session_state.get("loaded_session") != session_id:
            st.session_state["selected_docs"] = [i for i in preference.get("documents", list(doc_map)) if i in doc_map]
            st.session_state["mode"] = preference.get("mode", "Hybrid")
            st.session_state["retrieval"] = preference.get("retrieval", "hybrid")
            st.session_state["answer_style"] = preference.get("detail", "concise")
            for key, default in {"rewrite_queries": True, "expand_queries": True, "route_queries": False}.items():
                st.session_state[key] = preference.get("query_options", {}).get(key, default)
            st.session_state["known_documents"] = list(doc_map)
            st.session_state["loaded_session"] = session_id
        new_docs = [i for i in doc_map if i not in st.session_state.get("known_documents", [])]
        st.session_state["selected_docs"] = [i for i in st.session_state.get("selected_docs", []) if i in doc_map]+new_docs
        st.session_state["known_documents"] = list(doc_map)
        st.caption("KNOWLEDGE SCOPE")
        selected = st.multiselect("Search within", list(doc_map), format_func=lambda i: doc_map[i]["source"], key="selected_docs")
        st.caption(f"{len(documents)} documents · {sum(d['chunks'] for d in documents)} passages")
        st.divider()
        with st.expander("Answer controls", expanded=False):
            backend_default = store.preference("engine").get("backend", "Gemini" if os.getenv("GEMINI_API_KEY", "").strip() else "Offline demo")
            backend_label = st.selectbox("Backend", ["Gemini", "Offline demo"], index=0 if backend_default == "Gemini" else 1, key="backend")
            backend = "demo" if backend_label == "Offline demo" else "gemini"
            mode = st.selectbox("RAG mode", MODES, key="mode")
            retrieval = st.selectbox("Retrieval", ["hybrid", "dense", "sparse"], key="retrieval")
            detail = st.radio("Answer style", ["concise", "detailed"], key="answer_style", horizontal=True)
            debug = st.checkbox("Show query traces", False, key="debug")
            rewriting = st.checkbox("Resolve follow-up questions", key="rewrite_queries")
            expansion = st.checkbox("Query expansion", key="expand_queries")
            routing = st.checkbox("Automatic retrieval routing", key="route_queries")
        with st.expander("Conversation actions"):
            title = st.text_input("Chat title", session_map[session_id], key="title_"+session_id)
            if st.button("Rename chat"):
                store.rename_session(session_id, title)
                st.rerun()
            if st.button("Clear conversation"):
                store.clear_session(session_id)
                st.rerun()
            st.download_button("Export conversation", json.dumps(store.history(session_id), ensure_ascii=False, indent=2),
                               "conversation.json", "application/json")
        settings = replace(base, backend=backend)
        service = None
        try:
            service = DemoService(settings) if backend == "demo" else GeminiService(settings)
            if hasattr(service, "error_observer"):
                service.error_observer = lambda event: store.event("api_error", event)
        except RAGError as exc:
            st.warning(str(exc))
        if backend == "demo":
            st.caption("Offline demo · lexical hash search and extractive quotes. No API calls.")
        else:
            st.caption("Local storage · selected evidence is sent to Google Gemini.")
        with st.expander("Quick start"):
            if st.button("Load example documents", disabled=service is None and settings.chunk_strategy == "semantic", width="stretch"):
                try:
                    library = Library(store, settings, service)
                    for name in ("mapreduce.md", "spark.txt", "biology.txt"):
                        library.ingest(name, (ROOT/"examples"/name).read_bytes())
                    st.rerun()
                except RAGError as exc:
                    st.error(str(exc))
    if store.preference("engine").get("backend") != backend_label:
        store.save_preference("engine", {"backend": backend_label})
    current_preference = {"documents": selected, "mode": mode, "retrieval": retrieval, "detail": detail,
        "query_options": {"rewrite_queries": rewriting, "expand_queries": expansion, "route_queries": routing}}
    if current_preference != preference:
        store.save_preference(session_scope, current_preference)
    header(page, backend, service is not None)
    context = SimpleNamespace(store=store, settings=settings, service=service, jobs=jobs,
        documents=documents, selected=selected, session_id=session_id, history=store.history(session_id),
        mode=mode, retrieval=retrieval, detail=detail, debug=debug,
        query_options={"rewriting": rewriting, "expansion": expansion, "routing": routing})
    pages = {"Workspace": workspace.chat_page, "Documents": workspace.documents_page,
             "Search studio": workspace.search_page, "Evaluation": workspace.evaluation_page,
             "Activity": workspace.activity_page, "Settings": workspace.settings_page}
    pages[page](context)
