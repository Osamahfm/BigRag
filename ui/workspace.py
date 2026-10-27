"""Focused workspace pages. Presentation logic stays outside the RAG core."""
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from config import ROOT
from rag.diversity import diversify
from rag.errors import RAGError
from rag.evaluation import compare, load_dataset
from rag.generation import build_context
from rag.pipeline import Pipeline
from rag.reranking import rerank
from rag.retrieval import Retriever, key
from ui.components import evidence_view, message_view, ranking_table
from ui.theme import card, empty_state


def chat_page(ctx):
    st.title("Turn knowledge into clarity.")
    st.caption("Ask across your library. Follow the reasoning back to its source.")
    metrics = st.columns(4)
    metrics[0].metric("Documents", len(ctx.documents))
    metrics[1].metric("Indexed passages", sum(d["chunks"] for d in ctx.documents))
    metrics[2].metric("Selected sources", len(ctx.selected))
    metrics[3].metric("Conversation turns", len(ctx.history)//2)
    st.write("")
    if not ctx.history:
        empty_state("Start with a question worth exploring.",
            "Upload your knowledge in Documents or load the example library. Ask a focused question, compare ideas, or build a summary with traceable evidence.")
        columns = st.columns(3)
        for column, data in zip(columns, [
            ("01 / DISCOVER", "Find the relevant detail", "Semantic and keyword search work together to surface your evidence."),
            ("02 / REASON", "Connect the bigger picture", "Use conversational, corrective or parent-document workflows."),
            ("03 / VERIFY", "Inspect every reference", "Explore cited excerpts, ranking changes and the selected context.")], strict=True):
            with column:
                card(*data)
        st.write("")
    st.caption(f"{ctx.mode} · {ctx.retrieval} retrieval · {ctx.settings.reranker} reranking · "
               f"{'diverse context' if ctx.settings.diversity else 'ranked context'}")
    for message in ctx.history:
        message_view(message, ctx.debug, ctx.store)
    columns = st.columns(3)
    question = None
    prompts = ["Summarize the selected documents", "Compare MapReduce and Spark", "ما مزايا MapReduce؟"]
    for column, prompt in zip(columns, prompts, strict=True):
        if column.button(prompt, disabled=ctx.service is None or not ctx.selected, width="stretch"):
            question = prompt
    typed = st.chat_input("Ask a question in English or Arabic…", disabled=ctx.service is None or not ctx.selected)
    question = typed or question
    if not question:
        return
    message_view({"role": "user", "content": question})
    try:
        with st.status("Preparing an evidence-grounded answer", expanded=True) as status:
            def progress(stage, detail):
                status.update(label=stage)
                status.write(detail)
            answer = Pipeline(ctx.store, ctx.settings, ctx.service).ask(question, ctx.history, ctx.mode,
                ctx.retrieval, ctx.selected, detail=ctx.detail, progress=progress, **ctx.query_options)
            status.update(label="Answer ready · citations checked", state="complete", expanded=False)
        ctx.store.add_turn(ctx.session_id, question, answer)
        st.rerun()
    except RAGError as exc:
        st.error(str(exc))
    except Exception:
        ctx.store.event("error", {"message": "Unexpected query failure; inspect local configuration."})
        st.error("The query could not be completed. Review the activity log or retry.")


@st.fragment(run_every="2s")
def job_status(jobs, store):
    rows = store.jobs(10)
    revision = store.revision()
    previous = st.session_state.get("displayed_library_revision", revision)
    st.session_state["displayed_library_revision"] = revision
    if previous != revision:
        st.rerun()
    if not rows:
        return
    with st.expander("Ingestion queue", expanded=any(r["status"] in {"queued", "running"} for r in rows)):
        for row in rows:
            with st.container(border=True):
                st.markdown(f"**{row['source']}** · {row['status'].title()}")
                if row["status"] in {"queued", "running"}:
                    st.progress(row["progress"], text="Waiting" if row["status"] == "queued" else "Extracting and chunking")
                    if row["status"] == "queued" and st.button("Cancel queued upload", key="cancel_"+row["id"]):
                        jobs.cancel(row["id"])
                        st.rerun(scope="fragment")
                elif row["status"] == "completed":
                    st.caption(f"{row['payload'].get('chunks', 0)} chunks · {'Duplicate skipped' if row['payload'].get('duplicate') else 'Ready for search'}")
                else:
                    st.caption(row["payload"].get("message", "Job ended."))


def documents_page(ctx):
    st.title("Your knowledge library.")
    st.caption("Ingest, inspect, and organize the documents behind your answers.")
    upload_tab, library_tab = st.tabs(["Library", "Add documents"])
    with library_tab:
        columns = st.columns(4)
        columns[0].metric("Documents", len(ctx.documents))
        columns[1].metric("Total size", f"{sum(d['bytes'] for d in ctx.documents)/1024/1024:.2f} MB")
        columns[2].metric("Extraction warnings", sum(len(d["warnings"]) for d in ctx.documents))
        with ctx.store.connect() as db:
            cached = db.execute("SELECT COUNT(*) FROM embeddings WHERE config=?", (key(ctx.settings.embedding_config()),)).fetchone()[0]
        columns[3].metric("Cached embeddings", f"{cached}/{sum(d['chunks'] for d in ctx.documents)}")
        if st.button("Prepare semantic index", disabled=ctx.service is None or not ctx.selected):
            try:
                with st.spinner("Preparing and caching embeddings for selected documents…"):
                    Retriever(ctx.store, ctx.settings, ctx.service).index(ctx.store.chunks(ctx.selected))
                st.success("Semantic index ready for the selected documents.")
            except RAGError as exc:
                st.error(str(exc))
        query = st.text_input("Filter library", placeholder="Search filenames or formats")
        filtered = [d for d in ctx.documents if query.casefold() in (d["source"]+" "+d["format"]).casefold()]
        if filtered:
            st.dataframe([{"Document": d["source"], "Format": d["format"].upper(), "Size KB": round(d["bytes"]/1024, 1),
                "Pages": d["pages"], "Chunks": d["chunks"], "Warnings": len(d["warnings"])} for d in filtered],
                hide_index=True, width="stretch")
            doc_map = {d["id"]: d for d in filtered}
            selected_doc = st.selectbox("Inspect document", list(doc_map), format_func=lambda i: doc_map[i]["source"])
            doc = doc_map[selected_doc]
            details = ctx.store.document_details(selected_doc)
            preview, chunks, metadata = st.tabs(["Extracted content", "Chunk explorer", "Metadata & provenance"])
            with preview:
                units = details["units"]
                number = st.selectbox("Page or section", list(range(len(units))),
                    format_func=lambda i: f"Page {units[i]['page']}" if units[i].get("page") else units[i]["section"])
                st.text(units[number]["text"][:20000])
                if len(units[number]["text"]) > 20000:
                    st.caption("Preview limited to 20,000 characters. Download the original for the complete content.")
            with chunks:
                corpus = ctx.store.chunks([selected_doc])
                index = st.number_input("Chunk number", 1, len(corpus), 1)-1
                current = corpus[index]
                st.caption(f"Child {current['id'][:16]} · Parent {current['parent_id']}")
                child, parent = st.columns(2)
                child.markdown("**Searchable child**")
                child.text(current["text"])
                parent.markdown("**Surrounding parent**")
                parent.text(current["parent_text"])
            with metadata:
                st.json({**doc, "chunk_configuration": details["config"]})
                current_config = ctx.settings.index_config()
                if current_config != details["config"]:
                    st.info("This document uses different chunk settings. Reindex to apply the current configuration.")
            for warning in doc["warnings"][:10]:
                st.warning(warning)
            actions = st.columns(3)
            filename, raw = ctx.store.raw_document(selected_doc)
            actions[0].download_button("Download original", raw, filename, key="original_"+selected_doc, width="stretch")
            if actions[1].button("Reindex document", disabled=ctx.service is None and ctx.settings.chunk_strategy == "semantic", width="stretch"):
                from rag.library import Library
                try:
                    with st.spinner("Rebuilding chunks using current settings…"):
                        Library(ctx.store, ctx.settings, ctx.service).reindex(selected_doc)
                    st.rerun()
                except RAGError as exc:
                    st.error(str(exc))
            if actions[2].button("Remove document", width="stretch"):
                st.session_state["remove_document_id"] = selected_doc
            if st.session_state.get("remove_document_id") == selected_doc:
                st.warning(f"Remove {doc['source']} from the library? Indexed evidence is removed; existing answer text remains with a source-deletion notice.")
                confirm, cancel = st.columns(2)
                if confirm.button("Remove and clear indexed evidence", type="primary"):
                    ctx.store.delete_document(selected_doc)
                    st.session_state.pop("remove_document_id", None)
                    st.session_state.pop("search_experiment", None)
                    st.rerun()
                if cancel.button("Keep document"):
                    st.session_state.pop("remove_document_id", None)
                    st.rerun()
        else:
            empty_state("A library built around your work.", "Add PDF, Word, spreadsheet or text files. Content stays locally indexed; only selected evidence is sent to Gemini when answering.")
    with upload_tab:
        st.subheader("Add knowledge")
        st.caption("PDF · TXT · Markdown · DOCX · CSV · XLSX · up to 25 MB per file")
        uploads = st.file_uploader("Add documents", type=["pdf", "txt", "md", "docx", "csv", "xlsx"], accept_multiple_files=True)
        if st.button("Index uploads", type="primary", disabled=not uploads):
            accepted = 0
            for upload in uploads:
                try:
                    ctx.jobs.submit(upload.name, upload.getvalue(), ctx.settings)
                    accepted += 1
                except RAGError as exc:
                    st.error(f"{upload.name}: {exc}")
            if accepted:
                st.success(f"{accepted} upload(s) queued. You can continue using the workspace.")
        st.caption("Background extraction uses one worker and a bounded queue. Semantic chunking uses the selected embedding API. Interrupted uploads must be submitted again.")
    job_status(ctx.jobs, ctx.store)


def search_page(ctx):
    st.title("Explore the retrieval layer.")
    st.caption("Compare search strategies and inspect the selected context before generating an answer.")
    with st.form("search_experiment_form"):
        query = st.text_input("Search query", placeholder="e.g. MapReduce scalability and fault tolerance")
        methods = st.multiselect("Compare methods", ["dense", "sparse", "hybrid"], default=["sparse", "hybrid"])
        execute = st.form_submit_button("Run search experiment", type="primary", disabled=not ctx.selected)
    if execute:
        if not query.strip() or not methods:
            st.warning("Enter a query and select at least one method.")
        elif ctx.service is None and any(m != "sparse" for m in methods):
            st.warning("Configure Gemini or choose Offline demo for dense/hybrid search.")
        else:
            results = {}
            for method in methods:
                try:
                    retriever = Retriever(ctx.store, ctx.settings, ctx.service)
                    candidates = retriever.search(query, method, ctx.selected)
                    reranked, warning = rerank(query, candidates, ctx.settings.reranker)
                    diverse, diversity_method = diversify(reranked, ctx.store, ctx.settings)
                    results[method] = {"candidates": candidates, "reranked": reranked, "diversified": diverse,
                        "context": build_context(diverse, ctx.settings), "warnings": retriever.warnings+([warning] if warning else []),
                        "diversity_method": diversity_method}
                except RAGError as exc:
                    results[method] = {"error": str(exc)}
            st.session_state["search_experiment"] = {"query": query, "results": results,
                "revision": ctx.store.revision(), "selected": ctx.selected, "configuration": asdict(ctx.settings)}
    experiment = st.session_state.get("search_experiment")
    if experiment and (experiment["revision"] != ctx.store.revision() or experiment["selected"] != ctx.selected
                       or experiment["configuration"] != asdict(ctx.settings)):
        st.info("The library, selection or settings changed. Run the experiment again for current results.")
        return
    if experiment:
        st.subheader(experiment["query"])
        tabs = st.tabs([name.title() for name in experiment["results"]])
        for tab, (_method, result) in zip(tabs, experiment["results"].items(), strict=True):
            with tab:
                if "error" in result:
                    st.error(result["error"])
                    continue
                for warning in result["warnings"]:
                    st.warning(warning)
                st.caption(f"{len(result['candidates'])} candidates · {len(result['context'])} selected passages · {result['diversity_method']}")
                st.dataframe(ranking_table(result["diversified"]), width="stretch", hide_index=True)
                with st.expander("Inspect selected context"):
                    evidence_view(result["context"], query=experiment["query"])
        st.download_button("Export experiment", json.dumps(experiment, ensure_ascii=False, indent=2),
                           "atlas-search-experiment.json", "application/json")


def evaluation_page(ctx):
    st.title("Measure. Compare. Improve.")
    st.caption("Reproducible retrieval evaluation with labeled sources, saved configurations, and run history.")
    run_tab, history_tab = st.tabs(["Run evaluation", "Previous runs"])
    with run_tab:
        with st.container(border=True):
            st.subheader("Evaluation configuration")
            st.caption("Use the five-question example dataset or upload a JSON array containing question, reference_answer and relevant_source_ids.")
            evaluation_file = st.file_uploader("Evaluation dataset (optional)", type=["json"], key="eval_upload")
            columns = st.columns(2)
            k = columns[0].number_input("Evaluation K", 1, 100, min(ctx.settings.top_k, max(1, len(ctx.documents))))
            title = columns[1].text_input("Run label", "Retrieval baseline")
            generation = st.checkbox("Evaluate generated answers (additional API calls in Gemini)", False)
            judge = st.checkbox("Include model-judged relevance, faithfulness and completeness", False,
                                disabled=ctx.settings.backend == "demo" or not generation)
            st.caption("Word overlap is a lexical proxy. Model judgments can be wrong. Local retrieval latency is measured with warm embedding caches.")
            if st.button("Run comparison", type="primary", disabled=ctx.service is None or not ctx.documents):
                try:
                    if evaluation_file:
                        if evaluation_file.size > 1024*1024:
                            raise RAGError("Evaluation dataset exceeds 1 MB.")
                        with tempfile.TemporaryDirectory(dir=ctx.store.root) as folder:
                            path = Path(folder)/"evaluation.json"
                            path.write_bytes(evaluation_file.getvalue())
                            dataset = load_dataset(path)
                    else:
                        dataset = load_dataset(ROOT/"examples/evaluation.json")
                    progress = st.progress(0.0)
                    with st.spinner("Scoring actual retrieved results…"):
                        result = compare(ctx.store, ctx.settings, ctx.service, dataset, k=int(k), generation=generation,
                            model_judge=judge and ctx.settings.backend != "demo" and generation, progress=progress.progress)
                    with ctx.store.connect() as db:
                        db.execute("UPDATE evaluations SET title=? WHERE id=?", (title[:100], result["run_id"]))
                    st.session_state["eval_result"] = result
                except (RAGError, ValueError) as exc:
                    st.error(str(exc))
        result = st.session_state.get("eval_result")
        if result:
            if result.get("library_revision") != ctx.store.revision():
                st.info("This evaluation was recorded before the latest library change. Run again for current scores.")
            render_evaluation(result, "current")
    with history_tab:
        runs = ctx.store.evaluation_runs()
        if not runs:
            empty_state("Build a record of improvement.", "Each evaluation stores its settings, dataset fingerprint, library revision and measured results.")
        else:
            by_id = {run["id"]: run for run in runs}
            selected = st.multiselect("Select runs to compare", list(by_id), default=[runs[0]["id"]],
                format_func=lambda i: f"{by_id[i]['title']} · {by_id[i]['created'][:16]}")
            comparison = []
            for identifier in selected:
                run = by_id[identifier]
                for row in run["result"]["summary"]:
                    comparison.append({"Run": run["title"], "Run ID": identifier[:8], "Backend": run["result"]["backend"],
                        "K": run["result"]["k"], **row})
            if comparison:
                st.dataframe(comparison, width="stretch", hide_index=True)
                st.caption("Only compare quality scores directly when dataset fingerprints, label granularity, K and library revisions match.")
            for identifier in selected:
                with st.expander("Run details · "+by_id[identifier]["title"]):
                    render_evaluation({**by_id[identifier]["result"], "run_id": identifier}, "history")


def render_evaluation(result, key_prefix):
    st.caption(f"{result['backend'].title()} · {result['questions'] if 'questions' in result else len(result['details'])//4} questions · K={result['k']} · {result['granularity']} labels")
    table = pd.DataFrame(result["summary"]).set_index("variant")
    st.dataframe(table, width="stretch")
    st.bar_chart(table[["precision@k", "recall@k", "mrr", "ndcg@k"]])
    with st.expander("Per-question scores & reproducibility"):
        st.dataframe(result["details"], width="stretch", hide_index=True)
        st.json({key: value for key, value in result.items() if key not in {"summary", "details"}})
    st.download_button("Download measured results", json.dumps(result, indent=2), "evaluation-results.json", "application/json",
                       key=key_prefix+"_evaluation_export_"+result.get("run_id", "latest"))


def activity_page(ctx):
    st.title("A clear view of your system.")
    st.caption("Latency, API health, ingestion history, and answer feedback — based on recorded activity.")
    events = ctx.store.events(1000)
    queries = [e for e in events if e["kind"] == "query"]
    errors = [e for e in events if e["kind"] == "api_error"]
    columns = st.columns(4)
    columns[0].metric("Queries recorded", len(queries))
    columns[1].metric("Median response", f"{np.median([q['total_ms'] for q in queries])/1000:.2f}s" if queries else "—")
    columns[2].metric("95th percentile", f"{np.percentile([q['total_ms'] for q in queries], 95)/1000:.2f}s" if queries else "—")
    columns[3].metric("API failed attempts", len(errors))
    metrics, jobs, feedback = st.tabs(["Query performance", "Ingestion & events", "Saved answers & feedback"])
    with metrics:
        if queries:
            st.line_chart(pd.DataFrame(list(reversed(queries)))[["retrieval_ms", "generation_ms", "total_ms"]])
            st.dataframe([{k: q.get(k) for k in ("created", "mode", "backend", "sources", "cited_documents", "tokens", "total_ms", "trace_id")}
                          for q in queries], width="stretch", hide_index=True)
            st.caption("Last 1000 events. Tokens use API generation metadata; context estimates are character-based. No confidence score is inferred from latency or ranking.")
        else:
            empty_state("Performance begins with a real question.", "Query latency, evidence usage and API events appear after you use the workspace.")
    with jobs:
        st.dataframe([{k: j[k] for k in ("source", "status", "created", "updated")} for j in ctx.store.jobs()], width="stretch", hide_index=True)
        event_filter = st.selectbox("Event type", ["All", "error", "api_error", "ingestion", "ingestion_error", "evaluation"])
        selected = [e for e in events if event_filter == "All" or e["kind"] == event_filter]
        st.json(selected[:50])
    with feedback:
        ratings = ctx.store.feedback()
        positives = sum(r["rating"] == 1 for r in ratings.values())
        negatives = sum(r["rating"] == -1 for r in ratings.values())
        st.caption(f"{positives} helpful · {negatives} need review · {sum(r['saved'] for r in ratings.values())} saved")
        for session in ctx.store.sessions():
            for message in ctx.store.history(session["id"]):
                rating = ratings.get(message["message_id"], {})
                if rating.get("saved"):
                    st.caption(session["title"])
                    message_view(message, False)


def settings_page(ctx):
    st.title("Tune your workspace.")
    st.caption("Saved locally. Changes to chunking require a document reindex; embedding settings create a compatible index automatically.")
    base = ctx.settings
    with st.form("workspace_settings"):
        retrieval, chunking, generation = st.tabs(["Retrieval quality", "Document processing", "Models & limits"])
        with retrieval:
            top_k = st.number_input("Evidence passages", 1, 100, base.top_k)
            candidates = st.number_input("Retrieval candidates", 1, 100, base.candidate_k)
            reranker = st.selectbox("Reranker", ["lexical", "none", "cross-encoder"], index=["lexical", "none", "cross-encoder"].index(base.reranker))
            dense_weight = st.slider("Hybrid semantic weight", 0.0, 1.0, float(base.dense_weight), 0.05)
            diversity = st.checkbox("Reduce repeated evidence with MMR", base.diversity)
            balance = st.slider("MMR relevance balance", 0.0, 1.0, float(base.mmr_lambda), 0.05)
            cap = st.number_input("Maximum chunks per document (0 = unlimited)", 0, 100, base.max_chunks_per_document)
            st.caption("MMR balances original relevance with cosine redundancy. It reuses cached embeddings, or uses local TF-IDF for keyword-only results.")
            left, right = st.columns(2)
            bm25_k1 = left.number_input("BM25 k1", 0.1, 3.0, float(base.bm25_k1), 0.1)
            bm25_b = right.number_input("BM25 b", 0.0, 1.0, float(base.bm25_b), 0.05)
        with chunking:
            strategies = ["recursive", "fixed", "sentence", "parent-child", "semantic"]
            strategy = st.selectbox("Chunking strategy", strategies, index=strategies.index(base.chunk_strategy))
            size = st.number_input("Child chunk size", 100, 8000, base.chunk_size, 100)
            overlap = st.number_input("Chunk overlap", 0, 7900, base.chunk_overlap, 10)
            parent = st.number_input("Parent size", 100, 16000, base.parent_size, 100)
            ocr = st.checkbox("Optional scanned PDF OCR", base.ocr)
            st.caption("Semantic chunking makes extra embedding calls. OCR requires Tesseract and optional Python packages.")
        with generation:
            model = st.text_input("Generation model", base.generation_model)
            embedding_model = st.text_input("Embedding model", base.embedding_model)
            dimension = st.number_input("Embedding dimensions", 1, 3072, base.dimension)
            context = st.number_input("Context character budget", 100, 100000, base.context_chars, 100)
            steps = st.number_input("Agent tool budget", 1, 5, base.agent_steps)
            st.caption("API keys stay in .env or environment variables. They are never stored in exported settings or traces.")
        submitted = st.form_submit_button("Save configuration", type="primary")
    if submitted:
        try:
            updated = replace(base, top_k=int(top_k), candidate_k=int(candidates), reranker=reranker,
                dense_weight=dense_weight, diversity=diversity, mmr_lambda=balance, max_chunks_per_document=int(cap),
                bm25_k1=bm25_k1, bm25_b=bm25_b, chunk_strategy=strategy, chunk_size=int(size), chunk_overlap=int(overlap),
                parent_size=int(parent), ocr=ocr, generation_model=model.strip(), embedding_model=embedding_model.strip(),
                dimension=int(dimension), context_chars=int(context), agent_steps=int(steps))
            ctx.store.save_preference("settings", asdict(updated))
            st.session_state["configuration_saved"] = True
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    st.download_button("Export configuration", json.dumps(asdict(base), indent=2), "atlas-configuration.json", "application/json")
    if st.session_state.pop("configuration_saved", False):
        st.success("Configuration saved. New questions use these settings.")
    with st.expander("Local storage & operating limits"):
        st.code(str(ctx.store.root), language=None)
        st.write("Designed for one local user. Gemini receives selected evidence. Authentication, tenant isolation and encrypted storage are required before public deployment.")
