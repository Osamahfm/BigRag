"""SQLite is authoritative. No pickle, no filenames used as filesystem paths."""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from rag.errors import RAGError


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "library.sqlite3"
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents(
                    id TEXT PRIMARY KEY, source TEXT NOT NULL, raw BLOB NOT NULL,
                    metadata TEXT NOT NULL, units TEXT NOT NULL, config TEXT NOT NULL,
                    warnings TEXT NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks(
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL, tokens TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS embeddings(
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    config TEXT NOT NULL, dimension INTEGER NOT NULL, vector BLOB NOT NULL,
                    PRIMARY KEY(chunk_id, config));
                CREATE TABLE IF NOT EXISTS query_cache(
                    key TEXT PRIMARY KEY, dimension INTEGER NOT NULL, vector BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value INTEGER NOT NULL);
                INSERT OR IGNORE INTO state VALUES('revision',0);
                CREATE TABLE IF NOT EXISTS sessions(
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS messages(
                    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS preferences(
                    scope TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(
                    id TEXT PRIMARY KEY, source TEXT NOT NULL, status TEXT NOT NULL,
                    progress REAL NOT NULL, payload TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluations(
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, payload TEXT NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS feedback(
                    message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
                    rating INTEGER NOT NULL, note TEXT NOT NULL, saved INTEGER NOT NULL DEFAULT 0, updated TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS embeddings_config ON embeddings(config);
                CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id,id);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def revision(self):
        with self.connect() as db:
            return db.execute("SELECT value FROM state WHERE key='revision'").fetchone()[0]

    def documents(self):
        with self.connect() as db:
            rows = db.execute("SELECT id,metadata,warnings,created FROM documents ORDER BY created,id").fetchall()
        return [{**json.loads(r["metadata"]), "id": r["id"], "warnings": json.loads(r["warnings"]),
                 "created": r["created"]} for r in rows]

    def chunks(self, document_ids=None):
        with self.connect() as db:
            if document_ids is None:
                rows = db.execute("SELECT payload FROM chunks ORDER BY document_id,id").fetchall()
            elif not document_ids:
                return []
            else:
                placeholders = ",".join("?" for _ in document_ids)
                rows = db.execute(f"SELECT payload FROM chunks WHERE document_id IN ({placeholders}) ORDER BY document_id,id",
                                  tuple(document_ids)).fetchall()
        chunks = [json.loads(r[0]) for r in rows]
        def order(chunk):
            parts = chunk.get("parent_id", "").split(":")
            unit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            parent = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            offset = chunk.get("ordinal", max(0, chunk.get("parent_text", "").find(chunk["text"])))
            return chunk["document_id"], unit, parent, offset
        chunks.sort(key=order)
        return chunks

    def put_document(self, metadata, raw, units, chunks, config, warnings):
        from rag.text import tokens
        doc_id = metadata["hash"]
        with self.connect() as db:
            db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                       "source=excluded.source,raw=excluded.raw,metadata=excluded.metadata,units=excluded.units,"
                       "config=excluded.config,warnings=excluded.warnings",
                (doc_id, metadata["source"], raw, json.dumps(metadata), json.dumps(units),
                 json.dumps(config, sort_keys=True), json.dumps(warnings), now()))
            keep_ids = {c["id"] for c in chunks}
            old_ids = [r[0] for r in db.execute("SELECT id FROM chunks WHERE document_id=?", (doc_id,))]
            db.executemany("DELETE FROM chunks WHERE id=?", [(i,) for i in old_ids if i not in keep_ids])
            db.executemany("INSERT INTO chunks VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                           "payload=excluded.payload,tokens=excluded.tokens",
                [(c["id"], doc_id, json.dumps(c, ensure_ascii=False), json.dumps(tokens(c["text"]))) for c in chunks])
            db.execute("UPDATE state SET value=value+1 WHERE key='revision'")
            db.execute("DELETE FROM query_cache")
        self.invalidate_indexes()

    def raw_document(self, doc_id):
        with self.connect() as db:
            row = db.execute("SELECT source,raw FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            raise RAGError("Document no longer exists.")
        return row["source"], bytes(row["raw"])

    def delete_document(self, doc_id):
        with self.connect() as db:
            db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            db.execute("DELETE FROM query_cache")
            db.execute("UPDATE state SET value=value+1 WHERE key='revision'")
            # Remove copied source passages from persisted chats while retaining the conversation.
            for row in db.execute("SELECT id,payload FROM messages").fetchall():
                message = json.loads(row["payload"])
                affected = any(c.get("document_id") == doc_id for c in message.get("sources", []))
                debug_affected = any(c.get("document_id") == doc_id
                    for field in ("candidates", "reranked", "diversified", "context")
                    for c in message.get("debug", {}).get(field, []))
                if debug_affected:
                    message.pop("debug", None)
                if affected:
                    message["sources"] = [c for c in message["sources"] if c.get("document_id") != doc_id]
                    message["cited_sources"] = [c for c in message.get("cited_sources", []) if c.get("document_id") != doc_id]
                    message.pop("debug", None)
                    message["source_notice"] = "Some original evidence was deleted from the library."
                if affected or debug_affected:
                    db.execute("UPDATE messages SET payload=? WHERE id=?", (json.dumps(message), row["id"]))
        self.invalidate_indexes()

    def invalidate_indexes(self):
        folder = self.root / "indexes"
        if folder.exists():
            for path in folder.iterdir():
                if path.is_file() and path.suffix in {".faiss", ".json", ".tmp"}:
                    path.unlink(missing_ok=True)

    def sessions(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM sessions ORDER BY created DESC")]

    def new_session(self, title="New conversation"):
        session_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO sessions VALUES(?,?,?)", (session_id, title[:100], now()))
        return session_id

    def rename_session(self, session_id, title):
        with self.connect() as db:
            db.execute("UPDATE sessions SET title=? WHERE id=?", (title.strip()[:100] or "Conversation", session_id))

    def history(self, session_id):
        with self.connect() as db:
            return [{**json.loads(r["payload"]), "message_id": r["id"]}
                    for r in db.execute("SELECT id,payload FROM messages WHERE session_id=? ORDER BY id", (session_id,))]

    def add_turn(self, session_id, question, answer):
        with self.connect() as db:
            db.executemany("INSERT INTO messages(session_id,payload,created) VALUES(?,?,?)",
                [(session_id, json.dumps(m, ensure_ascii=False), now()) for m in
                 [{"role": "user", "content": question}, answer]])
            count = db.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)).fetchone()[0]
            if count == 2:
                db.execute("UPDATE sessions SET title=? WHERE id=?", (question[:70], session_id))

    def clear_session(self, session_id):
        with self.connect() as db:
            db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))

    def event(self, kind, payload):
        with self.connect() as db:
            db.execute("INSERT INTO events(kind,payload,created) VALUES(?,?,?)", (kind, json.dumps(payload), now()))

    def events(self, limit=100):
        with self.connect() as db:
            return [{"kind": r["kind"], "created": r["created"], **json.loads(r["payload"])}
                    for r in db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))]

    def preference(self, scope, default=None):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM preferences WHERE scope=?", (scope,)).fetchone()
        return json.loads(row[0]) if row else ({} if default is None else default)

    def save_preference(self, scope, payload):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO preferences VALUES(?,?)", (scope, json.dumps(payload)))

    def document_details(self, doc_id):
        with self.connect() as db:
            row = db.execute("SELECT units,config FROM documents WHERE id=?", (doc_id,)).fetchone()
        if row is None:
            raise RAGError("Document no longer exists.")
        return {"units": json.loads(row[0]), "config": json.loads(row[1])}

    def new_job(self, source, payload):
        identifier = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)",
                       (identifier, source, "queued", 0, json.dumps(payload), now(), now()))
        return identifier

    def update_job(self, identifier, status, progress, payload=None):
        if status not in {"queued", "running", "completed", "failed", "cancelled", "interrupted"} or not 0 <= progress <= 1:
            raise ValueError("Invalid job state.")
        with self.connect() as db:
            if payload is None:
                db.execute("UPDATE jobs SET status=?,progress=?,updated=? WHERE id=?", (status, progress, now(), identifier))
            else:
                db.execute("UPDATE jobs SET status=?,progress=?,payload=?,updated=? WHERE id=?",
                           (status, progress, json.dumps(payload), now(), identifier))

    def jobs(self, limit=50):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def save_evaluation(self, title, result):
        identifier = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO evaluations VALUES(?,?,?,?)", (identifier, title[:100], json.dumps(result), now()))
        return identifier

    def evaluation_runs(self):
        with self.connect() as db:
            return [{"id": r["id"], "title": r["title"], "created": r["created"], "result": json.loads(r["payload"])}
                    for r in db.execute("SELECT * FROM evaluations ORDER BY created DESC LIMIT 50")]

    def set_feedback(self, message_id, rating=0, note="", saved=False):
        if rating not in {-1, 0, 1}:
            raise ValueError("Feedback must be -1, 0 or 1.")
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO feedback VALUES(?,?,?,?,?)",
                       (message_id, rating, note[:2000], int(saved), now()))

    def feedback(self):
        with self.connect() as db:
            return {r["message_id"]: dict(r) for r in db.execute("SELECT * FROM feedback")}
