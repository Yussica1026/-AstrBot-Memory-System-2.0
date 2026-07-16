from __future__ import annotations

import math
import os
import sqlite3
import struct
from datetime import datetime, timedelta
from typing import Any

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(PLUGIN_DIR, "memory_manager.db")

MEMORY_CATEGORIES = {"happy", "daily", "sad", "important", "fight", "milestone"}
DECAY_LAMBDA = 0.05
ARCHIVE_THRESHOLD = 1.0


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_dt(value: str | None) -> datetime:
    if not value:
        return datetime.now()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now()


def clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def normalize_category(category: str | None) -> str:
    value = (category or "daily").strip().lower()
    return value if value in MEMORY_CATEGORIES else "daily"


def bigrams(text: str) -> set[str]:
    compact = "".join(str(text).lower().split())
    if not compact:
        return set()
    if len(compact) == 1:
        return {compact}
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def jaccard(a: str, b: str) -> float:
    left = bigrams(a)
    right = bigrams(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def pack_embedding(vec: list[float] | None) -> bytes | None:
    if not vec:
        return None
    return struct.pack(f"{len(vec)}f", *[float(x) for x in vec])


def unpack_embedding(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    if len(blob) % 4 != 0:
        return None
    size = len(blob) // 4
    return list(struct.unpack(f"{size}f", blob))


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def split_tags(tags: str | None) -> list[str]:
    if not tags:
        return []
    return [x.strip() for x in tags.split(",") if x.strip()]


def merge_tags(left: str | None, right: str | None) -> str:
    seen: list[str] = []
    for tag in split_tags(left) + split_tags(right):
        if tag not in seen:
            seen.append(tag)
    return ",".join(seen)


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT DEFAULT '',
                    valence REAL DEFAULT 0,
                    arousal REAL DEFAULT 0.2,
                    importance INTEGER DEFAULT 5,
                    forgetting_score REAL DEFAULT 0,
                    decay_score REAL DEFAULT 5,
                    status TEXT DEFAULT 'active',
                    layer TEXT DEFAULT 'event',
                    activation_count INTEGER DEFAULT 1,
                    last_activated TEXT,
                    resolved INTEGER DEFAULT 0,
                    embedding BLOB,
                    kb_doc_id TEXT,
                    related_ids TEXT DEFAULT '',
                    fact_status TEXT DEFAULT 'current',
                    superseded_by INTEGER,
                    aliases TEXT DEFAULT '',
                    source TEXT DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_id INTEGER NOT NULL,
                    to_id INTEGER NOT NULL,
                    link_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    note TEXT DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS commitments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    type TEXT NOT NULL,
                    who TEXT DEFAULT '',
                    content TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    due_date TEXT,
                    related_memory_id INTEGER
                )
                """
            )
            self.migrate(conn)

    def migrate(self, conn: sqlite3.Connection) -> None:
        expected = {
            "aliases": "TEXT DEFAULT ''",
            "source": "TEXT DEFAULT ''",
            "fact_status": "TEXT DEFAULT 'current'",
            "superseded_by": "INTEGER",
            "related_ids": "TEXT DEFAULT ''",
            "kb_doc_id": "TEXT",
        }
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(memories)")}
        for column, ddl in expected.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE memories ADD COLUMN {column} {ddl}")

    def calculate_decay(self, row: sqlite3.Row, when: datetime | None = None) -> float:
        if row["layer"] == "core":
            return 9999.0
        current = when or datetime.now()
        last = parse_dt(row["last_activated"] or row["created_at"])
        days = max(0.0, (current - last).total_seconds() / 86400)
        importance = max(1, int(row["importance"] or 1))
        activations = max(1, int(row["activation_count"] or 1))
        arousal = clamp_float(row["arousal"], 0, 1, 0.2)
        return importance * math.sqrt(activations) * math.exp(-DECAY_LAMBDA * days) * (0.7 + arousal * 0.3)

    def refresh_decay(self) -> None:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM memories WHERE fact_status = 'current'").fetchall()
            for row in rows:
                score = self.calculate_decay(row)
                layer = row["layer"]
                status = row["status"]
                if layer == "event" and score < ARCHIVE_THRESHOLD:
                    layer = "archive"
                    status = "archived"
                conn.execute(
                    "UPDATE memories SET decay_score = ?, layer = ?, status = ? WHERE id = ?",
                    (score, layer, status, row["id"]),
                )

    def find_duplicate(self, conn: sqlite3.Connection, category: str, content: str) -> sqlite3.Row | None:
        since = (datetime.now() - timedelta(hours=24)).isoformat(timespec="seconds")
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE category = ? AND created_at >= ? AND fact_status = 'current'
            ORDER BY id DESC
            """,
            (category, since),
        ).fetchall()
        for row in rows:
            old = row["content"]
            if content in old or old in content or jaccard(content, old) >= 0.7:
                return row
        return None

    def related_ids_for(
        self,
        conn: sqlite3.Connection,
        content: str,
        exclude_id: int | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        rows = conn.execute(
            "SELECT id, content, aliases, embedding FROM memories WHERE fact_status = 'current' AND layer != 'archive'"
        ).fetchall()
        scored: list[tuple[float, int]] = []
        for row in rows:
            if exclude_id is not None and int(row["id"]) == exclude_id:
                continue
            text = f"{row['content']} {row['aliases'] or ''}"
            score = 0.0
            if embedding is not None:
                score = cosine_similarity(embedding, unpack_embedding(row["embedding"]))
            if score <= 0:
                score = jaccard(content, text)
            if score > 0:
                scored.append((score, int(row["id"])))
        scored.sort(reverse=True)
        return ",".join(str(item[1]) for item in scored[:3])

    def save_memory(
        self,
        category: str,
        content: str,
        tags: str = "",
        valence: float = 0,
        arousal: float = 0.2,
        importance: int = 5,
        source: str = "",
        embedding: list[float] | None = None,
    ) -> tuple[int, bool]:
        category = normalize_category(category)
        content = content.strip()
        if not content:
            raise ValueError("记忆内容不能为空。")
        valence = clamp_float(valence, -1, 1, 0)
        arousal = clamp_float(arousal, 0, 1, 0.2)
        importance = clamp_int(importance, 1, 10, 5)

        with self.connect() as conn:
            duplicate = self.find_duplicate(conn, category, content)
            if duplicate:
                merged_content = duplicate["content"]
                if content not in merged_content:
                    merged_content = f"{merged_content}\n{content}"
                merged_tags = merge_tags(duplicate["tags"], tags)
                merged_importance = max(int(duplicate["importance"] or 1), importance)
                related = self.related_ids_for(conn, merged_content, int(duplicate["id"]), embedding)
                decay = self.calculate_decay(duplicate)
                conn.execute(
                    """
                    UPDATE memories
                    SET content = ?, tags = ?, importance = ?, valence = ?, arousal = ?,
                        related_ids = ?, decay_score = ?, last_activated = ?, activation_count = activation_count + 1,
                        embedding = COALESCE(?, embedding)
                    WHERE id = ?
                    """,
                    (
                        merged_content,
                        merged_tags,
                        merged_importance,
                        valence,
                        arousal,
                        related,
                        decay,
                        now_iso(),
                        pack_embedding(embedding),
                        duplicate["id"],
                    ),
                )
                return int(duplicate["id"]), True

            created = now_iso()
            cursor = conn.execute(
                """
                INSERT INTO memories (
                    created_at, category, content, tags, valence, arousal, importance,
                    forgetting_score, decay_score, status, layer, activation_count,
                    last_activated, resolved, embedding, related_ids, fact_status, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 'event', 1, ?, 0, ?, '', 'current', ?)
                """,
                (
                    created,
                    category,
                    content,
                    tags,
                    valence,
                    arousal,
                    importance,
                    importance,
                    importance,
                    created,
                    pack_embedding(embedding),
                    source,
                ),
            )
            memory_id = int(cursor.lastrowid)
            related = self.related_ids_for(conn, content, memory_id, embedding)
            conn.execute("UPDATE memories SET related_ids = ? WHERE id = ?", (related, memory_id))
            return memory_id, False

    def activate(self, conn: sqlite3.Connection, memory_id: int) -> None:
        conn.execute(
            """
            UPDATE memories
            SET activation_count = activation_count + 1, last_activated = ?
            WHERE id = ?
            """,
            (now_iso(), memory_id),
        )

    def query(
        self,
        keyword: str = "",
        category: str = "",
        limit: int = 5,
        include_archive: bool = False,
        query_embedding: list[float] | None = None,
    ) -> list[sqlite3.Row]:
        self.refresh_decay()
        limit = clamp_int(limit, 1, 20, 5)
        keyword = (keyword or "").strip()
        category = (category or "").strip().lower()
        params: list[Any] = []
        clauses = ["fact_status = 'current'"]
        if category in MEMORY_CATEGORIES:
            clauses.append("category = ?")
            params.append(category)
        if not include_archive:
            clauses.append("layer != 'archive'")
        sql = f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT 200"

        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            if keyword or query_embedding is not None:
                scored: list[tuple[float, sqlite3.Row]] = []
                for row in rows:
                    text = f"{row['content']} {row['tags'] or ''} {row['aliases'] or ''}"
                    vector_score = cosine_similarity(query_embedding, unpack_embedding(row["embedding"]))
                    text_score = 1.0 if keyword and keyword in text else jaccard(keyword, text)
                    score = max(vector_score, text_score)
                    threshold = 0.3 if vector_score >= text_score else 0.15
                    if score >= threshold:
                        scored.append((score, row))
                scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
                rows = [item[1] for item in scored[:limit]]
            else:
                rows = rows[:limit]
            for row in rows:
                self.activate(conn, int(row["id"]))
            return rows

    def surface(self, limit: int = 3) -> list[sqlite3.Row]:
        self.refresh_decay()
        limit = clamp_int(limit, 1, 10, 3)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE fact_status = 'current' AND layer != 'archive'"
            ).fetchall()
            scored: list[tuple[float, sqlite3.Row]] = []
            for row in rows:
                base = abs(float(row["valence"] or 0)) + float(row["arousal"] or 0) + int(row["importance"] or 1) / 10
                base += float(row["decay_score"] or 0)
                if int(row["resolved"] or 0) == 0:
                    base *= 1.5
                if datetime.now() - parse_dt(row["last_activated"]) <= timedelta(hours=72):
                    base *= 1.2
                if row["layer"] == "core":
                    base += 10
                scored.append((base, row))
            scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
            selected = [item[1] for item in scored[:limit]]
            for row in selected:
                self.activate(conn, int(row["id"]))
            return selected

    def mark_core(self, memory_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "UPDATE memories SET layer = 'core', status = 'active', decay_score = 9999 WHERE id = ?",
                (memory_id,),
            )
            return cur.rowcount > 0

    def resolve(self, memory_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute("UPDATE memories SET resolved = 1 WHERE id = ?", (memory_id,))
            return cur.rowcount > 0

    def supersede(self, old_id: int, new_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "UPDATE memories SET fact_status = 'superseded', superseded_by = ? WHERE id = ?",
                (new_id, old_id),
            )
            return cur.rowcount > 0

    def link(self, from_id: int, to_id: int, link_type: str = "related", note: str = "") -> int:
        link_type = link_type.strip() or "related"
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO memory_links (from_id, to_id, link_type, created_at, note) VALUES (?, ?, ?, ?, ?)",
                (from_id, to_id, link_type, now_iso(), note),
            )
            if link_type == "related":
                conn.execute(
                    "INSERT INTO memory_links (from_id, to_id, link_type, created_at, note) VALUES (?, ?, ?, ?, ?)",
                    (to_id, from_id, link_type, now_iso(), note),
                )
            return int(cur.lastrowid)

    def links_for(self, memory_id: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM memory_links WHERE from_id = ? OR to_id = ? ORDER BY id DESC",
                (memory_id, memory_id),
            ).fetchall()

    def get_memory(self, memory_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()

    def list_memories_for_embedding(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT id, content FROM memories WHERE fact_status = 'current' AND embedding IS NULL ORDER BY id"
            ).fetchall()

    def update_embedding(self, memory_id: int, embedding: list[float]) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE memories SET embedding = ? WHERE id = ?", (pack_embedding(embedding), memory_id))

    def stats(self) -> dict[str, int]:
        self.refresh_decay()
        with self.connect() as conn:
            rows = conn.execute("SELECT layer, COUNT(*) AS count FROM memories GROUP BY layer").fetchall()
            return {row["layer"]: int(row["count"]) for row in rows}

    def save_commitment(self, kind: str, who: str, content: str, due_date: str = "", related_memory_id: int | None = None) -> int:
        kind = kind if kind in {"promise", "wish", "pact"} else "promise"
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO commitments (created_at, type, who, content, status, due_date, related_memory_id)
                VALUES (?, ?, ?, ?, 'active', ?, ?)
                """,
                (now_iso(), kind, who, content, due_date or None, related_memory_id),
            )
            return int(cur.lastrowid)

    def query_commitments(self, status: str = "active", limit: int = 10) -> list[sqlite3.Row]:
        limit = clamp_int(limit, 1, 30, 10)
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM commitments WHERE status = ? ORDER BY id DESC LIMIT ?",
                (status or "active", limit),
            ).fetchall()

    def fulfill_commitment(self, commitment_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute("UPDATE commitments SET status = 'fulfilled' WHERE id = ?", (commitment_id,))
            return cur.rowcount > 0


def format_memory(row: sqlite3.Row) -> str:
    return (
        f"#{row['id']} [{row['category']}/{row['layer']}] "
        f"重要度{row['importance']} 衰减{float(row['decay_score'] or 0):.2f}\n{row['content']}"
    )


def format_memory_list(rows: list[sqlite3.Row], empty: str = "没有找到记忆。") -> str:
    if not rows:
        return empty
    return "\n\n".join(format_memory(row) for row in rows)


@register("astrbot_plugin_memory_manager", "沈砚清", "AstrBot 记忆系统 2.0", "2.0.0")
class AstrBotMemorySystem(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.store = MemoryStore(DB_FILE)

    def get_embedding_provider(self):
        getter = getattr(self.context, "get_all_embedding_providers", None)
        if not getter:
            return None
        try:
            providers = getter() or []
        except Exception:
            return None
        return providers[0] if providers else None

    async def embed_text(self, text: str) -> list[float] | None:
        provider = self.get_embedding_provider()
        if not provider:
            return None
        try:
            vector = await provider.get_embedding(text)
        except Exception:
            return None
        if not isinstance(vector, list) or not vector:
            return None
        try:
            return [float(x) for x in vector]
        except (TypeError, ValueError):
            return None

    @filter.command_group("memory")
    def memory(self):
        pass

    @memory.command("save")
    async def cmd_memory_save(self, event: AstrMessageEvent, category: str, content: str):
        """保存记忆"""
        embedding = await self.embed_text(content)
        memory_id, merged = self.store.save_memory(category, content, source="qq", embedding=embedding)
        action = "合并到已有记忆" if merged else "已保存记忆"
        yield event.plain_result(f"{action} #{memory_id}。")

    @memory.command("query")
    async def cmd_memory_query(self, event: AstrMessageEvent, category: str = "daily", count: int = 5):
        """按分类查询记忆"""
        rows = self.store.query(category=category, limit=count)
        yield event.plain_result(format_memory_list(rows))

    @memory.command("search")
    async def cmd_memory_search(self, event: AstrMessageEvent, keyword: str):
        """关键词搜索记忆"""
        embedding = await self.embed_text(keyword)
        rows = self.store.query(keyword=keyword, limit=5, include_archive=True, query_embedding=embedding)
        yield event.plain_result(format_memory_list(rows))

    @memory.command("semantic")
    async def cmd_memory_semantic(self, event: AstrMessageEvent, query: str):
        """语义搜索记忆"""
        embedding = await self.embed_text(query)
        rows = self.store.query(keyword=query, limit=5, include_archive=True, query_embedding=embedding)
        yield event.plain_result(format_memory_list(rows))

    @memory.command("today")
    async def cmd_memory_today(self, event: AstrMessageEvent):
        """查看今日记忆"""
        today = datetime.now().date().isoformat()
        with self.store.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE substr(created_at, 1, 10) = ? ORDER BY id DESC LIMIT 20",
                (today,),
            ).fetchall()
        yield event.plain_result(format_memory_list(rows, "今天还没有记忆。"))

    @memory.command("count")
    async def cmd_memory_count(self, event: AstrMessageEvent):
        """统计记忆数量"""
        stats = self.store.stats()
        lines = ["记忆统计："]
        for layer in ("core", "event", "archive"):
            lines.append(f"- {layer}: {stats.get(layer, 0)}")
        yield event.plain_result("\n".join(lines))

    @memory.command("surface")
    async def cmd_memory_surface(self, event: AstrMessageEvent):
        """主动浮现记忆"""
        rows = self.store.surface(limit=3)
        yield event.plain_result(format_memory_list(rows, "暂时没有可浮现的记忆。"))

    @memory.command("reindex")
    async def cmd_memory_reindex(self, event: AstrMessageEvent):
        """重新计算衰减和关联"""
        self.store.refresh_decay()
        provider = self.get_embedding_provider()
        embedded = 0
        if provider:
            for row in self.store.list_memories_for_embedding():
                vector = await self.embed_text(row["content"])
                if vector:
                    self.store.update_embedding(int(row["id"]), vector)
                    embedded += 1
        suffix = f"，补算向量 {embedded} 条。" if provider else "。未检测到向量模型，保留文本相似度 fallback。"
        yield event.plain_result(f"已重新计算记忆衰减状态{suffix}")

    @filter.llm_tool(name="memory_save")
    async def memory_save(
        self,
        event: AstrMessageEvent,
        category: str,
        content: str,
        tags: str = "",
        valence: float = 0,
        arousal: float = 0.2,
        importance: int = 5,
    ):
        """保存一条长期记忆。

        Args:
            category(string): 记忆分类，happy/daily/sad/important/fight/milestone
            content(string): 记忆内容
            tags(string): 逗号分隔标签
            valence(number): 情绪效价，-1 到 1
            arousal(number): 唤醒度，0 到 1
            importance(number): 重要度，1 到 10
        """
        embedding = await self.embed_text(content)
        memory_id, merged = self.store.save_memory(
            category, content, tags, valence, arousal, importance, "llm", embedding=embedding
        )
        action = "merged" if merged else "saved"
        return event.plain_result(f"{action} memory #{memory_id}")

    @filter.llm_tool(name="memory_query")
    async def memory_query(self, event: AstrMessageEvent, keyword: str = "", category: str = "", limit: int = 5):
        """查询长期记忆。

        Args:
            keyword(string): 搜索关键词，可为空
            category(string): 记忆分类，可为空
            limit(number): 返回数量
        """
        embedding = await self.embed_text(keyword) if keyword else None
        rows = self.store.query(
            keyword=keyword, category=category, limit=limit, include_archive=True, query_embedding=embedding
        )
        return event.plain_result(format_memory_list(rows))

    @filter.llm_tool(name="memory_surface")
    async def memory_surface(self, event: AstrMessageEvent, limit: int = 3):
        """主动浮现高权重记忆。

        Args:
            limit(number): 返回数量
        """
        rows = self.store.surface(limit=limit)
        return event.plain_result(format_memory_list(rows, "no active memory surfaced"))

    @filter.llm_tool(name="memory_mark_core")
    async def memory_mark_core(self, event: AstrMessageEvent, memory_id: int):
        """将记忆标记为 core 层。

        Args:
            memory_id(number): 记忆 ID
        """
        ok = self.store.mark_core(int(memory_id))
        return event.plain_result("marked core" if ok else "memory not found")

    @filter.llm_tool(name="memory_resolve")
    async def memory_resolve(self, event: AstrMessageEvent, memory_id: int):
        """标记记忆事件已解决。

        Args:
            memory_id(number): 记忆 ID
        """
        ok = self.store.resolve(int(memory_id))
        return event.plain_result("resolved" if ok else "memory not found")

    @filter.llm_tool(name="memory_decay_status")
    async def memory_decay_status(self, event: AstrMessageEvent, none: str = ""):
        """查看记忆衰减状态。

        Args:
            none(string): 无参数，传空字符串
        """
        stats = self.store.stats()
        return event.plain_result(str(stats))

    @filter.llm_tool(name="memory_link")
    async def memory_link(self, event: AstrMessageEvent, from_id: int, to_id: int):
        """建立双向普通关联。

        Args:
            from_id(number): 起始记忆 ID
            to_id(number): 目标记忆 ID
        """
        link_id = self.store.link(int(from_id), int(to_id), "related")
        return event.plain_result(f"linked #{link_id}")

    @filter.llm_tool(name="memory_causal_link")
    async def memory_causal_link(self, event: AstrMessageEvent, from_id: int, to_id: int, link_type: str = "causes"):
        """建立因果或时序关联。

        Args:
            from_id(number): 起始记忆 ID
            to_id(number): 目标记忆 ID
            link_type(string): causes/follows/resolves/escalates/contradicts
        """
        link_id = self.store.link(int(from_id), int(to_id), link_type)
        return event.plain_result(f"linked #{link_id}")

    @filter.llm_tool(name="memory_query_links")
    async def memory_query_links(self, event: AstrMessageEvent, memory_id: int):
        """查询某条记忆的所有关联。

        Args:
            memory_id(number): 记忆 ID
        """
        rows = self.store.links_for(int(memory_id))
        if not rows:
            return event.plain_result("no links")
        text = "\n".join(f"#{r['id']} {r['from_id']} -[{r['link_type']}]-> {r['to_id']}" for r in rows)
        return event.plain_result(text)

    @filter.llm_tool(name="memory_supersede")
    async def memory_supersede(self, event: AstrMessageEvent, old_id: int, new_id: int):
        """标记旧事实被新事实替代。

        Args:
            old_id(number): 旧记忆 ID
            new_id(number): 新记忆 ID
        """
        ok = self.store.supersede(int(old_id), int(new_id))
        return event.plain_result("superseded" if ok else "memory not found")

    @filter.llm_tool(name="memory_kb_archive")
    async def memory_kb_archive(self, event: AstrMessageEvent, none: str = ""):
        """手动触发知识库归档占位接口。

        Args:
            none(string): 无参数，传空字符串
        """
        return event.plain_result("kb archive hook is ready; connect AstrBot knowledge base API before enabling writes")

    @filter.llm_tool(name="memory_event_view")
    async def memory_event_view(self, event: AstrMessageEvent, memory_id: int):
        """查看完整事件档案。

        Args:
            memory_id(number): 记忆 ID
        """
        row = self.store.get_memory(int(memory_id))
        links = self.store.links_for(int(memory_id))
        link_text = "\n".join(f"{r['from_id']} -[{r['link_type']}]-> {r['to_id']}" for r in links) or "no links"
        memory_text = format_memory(row) if row else f"memory #{memory_id} not found"
        return event.plain_result(f"{memory_text}\n\nlinks:\n{link_text}")

    @filter.llm_tool(name="commitment_save")
    async def commitment_save(self, event: AstrMessageEvent, type: str, who: str, content: str, due_date: str = ""):
        """记录承诺、心愿或约定。

        Args:
            type(string): promise/wish/pact
            who(string): 谁的承诺或心愿
            content(string): 内容
            due_date(string): 截止日期，可为空
        """
        cid = self.store.save_commitment(type, who, content, due_date)
        return event.plain_result(f"saved commitment #{cid}")

    @filter.llm_tool(name="commitment_query")
    async def commitment_query(self, event: AstrMessageEvent, status: str = "active", limit: int = 10):
        """查询承诺状态。

        Args:
            status(string): active/fulfilled/broken/cancelled
            limit(number): 返回数量
        """
        rows = self.store.query_commitments(status, limit)
        if not rows:
            return event.plain_result("no commitments")
        text = "\n".join(f"#{r['id']} [{r['type']}/{r['status']}] {r['who']}: {r['content']}" for r in rows)
        return event.plain_result(text)

    @filter.llm_tool(name="commitment_fulfill")
    async def commitment_fulfill(self, event: AstrMessageEvent, commitment_id: int):
        """标记承诺已兑现。

        Args:
            commitment_id(number): 承诺 ID
        """
        ok = self.store.fulfill_commitment(int(commitment_id))
        return event.plain_result("fulfilled" if ok else "commitment not found")
