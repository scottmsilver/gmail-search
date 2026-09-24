"""Capability-bound structured mail metadata through the restricted SQL gateway.

Only this module builds the fixed SQL shape. No caller SQL, database credential,
embedding provider, search index or bootstrap identity is accepted.
"""

import asyncio
from datetime import date
import json
import re

from psycopg import sql

from .database import QueryResult
from .registry import AccessDenied
from .service import _drain
from .tool_deadline import tool_deadline

MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_COLUMNS = (
    "owner_id",
    "thread_id",
    "summary_id",
    "subject",
    "participants",
    "message_count",
    "date_first",
    "date_last",
    "snippet",
)


def _text(value, maximum, *, empty=True):
    if (
        type(value) is not str
        or len(value) > maximum
        or "\x00" in value
        or (not empty and not value)
    ):
        raise ValueError("Invalid metadata query text")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ValueError("Invalid metadata query text") from None
    return value


def _options(
    sender, subject_contains, date_from, date_to, label, has_attachment, order_by, limit
):
    _text(sender, 1000)
    _text(subject_contains, 1000)
    _text(label, 256)
    for value in (date_from, date_to):
        _text(value, 10)
        if value:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError("Invalid metadata query date")
            try:
                date.fromisoformat(value)
            except ValueError:
                raise ValueError("Invalid metadata query date") from None
    if (
        (date_from and date_to and date_from > date_to)
        or (has_attachment is not None and type(has_attachment) is not bool)
        or type(order_by) is not str
        or order_by not in ("date_desc", "date_asc")
        or type(limit) is not int
        or not 1 <= limit <= 100
    ):
        raise ValueError("Invalid metadata query options")


def _statement(
    owner,
    sender,
    subject_contains,
    date_from,
    date_to,
    label,
    has_attachment,
    order_by,
    limit,
):
    # Literal quoting handles apostrophes/backslashes without interpolating SQL
    # syntax from values; the analytical compiler subsequently binds literals.
    def literal(value):
        return sql.Literal(value).as_string()

    where = ["m.user_id=" + literal(owner)]
    for column, value in (("from_addr", sender), ("subject", subject_contains)):
        if value:
            where.append("m." + column + " LIKE " + literal("%" + value + "%"))
    if date_from:
        where.append("m.date>=" + literal(date_from))
    if date_to:
        where.append("m.date<=" + literal(date_to + "T23:59:59+00:00"))
    if label:
        where.append("m.labels LIKE " + literal('%"' + label + '"%'))
    if has_attachment is not None:
        where.append(
            ("" if has_attachment else "NOT ")
            + "EXISTS(SELECT 1 FROM attachments a WHERE a.user_id=m.user_id AND a.message_id=m.id)"
        )
    direction = "DESC" if order_by == "date_desc" else "ASC"
    return f"""WITH matched AS (
        SELECT m.user_id,m.thread_id,MAX(m.date) AS matching_date FROM messages m
        WHERE {" AND ".join(where)} GROUP BY m.user_id,m.thread_id
    ), picked AS (
        SELECT user_id,thread_id,matching_date FROM matched
        ORDER BY matching_date {direction},thread_id ASC LIMIT {limit + 1}
    ), latest AS (
        SELECT m.user_id,m.thread_id,substr(m.body_text,1,500) AS snippet,
            row_number() OVER(PARTITION BY m.user_id,m.thread_id
                ORDER BY m.date DESC NULLS LAST,m.id ASC) AS position
        FROM messages m JOIN picked p ON p.user_id=m.user_id AND p.thread_id=m.thread_id
        WHERE m.user_id={literal(owner)}
    )
    SELECT p.user_id AS owner_id,p.thread_id,t.thread_id AS summary_id,t.subject,t.participants,
        t.message_count,t.date_first,t.date_last,l.snippet
    FROM picked p LEFT JOIN thread_summary t ON t.user_id=p.user_id AND t.thread_id=p.thread_id
        LEFT JOIN latest l ON l.user_id=p.user_id AND l.thread_id=p.thread_id AND l.position=1
    ORDER BY p.matching_date {direction},p.thread_id ASC"""


def _invalid():
    raise RuntimeError("Invalid metadata result")


def _format(result, owner, limit):
    if (
        not isinstance(result, QueryResult)
        or result.columns != _COLUMNS
        or type(result.complete) is not bool
        or not isinstance(result.rows, (tuple, list))
        or len(result.rows) > limit + 1
    ):
        _invalid()
    reasons = set()
    if not result.complete:
        reasons.add("query_budget")
    if len(result.rows) > limit:
        reasons.add("result_limit")
    output = []
    seen = set()
    size = 0
    for index, raw in enumerate(result.rows):
        if not isinstance(raw, (tuple, list)) or len(raw) != len(_COLUMNS):
            _invalid()
        item = dict(zip(_COLUMNS, raw, strict=True))
        try:
            if item["owner_id"] != owner:
                _invalid()
            tid = _text(item["thread_id"], 2048, empty=False)
            if tid in seen:
                _invalid()
            seen.add(tid)
            if item["summary_id"] is None:
                if index < limit:
                    reasons.add("missing_summary")
                continue
            if item["summary_id"] != tid:
                _invalid()
            if item["snippet"] is None:
                item["snippet"] = ""
            for key in (
                "subject",
                "date_first",
                "date_last",
                "snippet",
                "participants",
            ):
                _text(item[key], MAX_RESPONSE_BYTES)
            if (
                len(item["snippet"]) > 500
                or type(item["message_count"]) is not int
                or item["message_count"] < 0
            ):
                _invalid()
            participants = json.loads(item["participants"])
            if type(participants) is not list:
                _invalid()
            for participant in participants:
                _text(participant, MAX_RESPONSE_BYTES)
        except (ValueError, TypeError, UnicodeError, RecursionError):
            _invalid()
        if index >= limit:
            continue
        entry = {
            key: item[key]
            for key in (
                "thread_id",
                "subject",
                "message_count",
                "date_first",
                "date_last",
                "snippet",
            )
        }
        entry.update(participants=participants, cite_ref=tid)
        encoded = json.dumps(entry, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if size + len(encoded) > MAX_RESPONSE_BYTES - 4096:
            reasons.add("response_bytes")
            break
        size += len(encoded)
        output.append(entry)
    if not output and reasons:
        raise RuntimeError("Metadata query could not complete within its limits.")
    response = dict(
        results=output,
        coverage=dict(
            selection_complete=not reasons,
            returned_threads=len(output),
            limit=limit,
            reasons=sorted(reasons),
        ),
    )
    if (
        len(json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        > MAX_RESPONSE_BYTES
    ):
        raise RuntimeError("Metadata query could not complete within its limits.")
    return response


class RunMetadataService:
    def __init__(self, capabilities, gateway):
        self.capabilities, self.gateway = capabilities, gateway

    async def authorize(self, token):
        return await _drain(
            asyncio.to_thread(
                self.capabilities.authorize,
                token,
                audience="retrieval",
                operation="query.emails",
            )
        )

    async def query_emails(
        self,
        token,
        *,
        sender="",
        subject_contains="",
        date_from="",
        date_to="",
        label="",
        has_attachment=None,
        order_by="date_desc",
        limit=20,
    ):
        deadline = tool_deadline()
        async with asyncio.timeout_at(deadline):
            lease = await self.authorize(token)
        _options(
            sender,
            subject_contains,
            date_from,
            date_to,
            label,
            has_attachment,
            order_by,
            limit,
        )
        statement = _statement(
            lease.owner_id,
            sender,
            subject_contains,
            date_from,
            date_to,
            label,
            has_attachment,
            order_by,
            limit,
        )

        async def check():
            current = await self.authorize(token)
            if current.owner_id != lease.owner_id or current.run_id != lease.run_id:
                raise AccessDenied()
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Metadata query deadline exceeded")

        async def watch():
            while True:
                await asyncio.sleep(0.05)
                await check()

        async def execute():
            async with asyncio.timeout_at(deadline):
                result = await self.gateway.query(lease.owner_id, statement)
                return _format(result, lease.owner_id, limit)

        work = asyncio.create_task(execute())
        watcher = asyncio.create_task(watch())
        try:
            done, _ = await asyncio.wait(
                (work, watcher), return_when=asyncio.FIRST_COMPLETED
            )
            if watcher in done:
                await watcher
            response = await work
        finally:
            for task in (work, watcher):
                if not task.done() and not task.cancelling():
                    task.cancel()
            await _drain(asyncio.gather(work, watcher, return_exceptions=True))
        await check()
        return response
