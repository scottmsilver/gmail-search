#!/usr/bin/env python3
"""Closed, guest-only stdio MCP adapter for :mod:`guest_mail_tools`.

This is deliberately a small JSON-RPC implementation instead of an application
server.  The guest image needs only the standard library; the installed MCP SDK
is used by tests as an independent client.  Tool execution remains entirely in
``GuestMailTools``: this adapter never accepts routing, identity, owner, or
capability configuration from the MCP client.
"""
import asyncio
import inspect
import json
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guest_tool_config import LEGACY_PROFILE, READ_PROFILE, RAW_PROFILE, LEGACY_TOOLS, READ_TOOLS, RAW_TOOLS
from guest_mail_tool_cli import RUN_ROOT, _capabilities
from guest_mail_tools import GuestMailTools, MAX_EXACT_INTEGER, ToolError, _invalid_constant, _object, _drain, _search_options, _facts_options, _judge_options, _metadata_options, _attachment_options


MAX_INPUT_LINE_BYTES = 256 * 1024
MAX_OUTPUT_LINE_BYTES = 20 * 1024**2
MAX_OUTSTANDING_CALLS = 2
MAX_REQUEST_ID_BYTES = 256
# A parallel mail-researcher child gets this many mail calls; checks are free.
# Asked in the prompt, children still made ~57 calls each (2026-09-24).
SUBAGENT_CALL_BUDGET = 10
BUDGET_FREE_TOOLS = frozenset(('judge', 'describe_schema'))
BUDGET_SPENT = ('Mail call budget for this subagent is spent. Stop searching: answer now from the evidence '
                'you already have, and name what is still missing.')
OUTPUT_SECONDS = 5
SUPPORTED_PROTOCOL_VERSIONS = frozenset((
    "2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25",
))
LATEST_PROTOCOL_VERSION = "2025-11-25"
_ID = "[A-Za-z0-9_-]{1,256}"


class ProtocolError(ValueError):
    """A fixed public protocol failure, safe to return to a client."""


def _strict_json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid_constant)
        # JSON's escaped surrogate form is not valid UTF-8 JSON-RPC content.
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ToolError, ValueError, TypeError, UnicodeError, RecursionError):
        raise ProtocolError("Invalid JSON-RPC message.") from None
    return value


def parse_message_line(raw):
    """Decode one bounded JSON-RPC line, rejecting ambiguity before dispatch."""
    if type(raw) is not bytes or len(raw) > MAX_INPUT_LINE_BYTES:
        raise ProtocolError("JSON-RPC message exceeds the byte limit.")
    value = _strict_json(raw)
    if type(value) is not dict:
        raise ProtocolError("JSON-RPC message must be an object.")
    return value


def _error(identifier, code, message):
    return {"jsonrpc": "2.0", "id": identifier, "error": {"code": code, "message": message}}


def _result(identifier, value):
    return {"jsonrpc": "2.0", "id": identifier, "result": value}


def _is_request_identifier(value):
    if type(value) is str:
        return len(value.encode("utf-8")) <= MAX_REQUEST_ID_BYTES
    return type(value) is int and -(2**53) < value < 2**53


def _request_key(value):
    return (type(value), value)


def _output_schema():
    # GuestMailTools always returns a JSON object.  The gateway owns the
    # operation-specific envelope, so a narrower schema here would be false.
    return {"type": "object"}


def _tools():
    """Construct the core only from the fixed, ownership-checked guest config."""
    return GuestMailTools(RUN_ROOT / "work", _capabilities(), port=18080)


def _allowed_tools(profile):
    return {LEGACY_PROFILE:LEGACY_TOOLS,READ_PROFILE:READ_TOOLS,RAW_PROFILE:RAW_TOOLS}[profile]


def _tool_definitions(profile=LEGACY_PROFILE):
    if profile not in (LEGACY_PROFILE, READ_PROFILE, RAW_PROFILE):
        raise ToolError("Invalid guest tool profile.")
    page_fields = {
        "message_offset": {"type": "integer", "minimum": 0, "maximum": 10000},
        "message_limit": {"type": "integer", "minimum": 1, "maximum": 100},
        "body_offset": {"type": "integer", "minimum": 0, "maximum": 2147483646},
        "body_limit": {"type": "integer", "minimum": 1, "maximum": 100000},
        "attachment_after_id": {"type": "integer", "minimum": 0, "maximum": 9223372036854775807},
        "attachment_limit": {"type": "integer", "minimum": 1, "maximum": 100},
    }
    if profile==RAW_PROFILE:
        page_fields['attachment_after_id']['maximum']=MAX_EXACT_INTEGER
    if profile == LEGACY_PROFILE:
        page_fields.pop("attachment_after_id")
        page_fields.pop("attachment_limit")
    batch = {"type": "array", "minItems": 1, "maxItems": 20}
    definitions = [
        {
            "name": "describe_schema",
            "description": "Return the guest-scoped SQL schema metadata.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            "outputSchema": _output_schema(),
        },
        {
            "name": "sql_query_batch",
            "description": ("Up to 20 SQL queries over mail metadata: ids, threads, senders, dates, labels, "
                            "attachment names. Not for finding mail by content: LIKE/ILIKE/regex on body_text "
                            "or extracted_text is rejected. Use search_emails_batch for that."),
            "inputSchema": {
                "type": "object", "properties": {"queries": {**batch, "items": {"type": "string", "minLength": 1}}},
                "required": ["queries"], "additionalProperties": False,
            },
            "outputSchema": _output_schema(),
        },
        {
            "name": "get_thread_batch",
            "description": "Read text-only pages for up to 20 guest-scoped mail threads.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "thread_ids": {**batch, "items": {"type": "string", "pattern": _ID}},
                    "body_format": {"const": "text"},
                    **page_fields,
                },
                "required": ["thread_ids"], "additionalProperties": False,
            },
            "outputSchema": _output_schema(),
        },
        {
            "name": "publish_artifact_batch",
            "description": "Publish up to 20 relative workspace files as octet-stream attachments.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "items": {
                        **batch,
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "minLength": 1},
                                "name": {"type": "string", "minLength": 1, "maxLength": 200},
                                "mime_type": {"const": "application/octet-stream"},
                            },
                            "required": ["path"], "additionalProperties": False,
                        },
                    },
                },
                "required": ["items"], "additionalProperties": False,
            },
            "outputSchema": _output_schema(),
        },
        {
            "name": "search_emails_batch",
            "description": ("The way to find mail by content: up to 20 hybrid searches (BM25 keyword + semantic) "
                            "with optional date_from/date_to and sender/recipient filters; use these instead of SQL for "
                            "who-sent-what questions. Read coverage metadata for approximate results "
                            "and clipped bodies."),
            "inputSchema": {
                "type": "object", "additionalProperties": False, "required": ["searches"],
                "properties": {"searches": {**batch, "items": {
                    "type": "object", "additionalProperties": False, "required": ["query"],
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
                        "detail": {"enum": ["refs", "snippet", "summary", "full"]},
                        "max_matches": {"type": "integer", "minimum": 0, "maximum": 100},
                        "date_from": {"anyOf": [{"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"}, {"type": "null"}]},
                        "date_to": {"anyOf": [{"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"}, {"type": "null"}]},
                        "sender": {"type": "string", "minLength": 1, "maxLength": 256,
                                   "description": "Only mail whose From contains this (address, domain or name)."},
                        "recipient": {"type": "string", "minLength": 1, "maxLength": 256,
                                      "description": "Only mail whose To contains this (address, domain or name)."},
                    },
                }}},
            },
            "outputSchema": _output_schema(),
        },
        {
            "name": "find_facts",
            "description": "Search facts extracted from your mail. Read coverage for corpus limits and extraction freshness; exhaustive requests may be incomplete.",
            "inputSchema": {
                "type": "object", "additionalProperties": False, "required": ["query"],
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                    "exhaustive": {"type": "boolean", "default": True},
                    "k": {"type": "integer", "minimum": 1, "maximum": 500, "default": 200},
                },
            },
            "outputSchema": _output_schema(),
        },
        {
            "name": "judge",
            "description": ("Ask Jev (a fast judgment model, ~100 ms) typed questions instead of guessing: "
                            "relevance, classification (e.g. order vs quote), which option is meant, or whether "
                            "the evidence gathered so far answers the user's question (use a noul with id 'answered' for that check; a low answer grants more reasoning). `state` is the text to "
                            "judge (<=32 KiB). Each question: noul (yes/no probability), choice (options "
                            "'label: description') or score (ordered level descriptions). Returns probabilities "
                            "and confidence per question."),
            "inputSchema": {
                "type": "object", "additionalProperties": False, "required": ["state", "questions"],
                "properties": {
                    "state": {"type": "string", "minLength": 1, "maxLength": 32768,
                              "description": "The text (or JSON text) to judge."},
                    "questions": {"type": "array", "minItems": 1, "maxItems": 8, "items": {
                        "type": "object", "additionalProperties": False, "required": ["id", "type", "instructions"],
                        "properties": {
                            "id": {"type": "string", "description": "Short identifier, e.g. answered."},
                            "type": {"enum": ["noul", "choice", "score"]},
                            "instructions": {"type": "string", "minLength": 1},
                            "options": {"type": "array", "items": {"type": "string"},
                                        "description": "choice: 'label: description' per option; score: ordered level descriptions; noul: omit."},
                        }}},
                },
            },
            "outputSchema": _output_schema(),
        },
        {
            "name": "query_emails_batch",
            "description": "Filter mail by sender, subject, dates, label or attachment presence. Returns threads in matching-message date order; read coverage for limits.",
            "inputSchema": {
                "type": "object", "additionalProperties": False, "required": ["filters"],
                "properties": {"filters": {**batch, "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "sender": {"type": "string", "maxLength": 1000},
                        "subject_contains": {"type": "string", "maxLength": 1000},
                        "label": {"type": "string", "maxLength": 256},
                        "date_from": {"type": "string", "pattern": "^([0-9]{4}-[0-9]{2}-[0-9]{2})?$"},
                        "date_to": {"type": "string", "pattern": "^([0-9]{4}-[0-9]{2}-[0-9]{2})?$"},
                        "has_attachment": {"type": ["boolean", "null"]},
                        "order_by": {"enum": ["date_desc", "date_asc"]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                }}},
            },
            "outputSchema": _output_schema(),
        },
    ]
    definitions.append({
        "name": "get_attachment_batch",
        "description": "Read attachment metadata or stored text pages. Text may be absent or incomplete; inspect coverage and paging fields.",
        "inputSchema": {"type": "object", "additionalProperties": False, "required": ["items"],
                        "properties": {"items": {**batch, "items": {"oneOf": [
                            {"type": "object", "additionalProperties": False, "required": ["attachment_id"],
                             "properties": {"attachment_id": {"type": "integer", "minimum": 1, "maximum": 9223372036854775807},
                                            "mode": {"const": "text", "default": "text"},
                                            "offset": {"type": "integer", "minimum": 0, "maximum": 2147483646, "default": 0},
                                            "limit": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 20000}}},
                            {"type": "object", "additionalProperties": False, "required": ["attachment_id", "mode"],
                             "properties": {"attachment_id": {"type": "integer", "minimum": 1, "maximum": 9223372036854775807},
                                            "mode": {"const": "meta"}}},
                        ]}}}},
        "outputSchema": _output_schema(),
    })
    if profile==RAW_PROFILE:
        attachment=definitions[-1]
        attachment['description']='Read attachment metadata/text or download opaque raw files into the private guest workspace. Raw returns only a relative path, byte size and SHA-256; stored text coverage remains explicit.'
        attachment['inputSchema']['properties']['items']['items']['oneOf'].append({
            'type':'object','additionalProperties':False,'required':['attachment_id','mode'],
            'properties':{'attachment_id':{'type':'integer','minimum':1,'maximum':9223372036854775807},
                          'mode':{'const':'raw'}}})
        for item in attachment['inputSchema']['properties']['items']['items']['oneOf']:
            item['properties']['attachment_id']['maximum']=MAX_EXACT_INTEGER
    if profile==RAW_PROFILE:
        # Fixed v3 provider schema subset. Runtime validators still enforce
        # these formats; the gateway intentionally does not accept regexes.
        def supported(value):
            if type(value) is dict:
                if value.get('maximum')==MAX_EXACT_INTEGER:
                    value.pop('maximum')
                    value['description']=(value.get('description','')+' Maximum exact supported ID: '+str(MAX_EXACT_INTEGER)+'.').strip()
                if 'pattern' in value:
                    pattern=value.pop('pattern')
                    value['description']=(value.get('description','')+' Required format: '+pattern).strip()
                    value.setdefault('maxLength',256)
                for child in value.values():supported(child)
            elif type(value) is list:
                for child in value:supported(child)
        for definition in definitions:supported(definition['inputSchema'])
    allowed=_allowed_tools(profile)
    return [item for item in definitions if item['name'] in allowed]


_TOOL_NAMES = frozenset(item["name"] for item in _tool_definitions(RAW_PROFILE))


def _valid_arguments(name, arguments, *, allow_raw=False, exact_ids=False):
    """Reject obviously invalid calls before opening trusted configuration."""
    if type(arguments) is not dict:
        return False
    if name == "describe_schema":
        return not arguments
    if name == "sql_query_batch":
        values = arguments.get("queries")
        return set(arguments) == {"queries"} and type(values) is list and 1 <= len(values) <= 20 and all(
            type(value) is str and value for value in values
        )
    if name == "query_emails_batch":
        values = arguments.get("filters")
        if set(arguments) != {"filters"} or type(values) is not list or not 1 <= len(values) <= 20:
            return False
        try:
            for item in values:
                _metadata_options(item)
        except (ValueError, TypeError, UnicodeError):
            return False
        return True
    if name == "get_attachment_batch":
        values=arguments.get("items")
        if set(arguments)!={"items"} or type(values) is not list or not 1<=len(values)<=20:
            return False
        try:
            for item in values:_attachment_options(item,allow_raw=allow_raw,exact_ids=exact_ids)
        except (ValueError,TypeError):return False
        return True
    if name == "judge":
        try:
            _judge_options(arguments)
        except (ValueError, TypeError, UnicodeError):
            return False
        return True
    if name == "find_facts":
        try:
            _facts_options(arguments)
        except (ValueError, TypeError, UnicodeError):
            return False
        return True
    if name == "search_emails_batch":
        values = arguments.get("searches")
        if set(arguments) != {"searches"} or type(values) is not list or not 1 <= len(values) <= 20:
            return False
        try:
            for item in values:
                _search_options(item)
        except (ValueError, TypeError, UnicodeError):
            return False
        return True
    if name == "get_thread_batch":
        allowed = {"thread_ids", "body_format", "message_offset", "message_limit", "body_offset", "body_limit", "attachment_after_id", "attachment_limit"}
        values = arguments.get("thread_ids")
        if set(arguments) - allowed or type(values) is not list or not 1 <= len(values) <= 20:
            return False
        if any(type(value) is not str or not value for value in values):
            return False
        if arguments.get("body_format", "text") != "text":
            return False
        limits = {"message_offset": (0, 10000), "message_limit": (1, 100),
                  "body_offset": (0, 2147483646), "body_limit": (1, 100000),
                  "attachment_after_id": (0, MAX_EXACT_INTEGER if exact_ids else 9223372036854775807), "attachment_limit": (1, 100)}
        return all(field not in arguments or type(arguments[field]) is int and low <= arguments[field] <= high
                   for field, (low, high) in limits.items())
    if name == "publish_artifact_batch":
        values = arguments.get("items")
        if set(arguments) != {"items"} or type(values) is not list or not 1 <= len(values) <= 20:
            return False
        return all(type(item) is dict and "path" in item and not (set(item) - {"path", "name", "mime_type"})
                   and type(item["path"]) is str and item["path"]
                   and ("name" not in item or type(item["name"]) is str)
                   and ("mime_type" not in item or item["mime_type"] == "application/octet-stream")
                   for item in values)
    return False


def _budget_from_argv(argv):
    """SUBAGENT_CALL_BUDGET for a mail-researcher child's server (started with
    --subagent by guest-agent-subagent-mail-mcp.ts), else no budget."""
    if argv == ['--subagent']:
        return SUBAGENT_CALL_BUDGET
    if argv:
        raise SystemExit('Unsupported arguments.')
    return None


class GuestMailMCP:
    """One stdio connection with a closed lifecycle and at most two tool calls."""
    def __init__(self, tools_factory=_tools, emit=None, call_budget=None):
        self._call_budget = call_budget
        self._budgeted_calls = 0
        self._tools_factory = tools_factory
        self._tools_instance = None
        self._emit_callback = emit
        self._initialized = False
        self._ready = False
        self._active = {}
        self._closed=False
        self._close_task=None

    def _get_tools(self):
        if self._closed:raise ToolError('Guest tools are closing.')
        if self._tools_instance is None:
            instance=self._tools_factory()
            # Cache one immutable profile together with one admission-owning core.
            self._profile=instance.tool_profile
            _tool_definitions(self._profile)
            self._tools_instance=instance
        return self._tools_instance

    async def _emit(self, message):
        if self._emit_callback is None:
            return
        value = self._emit_callback(message)
        if inspect.isawaitable(value):
            await value

    async def receive(self, message):
        """Handle a decoded message.  Tool calls run in tracked background tasks."""
        if self._closed:return
        if type(message) is not dict or message.get("jsonrpc") != "2.0":
            await self._emit(_error(None, -32600, "Invalid Request."))
            return
        identifier = message.get("id")
        request = "id" in message
        if request and not _is_request_identifier(identifier):
            await self._emit(_error(None, -32600, "Invalid Request."))
            return
        method = message.get("method")
        params = message.get("params")
        if type(method) is not str or (params is not None and type(params) is not dict):
            if request:
                await self._emit(_error(identifier if _is_request_identifier(identifier) else None, -32600, "Invalid Request."))
            return
        if not request:
            await self._notification(method, params)
            return
        await self._request(identifier, method, params)

    async def _notification(self, method, params):
        if method == "notifications/initialized":
            if self._initialized:
                self._ready = True
            return
        if method != "notifications/cancelled" or type(params) is not dict:
            return
        identifier = params.get("requestId")
        if not _is_request_identifier(identifier):
            return
        task = self._active.get(_request_key(identifier))
        if task is not None and not task.done():
            task.cancel()

    async def _request(self, identifier, method, params):
        if method == "ping":
            await self._emit(_result(identifier, {}))
            return
        if method == "initialize":
            await self._initialize(identifier, params)
            return
        if not self._ready:
            await self._emit(_error(identifier, -32002, "Server is not initialized."))
            return
        if method == "tools/list":
            if params not in (None, {}) and set(params) != {"cursor"}:
                await self._emit(_error(identifier, -32602, "Invalid tools/list parameters."))
            elif params and params.get("cursor") not in (None, ""):
                await self._emit(_error(identifier, -32602, "This server does not paginate tools."))
            else:
                try:
                    self._get_tools()
                    result=_result(identifier, {"tools": _tool_definitions(self._profile)})
                except (ToolError,OSError,ValueError,TypeError):
                    result=_error(identifier,-32000,"Guest tools are unavailable.")
                await self._emit(result)
            return
        if method == "tools/call":
            await self._start_call(identifier, params)
            return
        await self._emit(_error(identifier, -32601, "Method not found."))

    async def _initialize(self, identifier, params):
        if self._initialized or type(params) is not dict:
            await self._emit(_error(identifier, -32600, "Invalid initialize request."))
            return
        version = params.get("protocolVersion")
        client = params.get("clientInfo")
        if (type(version) is not str or type(params.get("capabilities")) is not dict or type(client) is not dict
                or type(client.get("name")) is not str or type(client.get("version")) is not str):
            await self._emit(_error(identifier, -32600, "Invalid initialize request."))
            return
        self._initialized = True
        selected = version if version in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
        await self._emit(_result(identifier, {
            "protocolVersion": selected,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "guest-mail-tools", "version": "1.0.0"},
        }))

    async def _start_call(self, identifier, params):
        if type(params) is not dict or set(params) - {"name", "arguments", "_meta"}:
            await self._emit(_error(identifier, -32602, "Invalid tools/call parameters."))
            return
        name = params.get("name")
        arguments = params.get("arguments", {})
        if type(name) is not str or name not in _TOOL_NAMES:
            await self._emit(_error(identifier, -32602, "Unknown guest tool."))
            return
        if not _valid_arguments(name, arguments,allow_raw=True):
            await self._emit(_result(identifier, _tool_error("Invalid tool arguments.")))
            return
        key = _request_key(identifier)
        if key in self._active:
            await self._emit(_error(identifier, -32600, "Duplicate request id."))
            return
        if len(self._active) >= MAX_OUTSTANDING_CALLS:
            await self._emit(_error(identifier, -32000, "Too many concurrent requests."))
            return
        task = asyncio.create_task(self._run_call(key, identifier, name, arguments))
        self._active[key] = task
        # A cancellation can land before the new coroutine reaches its finally
        # block.  Keep the admission table correct in that narrow race too.
        def remove_finished(done):
            if not done.cancelled():
                done.exception()  # Consume transport failure without logging guest data.
            if self._active.get(key) is done:
                self._active.pop(key, None)
        task.add_done_callback(remove_finished)

    async def _run_call(self, key, identifier, name, arguments):
        try:
            # Construction has no await: concurrent requests share one core
            # and its two-socket admission limit for this MCP connection.
            tools=self._get_tools()
            allowed=_allowed_tools(self._profile)
            if (name not in allowed or self._profile==LEGACY_PROFILE and name=='get_thread_batch'
                    and {'attachment_after_id','attachment_limit'} & set(arguments)):
                raise ToolError('Unsupported guest tool.')
            if not _valid_arguments(name,arguments,allow_raw=self._profile==RAW_PROFILE,exact_ids=self._profile==RAW_PROFILE):
                raise ToolError('Unsupported guest tool arguments for this profile.')
            self._spend_call(name)
            result = await tools.dispatch(name, arguments)
            await self._emit(_result(identifier, _tool_result(result)))
        except asyncio.CancelledError:
            # The core's cancellation path closes and drains its socket/tasks.
            # MCP cancellation asks us not to send a late response.
            raise
        except ToolError as error:
            # Fixed guest/gateway strings (never mail content) the agent can act on.
            await self._emit(_result(identifier, _tool_error(str(error))))
        except (OSError, TimeoutError, ValueError, TypeError, UnicodeError, RecursionError):
            await self._emit(_result(identifier, _tool_error("Guest tool is unavailable.")))
        finally:
            current = asyncio.current_task()
            if self._active.get(key) is current:
                self._active.pop(key, None)

    def _spend_call(self, name):
        if self._call_budget is None or name in BUDGET_FREE_TOOLS:
            return
        self._budgeted_calls += 1
        if self._budgeted_calls > self._call_budget:
            raise ToolError(BUDGET_SPENT)

    async def wait_idle(self):
        while self._active:
            tasks = tuple(self._active.values())
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self):
        self._closed=True
        async def close():
            for task in tuple(self._active.values()):
                if not task.done() and not task.cancelling():task.cancel()
            await self.wait_idle()
            if self._tools_instance is not None:
                await self._tools_instance.aclose()
        if self._close_task is None or (self._close_task.done() and not self._close_task.cancelled()
                                       and self._close_task.exception() is not None):
            self._close_task=asyncio.create_task(close())
        await _drain(self._close_task)


def _tool_error(message):
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _tool_result(value):
    try:
        if type(value) is not dict:
            raise ValueError
        text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        text.encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return _tool_error("Guest tool returned an invalid result.")
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": value,
        "isError": type(value.get("error")) is str,
    }


async def _ready(fd, *, write=False):
    loop = asyncio.get_running_loop()
    ready = loop.create_future()

    def wake():
        if not ready.done():
            ready.set_result(None)

    add = loop.add_writer if write else loop.add_reader
    remove = loop.remove_writer if write else loop.remove_reader
    add(fd, wake)
    try:
        await ready
    finally:
        remove(fd)


async def _input_lines():
    """Yield bounded lines without accumulating an overlong client frame."""
    os.set_blocking(0, False)
    line = bytearray()
    discarding = False
    while True:
        try:
            chunk = os.read(0, 65536)
        except BlockingIOError:
            await _ready(0)
            continue
        if not chunk:
            if line and not discarding:
                yield bytes(line)
            elif discarding:
                yield None
            return
        for byte in chunk:
            if discarding:
                if byte == 10:
                    discarding = False
                    yield None
                continue
            if byte == 10:
                yield bytes(line)
                line.clear()
            elif len(line) >= MAX_INPUT_LINE_BYTES:
                line.clear()
                discarding = True
            else:
                line.append(byte)


class _Stdout:
    def __init__(self, on_failure=None):
        self._lock = asyncio.Lock()
        self.failed = False
        self._on_failure = on_failure

    async def __call__(self, message):
        try:
            data = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
            if len(data) > MAX_OUTPUT_LINE_BYTES:
                raise ValueError
        except (ValueError, TypeError, UnicodeError, RecursionError):
            identifier = message.get("id") if type(message) is dict else None
            try:
                if not _is_request_identifier(identifier):
                    identifier = None
            except UnicodeError:
                identifier = None
            data = json.dumps(_error(identifier, -32603, "Response exceeds server limits."),
                              ensure_ascii=True, separators=(",", ":")).encode() + b"\n"
        async with self._lock:
            if self.failed:
                raise ConnectionError("MCP output is unavailable")
            # Once any part of a frame is emitted, cancellation must not allow
            # another JSON message to be appended to its unfinished line.
            await _drain(self._write(data))

    async def _write(self, data):
        try:
            async with asyncio.timeout(OUTPUT_SECONDS):
                os.set_blocking(1, False)
                view = memoryview(data)
                while view:
                    try:
                        written = os.write(1, view)
                    except BlockingIOError:
                        await _ready(1, write=True)
                        continue
                    if written <= 0:
                        raise OSError("closed stdout")
                    view = view[written:]
        except (OSError, TimeoutError):
            # A failed partial frame cannot be repaired. Poison the transport
            # and terminate its reader rather than writing another response.
            self.failed = True
            if self._on_failure is not None:
                self._on_failure()
            raise


async def main(argv=()):
    current = asyncio.current_task()
    output = _Stdout(on_failure=current.cancel)
    server = GuestMailMCP(emit=output, call_budget=_budget_from_argv(list(argv)))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, current.cancel)
    try:
        async for raw in _input_lines():
            if raw is None:
                await server._emit(_error(None, -32700, "JSON-RPC message exceeds the byte limit."))
                continue
            try:
                await server.receive(parse_message_line(raw))
            except ProtocolError:
                await server._emit(_error(None, -32700, "Parse error."))
    except (asyncio.CancelledError, OSError, TimeoutError):
        pass
    finally:
        try:
            await _drain(server.close())
        except asyncio.CancelledError:
            # _drain has completed cleanup before propagating repeated signals.
            pass
    return 1 if output.failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
