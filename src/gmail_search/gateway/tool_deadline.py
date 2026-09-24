"""One wall-clock budget for every guest mail-tool call (search, thread,
facts, metadata, attachment text). Calls are meant to be fast; a call that
cannot finish in this budget fails instead of stalling the agent."""
import asyncio

TOOL_DEADLINE_SECONDS = 5.0
# Publication runs after the service settles, so it gets a little longer.
PUBLICATION_MARGIN_SECONDS = 0.5


def tool_deadline():
    return asyncio.get_running_loop().time() + TOOL_DEADLINE_SECONDS


def publication_deadline():
    return tool_deadline() + PUBLICATION_MARGIN_SECONDS
