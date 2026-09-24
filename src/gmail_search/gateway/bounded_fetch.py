"""Batched reads from a server-side cursor whose rows are bounded in SQL.

Each row is already capped at `row_byte_cap` by the statement itself, so one
batch holds at most `batch_rows * row_byte_cap` bytes. The batch size is chosen
to keep that product under a fixed ceiling. Liveness is checked between
batches, not per row: one round trip and one check per batch instead of one
per row (per-row checks made a 10k-row read take ~30 s).
"""
from contextlib import aclosing

BATCH_MEMORY_BYTES = 64 * 1024 * 1024
MAX_BATCH_ROWS = 512


def batch_rows_for(row_byte_cap):
    """Largest batch whose worst case stays within BATCH_MEMORY_BYTES."""
    if type(row_byte_cap) is not int or row_byte_cap <= 0:
        raise ValueError('Invalid row byte cap')
    return max(1, min(MAX_BATCH_ROWS, BATCH_MEMORY_BYTES // row_byte_cap))


async def _batched_rows(cursor, batch_rows, between_batches):
    while True:
        batch = await cursor.fetchmany(batch_rows)
        if not batch:
            return
        for item in batch:
            yield item
        if len(batch) < batch_rows:
            return
        await between_batches()


def bounded_rows(cursor, *, row_byte_cap, between_batches):
    """Async-iterate rows; use as `async with bounded_rows(...) as rows`."""
    return aclosing(_batched_rows(cursor, batch_rows_for(row_byte_cap), between_batches))
