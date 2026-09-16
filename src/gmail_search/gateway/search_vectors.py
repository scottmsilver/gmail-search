"""Page bounded owner vectors and retain only the best dot-product candidates.

The caller supplies an already owner-bound snapshot. Completion refers to that
restricted selection; it does not imply an up-to-date whole-mailbox index.
Scores preserve the legacy restricted-search raw float32 dot product. The
existing ANN manual reranker normalizes vectors; do not call these uniformly
cosine scores for a corpus containing non-unit stored embeddings.
"""
import asyncio
from dataclasses import dataclass
import heapq
import inspect
import math
from numbers import Real

import numpy as np

from .search_queries import Selection, _ids


@dataclass(frozen=True)
class VectorCandidates:
    ids: tuple[int, ...]
    scores: tuple[float, ...]
    complete: bool
    reasons: tuple[str, ...]
    examined: int


async def exact_candidates(queries, candidate_ids, vector, *, dimensions, top_k,
                           deadline, check_active, max_vectors=200_000):
    candidates = _ids(candidate_ids, maximum=20_000)
    if (type(dimensions) is not int or not 1 <= dimensions <= 4096
            or type(top_k) is not int or not 1 <= top_k <= 10_000
            or type(max_vectors) is not int or not 1 <= max_vectors <= 200_000
            or type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError('Invalid exact search options')
    if isinstance(vector, np.ndarray):
        if vector.ndim != 1 or vector.size != dimensions or vector.dtype.kind not in 'fiu':
            raise ValueError('Invalid query vector')
    elif isinstance(vector, (tuple, list)):
        if len(vector) != dimensions or any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in vector):
            raise ValueError('Invalid query vector')
    else:
        raise ValueError('Invalid query vector')
    with np.errstate(over='ignore', invalid='ignore'):
        try:
            query = np.array(vector, dtype=np.float32, copy=True)
        except (TypeError, ValueError, OverflowError):
            raise ValueError('Invalid query vector') from None
    if not np.isfinite(query).all() or not np.any(query) or np.max(np.abs(query)) > 1e6:
        raise ValueError('Invalid query vector')
    allowed = frozenset(candidates)
    heap = []
    reasons = set()
    examined = 0
    after = None

    async def check():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError('Search deadline exceeded')
        active = check_active()
        if inspect.isawaitable(active):
            active = await active
        if active is False:
            raise PermissionError('Mailbox search was revoked')

    async with asyncio.timeout_at(deadline):
        await check()
        while allowed:
            limit = min(128, max_vectors - examined)
            if limit <= 0:
                reasons.add('vector_limit')
                break
            page = await queries.restricted_vectors_page(candidates, after=after, limit=limit)
            await check()
            if not isinstance(page, Selection) or len(page.rows) > limit:
                raise RuntimeError('Invalid vector page')
            last = after
            for row in page.rows:
                if (type(row.id) is not int or not 0 < row.id < 2**63
                        or (last is not None and row.id <= last)
                        or row.message_id not in allowed):
                    raise RuntimeError('Invalid vector page')
                last = row.id
                examined += 1
                if row.vector_status != 'ok':
                    if row.vector_status not in ('missing_vector', 'model_mismatch', 'invalid_vector'):
                        raise RuntimeError('Invalid vector page')
                    reasons.add(row.vector_status)
                    continue
                if not isinstance(row.embedding, bytes) or len(row.embedding) != dimensions * 4:
                    reasons.add('invalid_vector')
                    continue
                values = np.frombuffer(row.embedding, dtype='<f4')
                if not np.isfinite(values).all() or not np.any(values):
                    reasons.add('invalid_vector')
                    continue
                with np.errstate(over='ignore', invalid='ignore'):
                    score = float(np.dot(values, query))
                if not math.isfinite(score):
                    reasons.add('invalid_vector')
                    continue
                item = (score, -row.id)
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            if page.complete:
                if page.next_cursor is not None:
                    raise RuntimeError('Invalid vector page')
                break
            if page.next_cursor is None:
                if page.reason not in ('row_limit', 'page_bytes', 'row_bytes', 'session_budget'):
                    raise RuntimeError('Invalid vector page')
                reasons.add(page.reason)
                break
            if (not page.rows or type(page.next_cursor) is not int
                    or page.next_cursor != last or (after is not None and page.next_cursor <= after)):
                raise RuntimeError('Invalid vector page')
            after = page.next_cursor
            await asyncio.sleep(0)
            await check()
        await check()
    ranked = sorted(heap, reverse=True)
    return VectorCandidates(tuple(-identifier for _, identifier in ranked),
                            tuple(score for score, _ in ranked), not reasons,
                            tuple(sorted(reasons)), examined)
