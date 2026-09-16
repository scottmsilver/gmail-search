"""Bounded exact vector selection preserves semantic recall across DB pages."""
import asyncio
import importlib
import struct

import pytest

from gmail_search.gateway.search_queries import EmbeddingVectorRow, Selection


def _row(identifier, vector=(1., 0.), status='ok'):
    return EmbeddingVectorRow(identifier, 'm', struct.pack('<2f', *vector) if vector else None,
                              'test-model', status)


class Pages:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.cursors = []

    async def restricted_vectors_page(self, candidates, *, after=None, limit=128):
        assert candidates == ('m',)
        assert limit <= 128
        self.cursors.append(after)
        return next(self.pages)


async def _run(pages, **kwargs):
    module = importlib.import_module('gmail_search.gateway.search_vectors')
    return await module.exact_candidates(pages, ('m',), [1., 0.], dimensions=2,
        top_k=2, deadline=asyncio.get_running_loop().time()+5, check_active=lambda: True, **kwargs)


def test_best_vector_on_later_page_is_retained_and_ties_are_stable():
    async def scenario():
        pages=Pages([Selection((_row(1,(0.,1.)),_row(2,(.5,.5))),False,'row_limit',2),
                     Selection((_row(3),_row(4)),True)])
        result=await _run(pages)
        assert result.ids == (3,4)
        assert result.scores == pytest.approx((1.,1.))
        assert result.complete and result.examined == 4
        assert pages.cursors == [None,2]
    asyncio.run(scenario())


def test_missing_and_invalid_vectors_are_explicit_coverage_gaps():
    async def scenario():
        pages=Pages([Selection((_row(1),_row(2,None,'missing_vector'),
                               _row(3,(0.,0.)),_row(4,(float('nan'),0.))),True)])
        result=await _run(pages)
        assert result.ids == (1,)
        assert not result.complete
        assert set(result.reasons) == {'missing_vector','invalid_vector'}
        assert result.examined == 4
    asyncio.run(scenario())


def test_unpageable_database_cap_never_claims_complete():
    async def scenario():
        result=await _run(Pages([Selection((_row(1),),False,'session_budget')]))
        assert not result.complete and 'session_budget' in result.reasons
    asyncio.run(scenario())


@pytest.mark.parametrize('cursor',[0,1,'wrong'])
def test_nonadvancing_or_mismatched_cursor_fails_closed(cursor):
    async def scenario():
        with pytest.raises(RuntimeError,match='Invalid vector page'):
            await _run(Pages([Selection((_row(2),),False,'row_limit',cursor)]))
    asyncio.run(scenario())


def test_owner_revocation_stops_before_next_page():
    async def scenario():
        module=importlib.import_module('gmail_search.gateway.search_vectors')
        checks=0
        def active():
            nonlocal checks
            checks+=1
            return checks<2
        pages=Pages([Selection((_row(1),),False,'row_limit',1)])
        with pytest.raises(PermissionError):
            await module.exact_candidates(pages,('m',),[1.,0.],dimensions=2,top_k=2,
                deadline=asyncio.get_running_loop().time()+5,check_active=active)
        assert pages.cursors == [None]
    asyncio.run(scenario())


def test_expired_deadline_never_reads():
    async def scenario():
        module=importlib.import_module('gmail_search.gateway.search_vectors')
        pages=Pages([])
        with pytest.raises(TimeoutError):
            await module.exact_candidates(pages,('m',),[1.,0.],dimensions=2,top_k=2,
                deadline=asyncio.get_running_loop().time()-1,check_active=lambda: True)
        assert pages.cursors == []
    asyncio.run(scenario())


def test_page_must_not_return_unrequested_message():
    async def scenario():
        row=EmbeddingVectorRow(1,'foreign-message',struct.pack('<2f',1,0),'test-model','ok')
        with pytest.raises(RuntimeError,match='Invalid vector page'):
            await _run(Pages([Selection((row,),True)]))
    asyncio.run(scenario())


def test_vector_limit_is_reported_without_loading_next_page():
    async def scenario():
        pages=Pages([Selection((_row(1),_row(2)),False,'row_limit',2)])
        result=await _run(pages,max_vectors=2)
        assert not result.complete and result.reasons == ('vector_limit',)
        assert pages.cursors == [None]
    asyncio.run(scenario())


@pytest.mark.parametrize('vector',[[True,0.],[float('inf'),0.],[0.,0.],[[1.,0.]],['1','0']])
def test_invalid_query_vectors_rejected_before_database(vector):
    async def scenario():
        module=importlib.import_module('gmail_search.gateway.search_vectors')
        pages=Pages([])
        with pytest.raises(ValueError,match='Invalid query vector'):
            await module.exact_candidates(pages,('m',),vector,dimensions=2,top_k=2,
                deadline=asyncio.get_running_loop().time()+1,check_active=lambda: True)
        assert pages.cursors == []
    asyncio.run(scenario())


def test_duplicate_vector_across_pages_is_rejected():
    async def scenario():
        pages=Pages([Selection((_row(1),),False,'row_limit',1),Selection((_row(1),),True)])
        with pytest.raises(RuntimeError,match='Invalid vector page'):
            await _run(pages)
    asyncio.run(scenario())


def test_preserves_legacy_raw_dot_product_for_nonunit_stored_vectors():
    async def scenario():
        result=await _run(Pages([Selection((_row(1,(2.,2.)),_row(2)),True)]))
        assert result.ids == (1,2)
        assert result.scores == pytest.approx((2.,1.))
    asyncio.run(scenario())


def test_nested_vector_rejected_before_array_allocation(monkeypatch):
    async def scenario():
        module=importlib.import_module('gmail_search.gateway.search_vectors')
        def forbidden(*args,**kwargs):
            pytest.fail('Nested input reached numpy conversion')
        monkeypatch.setattr(module.np,'array',forbidden)
        monkeypatch.setattr(module.np,'asarray',forbidden)
        with pytest.raises(ValueError,match='Invalid query vector'):
            await module.exact_candidates(Pages([]),('m',),[[1.,0.],[1.,0.]],dimensions=2,top_k=2,
                deadline=asyncio.get_running_loop().time()+1,check_active=lambda: True)
    asyncio.run(scenario())
