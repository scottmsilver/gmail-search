"""Pure preflight geometry/protocol guards; native parsing is tested in the VM."""
import importlib.util
from pathlib import Path
import json
import sys
import types

import pytest

spec=importlib.util.spec_from_file_location('guest_parser', Path(__file__).resolve().parents[1]/'deploy/public/worker/guest-attachment-parser.py')
guest=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guest)


def test_geometry_rejects_pixel_bombs_before_native_rasterization():
    assert guest.pixel_size(72,72,100)==(100,100)
    for width,height in [(100000,100000),(float('inf'),72),(-1,72),(3000,3000)]:
        with pytest.raises(ValueError):
            guest.pixel_size(width,height,200)


def test_untrusted_options_rejected_without_parser(monkeypatch):
    def forbidden(*args):
        raise AssertionError('native parser called')
    monkeypatch.setattr(guest,'single',forbidden)
    for options in [{'dpi':100,'pages':[],'path':'/etc/passwd'}, {'dpi':100,'pages':[1]*9}, {'dpi':True,'pages':[]}]:
        with pytest.raises(ValueError):
            guest.parse(b'%PDF-x','application/pdf',options)


def test_pdf_text_over_limit_marks_result_truncated_before_slicing(monkeypatch):
    class Page:
        rect = types.SimpleNamespace(width=72, height=72)
        def get_text(self):
            return 'x' * guest.MAX_TEXT + 'y'
        def get_pixmap(self, **kwargs):
            return types.SimpleNamespace(width=100, height=100, tobytes=lambda _: b'png')

    class Doc:
        is_encrypted = False
        def __len__(self): return 1
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getitem__(self, number): return Page()

    monkeypatch.setitem(sys.modules, 'pymupdf', types.SimpleNamespace(open=lambda **kwargs: Doc(), csRGB=object()))
    result = json.loads(guest.parse(b'%PDF-x', 'application/pdf', {'dpi': 100, 'pages': []}))
    assert result['truncated'] is True
    assert len(result['text'].encode()) == guest.MAX_TEXT
