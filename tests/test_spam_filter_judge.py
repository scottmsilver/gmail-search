"""Unit tests for the pure logic in scripts/spam_filter_judge.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from spam_filter_judge import derive_filtered, hi_capture, is_bulk_labels, parse_grades_str


class TestIsBulkLabels:
    def test_promotions_is_bulk(self):
        assert is_bulk_labels({"INBOX", "CATEGORY_PROMOTIONS"}) is True

    def test_social_is_not_bulk(self):
        # Spec: promotions ONLY — social/updates stay in.
        assert is_bulk_labels({"CATEGORY_SOCIAL"}) is False
        assert is_bulk_labels({"CATEGORY_UPDATES"}) is False

    def test_empty_and_personal(self):
        assert is_bulk_labels(set()) is False
        assert is_bulk_labels(["INBOX", "IMPORTANT"]) is False

    def test_accepts_list(self):
        assert is_bulk_labels(["CATEGORY_PROMOTIONS"]) is True


class TestDeriveFiltered:
    def test_removes_bulk_then_truncates(self):
        ranked = [f"t{i}" for i in range(15)]  # t0..t14
        bulk = {"t1", "t3"}
        out = derive_filtered(ranked, bulk, k=10)
        assert out == ["t0", "t2", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"]
        assert len(out) == 10

    def test_no_bulk_equals_top_k(self):
        ranked = [f"t{i}" for i in range(15)]
        assert derive_filtered(ranked, set(), k=10) == ranked[:10]

    def test_fewer_than_k_survivors(self):
        ranked = ["a", "b", "c"]
        assert derive_filtered(ranked, {"b"}, k=10) == ["a", "c"]

    def test_preserves_order(self):
        ranked = ["z", "a", "m"]
        assert derive_filtered(ranked, set(), k=2) == ["z", "a"]


class TestParseGradesStr:
    def test_parses_hex_thread_ids(self):
        text = "18c2ab34ff 3\n19d0e0aa01 0\n1a1111beef 2"
        expected = {"18c2ab34ff", "19d0e0aa01", "1a1111beef"}
        assert parse_grades_str(text, expected) == {
            "18c2ab34ff": 3,
            "19d0e0aa01": 0,
            "1a1111beef": 2,
        }

    def test_ignores_unexpected_ids_and_junk(self):
        text = "unknown99 3\nnot a grade line\n18c2 5\n18c2 2"
        assert parse_grades_str(text, {"18c2"}) == {"18c2": 2}  # 5 out of range, unknown dropped

    def test_empty(self):
        assert parse_grades_str("", {"x"}) == {}


class TestHiCapture:
    def test_fraction_of_grade3_in_topk(self):
        grades = {"a": 3, "b": 3, "c": 2, "d": 0}
        assert hi_capture(["a", "c", "d"], grades, k=3) == 0.5  # a in, b out

    def test_none_when_no_grade3(self):
        assert hi_capture(["a"], {"a": 2}, k=10) is None

    def test_full_capture(self):
        assert hi_capture(["a", "b"], {"a": 3, "b": 3}, k=10) == 1.0
