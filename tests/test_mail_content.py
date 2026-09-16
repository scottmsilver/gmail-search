"""Readable mail keeps source facts while discarding HTML presentation."""

import pytest

from gmail_search.agents.mail_content import html_to_markdown, readable_body


def test_receipt_rows_keep_flight_date_and_amount_together():
    rendered = html_to_markdown('''<html><head><style>noise</style></head><body>
      <h1>Receipt</h1><table><tr><th>Flight</th><th>Date</th><th>Total</th></tr>
      <tr><td>UA 123</td><td>October 4, 2026</td><td>$1,234.56</td></tr>
      <tr><td>UA 456</td><td>October 8, 2026</td><td>$45.00</td></tr></table>
      <script>tracking()</script></body></html>''')
    lines = rendered.splitlines()
    assert any('UA 123' in line and 'October 4, 2026' in line and '$1,234.56' in line for line in lines)
    assert any('UA 456' in line and 'October 8, 2026' in line and '$45.00' in line for line in lines)
    assert '# Receipt' in rendered
    assert 'noise' not in rendered and 'tracking' not in rendered


def test_nested_layout_tables_do_not_duplicate_receipts():
    rendered = html_to_markdown('''<table><tr><td>Booking ABC123<table>
      <tr><td>October 4</td><td>Flight UA 123</td></tr>
      <tr><td>October 8</td><td>Flight UA 456</td></tr>
      </table>Paid $300</td></tr></table>''')
    for fact in ['Booking ABC123', 'October 4', 'October 8', 'Flight UA 123', 'Flight UA 456', 'Paid $300']:
        assert rendered.count(fact) == 1
    assert any('October 4' in line and 'Flight UA 123' in line for line in rendered.splitlines())


def test_quoted_reply_and_correction_remain_distinct():
    rendered = html_to_markdown('<p>Correction: depart October 5, not October 4.</p>'
                                '<blockquote><p>Original: depart October 4.</p>'
                                '<blockquote>Earlier: October 3.</blockquote></blockquote>')
    assert 'Correction: depart October 5, not October 4.' in rendered
    assert '> Original: depart October 4.' in rendered
    assert '> > Earlier: October 3.' in rendered


def test_links_and_image_alt_preserved_without_tracking_markup():
    rendered = html_to_markdown('<p><a href="https://example.com/receipt">View receipt</a></p>'
                                '<img src="cid:ticket" alt="Confirmation ABC123">'
                                '<img width="1" height="1" src="https://tracker.example/pixel" alt="tracking">')
    assert '[View receipt](https://example.com/receipt)' in rendered
    assert 'Confirmation ABC123' in rendered
    assert 'tracker.example' not in rendered and 'tracking' not in rendered


@pytest.mark.parametrize('url', ['javascript:alert(1)', 'data:text/html,payload', '//tracker.example', 'https://example.com/' + 'x' * 2000])
def test_unsafe_or_giant_link_targets_keep_labels_only(url):
    assert html_to_markdown(f'<a href="{url}">View receipt</a>') == 'View receipt'


def test_markdown_link_target_and_label_cannot_escape_link():
    rendered = html_to_markdown('<a href="https://example.com/a)b">a]b</a>')
    assert rendered == '[a\\]b](https://example.com/a%29b)'


def test_lists_breaks_entities_and_malformed_html():
    rendered = html_to_markdown('<p>Receipt&nbsp;&amp; itinerary<br>ABC123<ul><li>Flight 123<li>$20.00</ul>')
    assert 'Receipt & itinerary' in rendered
    assert '\nABC123' in rendered
    assert '- Flight 123' in rendered and '- $20.00' in rendered


def test_html_preferred_and_plaintext_preserved_exactly_when_needed():
    assert readable_body('Plain summary', '<p>Complete receipt $99.00</p>') == ('Complete receipt $99.00', 'html')
    plain = '  flight\t123\n\n $99.00  \n'
    assert readable_body(plain, '') == (plain, 'text')
    assert readable_body(plain, '<html><style>noise</style><script>noise</script></html>') == (plain, 'text')
    assert readable_body('', '') == ('', 'text')


def test_plaintext_fallback_on_conversion_error(monkeypatch):
    def broken(_html):
        raise ValueError('Malformed source')
    monkeypatch.setattr('gmail_search.agents.mail_content.html_to_markdown', broken)
    assert readable_body('Original text', '<p>broken</p>') == ('Original text', 'text')
