"""Deterministic, local HTML-to-Markdown rendering for agent mail reads.

This is a readable projection, not a summary or a lossless HTML round-trip.
Links retain their labels; destinations are included only for absolute HTTP(S)
URLs and mailto addresses of at most 300 characters without whitespace/control
characters. Longer and unsafe destinations are omitted, never truncated into a
misleading URL. Images retain non-tracking alt text, never their source URLs.
Stored originals remain the authority for omitted markup and link destinations.
"""

from __future__ import annotations

import re
from urllib.parse import quote, urlsplit

from bs4 import BeautifulSoup, Comment, NavigableString, Tag


_BLOCKS = frozenset({
    'address', 'article', 'aside', 'center', 'dd', 'div', 'dl', 'dt',
    'fieldset', 'figcaption', 'figure', 'footer', 'form', 'header', 'main',
    'nav', 'p', 'section', 'summary', 'ul', 'ol', 'table', 'thead', 'tbody', 'tfoot',
})


def _link_target(value: str) -> str | None:
    if len(value) > 300 or any(char.isspace() or ord(char) < 32 for char in value):
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme.lower() in {'http', 'https'} and parsed.netloc:
        return quote(value, safe=':/?#[]@!$&\'*,;=+%-._~')
    if parsed.scheme.lower() == 'mailto' and parsed.path:
        return quote(value, safe=':@?&=+%-._~')
    return None


def _tracking_image(node: Tag) -> bool:
    for dimension in ('width', 'height'):
        value = str(node.get(dimension, '')).strip().lower().removesuffix('px')
        try:
            if value and float(value) <= 1:
                return True
        except ValueError:
            pass
    style = re.sub(r'\s+', '', str(node.get('style', '')).lower())
    return ('display:none' in style or 'visibility:hidden' in style
            or bool(re.search(r'(?:^|;)(?:width|height):[01](?:px)?(?:;|$)', style)))


def _render(node: Tag | NavigableString) -> str:
    if isinstance(node, Comment):
        return ''
    if isinstance(node, NavigableString):
        return re.sub(r'\s+', ' ', str(node))
    name = node.name
    if name == 'img':
        return '' if _tracking_image(node) else str(node.get('alt', '')).strip()
    if name == 'br':
        return '\n'
    if name == 'hr':
        return '\n\n---\n\n'
    if name == 'pre':
        return '\n\n' + node.get_text().strip('\n') + '\n\n'
    content = ''.join(_render(child) for child in node.children)
    if name == 'a':
        label = content.strip()
        target = _link_target(str(node.get('href', '')))
        if label and target:
            label = label.replace('\\', '\\\\').replace('[', '\\[').replace(']', '\\]')
            return f'[{label}]({target})'
        return content
    if name == 'blockquote':
        return '\n\n' + '\n'.join('> ' + line for line in content.strip().splitlines()) + '\n\n'
    if name in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
        return '\n\n' + '#' * int(name[1]) + ' ' + content.strip() + '\n\n'
    if name == 'li':
        return '\n- ' + content.strip() + '\n'
    if name == 'tr':
        cells = node.find_all(['td', 'th'], recursive=False)
        if cells and not node.find('table'):
            rendered = [' '.join(_render(cell).split()) for cell in cells]
            return '\n| ' + ' | '.join(cell.replace('|', '\\|') for cell in rendered) + ' |\n'
        return '\n' + content.strip() + '\n'
    if name in {'td', 'th'}:
        return ' ' + content.strip() + ' '
    if name in _BLOCKS:
        return '\n\n' + content.strip() + '\n\n'
    return content


def html_to_markdown(html: str) -> str:
    """Render mail facts in source order, retaining table rows and quotations."""
    soup = BeautifulSoup(html, 'html.parser')
    for node in soup.find_all(['head', 'script', 'style', 'template']):
        node.decompose()
    rendered = _render(soup).replace('\xa0', ' ')
    rendered = re.sub(r'[ \t]+\n', '\n', rendered)
    rendered = re.sub(r'\n[ \t]+', '\n', rendered)
    return re.sub(r'\n{3,}', '\n\n', rendered).strip()


def readable_body(body_text: str, body_html: str) -> tuple[str, str]:
    """Prefer rendered HTML; preserve exact plain text when HTML is unusable."""
    if body_html:
        try:
            rendered = html_to_markdown(body_html)
            if rendered:
                return rendered, 'html'
        except Exception:
            # Rendering must never prevent access to the original plain body.
            pass
    return body_text, 'text'
