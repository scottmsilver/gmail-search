#!/usr/bin/env python3
"""Fixed one-job guest entrypoint. No credentials, paths, network or agent runtime."""
import base64
import io
import json
import math
import os
import resource
import socket
import struct
import sys
import time
import warnings
import zipfile

INPUT = 10*1024**2
OUTPUT = 8*1024**2
PIXELS = 4_000_000
MAX_TEXT = 200_000
MIMES = {'application/pdf', 'application/zip', 'image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/tiff'}


def pixel_size(width, height, dpi):
    if not all(math.isfinite(x) and x > 0 for x in (width, height)):
        raise ValueError('geometry')
    width, height = math.ceil(width*dpi/72), math.ceil(height*dpi/72)
    if width > 4000 or height > 4000 or width*height > PIXELS:
        raise ValueError('pixels')
    return width, height


def png_page(number, width, height, png):
    if not 0 < width <= 4000 or not 0 < height <= 4000 or width*height > PIXELS or len(png) > OUTPUT//2:
        raise ValueError('pixels')
    return dict(number=number, width=width, height=height, png=base64.b64encode(png).decode('ascii'))


def single(data, mime, options):
    if mime == 'application/pdf':
        import pymupdf
        with pymupdf.open(stream=data, filetype='pdf') as doc:
            if doc.is_encrypted or len(doc) > 100000:
                raise ValueError('document')
            requested = options['pages'] or list(range(1, min(len(doc), 8)+1))
            if any(number > len(doc) for number in requested):
                raise ValueError('page')
            pages, texts = [], []
            for number in requested:
                page = doc[number-1]
                pixel_size(page.rect.width, page.rect.height, options['dpi'])
                texts.append(page.get_text())
                pix = page.get_pixmap(dpi=options['dpi'], colorspace=pymupdf.csRGB, alpha=False)
                pages.append(png_page(number, pix.width, pix.height, pix.tobytes('png')))
            text = '\n'.join(texts).encode('utf-8')
            return dict(text=text[:MAX_TEXT].decode('utf-8', errors='ignore'), pages=pages,
                        truncated=len(text)>MAX_TEXT or len(requested)<len(doc))
    if mime.startswith('image/'):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = PIXELS
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
            if width*height > PIXELS or max(width, height)>4000:
                raise ValueError('pixels')
            im.load()
            output = io.BytesIO()
            im.convert('RGB').save(output, format='PNG')
            return dict(text='', pages=[png_page(1, width, height, output.getvalue())],
                        truncated=getattr(im, 'n_frames', 1)>1)
    raise ValueError('mime')


def parse(data, mime, options):
    if (mime not in MIMES or not 0 < len(data) <= INPUT or type(options) is not dict
            or set(options) != {'dpi', 'pages'} or type(options['dpi']) is not int or not 72 <= options['dpi'] <= 200
            or type(options['pages']) is not list or len(options['pages']) > 8
            or any(type(n) is not int or not 1 <= n <= 100000 for n in options['pages'])
            or len(set(options['pages'])) != len(options['pages'])):
        raise ValueError('request')
    if mime != 'application/zip':
        result = single(data, mime, options)
    else:
        result = dict(text='', pages=[], truncated=False)
        expanded = 0
        suffixes = {'.pdf':'application/pdf', '.png':'image/png', '.jpg':'image/jpeg', '.jpeg':'image/jpeg',
                    '.gif':'image/gif', '.webp':'image/webp', '.tif':'image/tiff', '.tiff':'image/tiff'}
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            result['truncated'] = len(entries)>20
            for info in entries[:20]:
                if info.is_dir():
                    continue
                # Archive names never become filesystem paths or manifest fields.
                mime = suffixes.get(os.path.splitext(info.filename)[1].lower())
                if not mime:
                    result['truncated'] = True
                    continue
                if info.file_size > INPUT or expanded+info.file_size > 20*1024**2:
                    raise ValueError('expanded')
                with archive.open(info) as member:
                    payload = member.read(INPUT+1)
                if len(payload)>INPUT:
                    raise ValueError('expanded')
                expanded += len(payload)
                item = single(payload, mime, options)
                result['text'] += item['text']
                result['pages'].extend(item['pages'])
                result['truncated'] |= item['truncated']
                if len(result['pages'])>8 or len(result['text'].encode())>MAX_TEXT:
                    raise ValueError('result')
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    if len(encoded)>OUTPUT:
        raise ValueError('result')
    return encoded


def exact(sock, size):
    result = bytearray()
    while len(result)<size:
        part = sock.recv(min(65536, size-len(result)))
        if not part:
            raise ValueError('eof')
        result.extend(part)
    return bytes(result)


def main():
    os.umask(0o077)
    os.environ.clear()
    os.environ.update(HOME='/tmp', PATH='/usr/bin:/bin', LANG='C.UTF-8')
    os.chdir('/tmp')
    os.setgroups([])
    os.setgid(1000)
    os.setuid(1000)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64,64))
    resource.setrlimit(resource.RLIMIT_NPROC, (32,32))
    resource.setrlimit(resource.RLIMIT_FSIZE, (INPUT,INPUT))
    sys.path.insert(0, '/tmp/parser/site-packages')
    peer = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    peer.settimeout(30)
    until = time.monotonic()+15
    while True:
        try:
            peer.connect((2,8002))
            break
        except OSError:
            if time.monotonic()>until:
                raise
            time.sleep(.1)
    with peer:
        size = struct.unpack('!I', exact(peer,4))[0]
        if size>1024:
            return
        header = json.loads(exact(peer,size))
        if (type(header) is not dict or set(header)!={'mime_type','options','size'}
                or type(header['size']) is not int or not 0<header['size']<=INPUT):
            return
        data = exact(peer,header['size'])
        try:
            result = parse(data, header['mime_type'], header['options'])
        except Exception:
            result = b'{"error":"rejected"}'
        peer.sendall(struct.pack('!I',len(result))+result)
    print('ATTACHMENT_JOB_DONE', flush=True)


if __name__ == '__main__':
    main()
