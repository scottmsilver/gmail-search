#!/usr/bin/env python3
"""Generate known synthetic fixtures; never parse existing documents or mail.

Host PyMuPDF/Pillow are used only for fixture creation. The smoke controller
reads these opaque bytes; all adversarial parsing happens inside inner VMs.
"""
from pathlib import Path
import struct
import tempfile
import zipfile
import zlib

import fitz
from PIL import Image

root=Path(tempfile.mkdtemp(prefix='gms-attachment-fixtures.'))
for name,size in [('hello.pdf',(300,300)),('huge.pdf',(100000,100000))]:
    with fitz.open() as doc:
        page=doc.new_page(width=size[0],height=size[1])
        page.insert_text((20,30),'SYNTHETIC ATTACHMENT 42')
        doc.save(root/name)
with fitz.open() as doc:
    page=doc.new_page(width=300,height=300)
    page.insert_text((1,2),'A'*200001,fontsize=.001)
    doc.save(root/'long-text.pdf',deflate=True)
(root/'malformed.pdf').write_bytes(b'%PDF-1.7\nmalformed synthetic input')
Image.new('RGB',(8,8),(20,40,60)).save(root/'tiny.png')
data=(root/'tiny.png').read_bytes()
ihdr=struct.pack('!II',100000,100000)+data[24:29]
(root/'huge.png').write_bytes(data[:16]+ihdr+struct.pack('!I',zlib.crc32(b'IHDR'+ihdr))+data[33:])
with zipfile.ZipFile(root/'hello.zip','w') as output:
    output.writestr('../../hello.pdf',(root/'hello.pdf').read_bytes())
with zipfile.ZipFile(root/'many.zip','w') as output:
    for index in range(25):
        output.writestr(str(index)+'.unsupported',b'synthetic')
    output.writestr('past-limit.pdf',(root/'hello.pdf').read_bytes())
print(root)
