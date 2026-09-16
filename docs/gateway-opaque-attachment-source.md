# Opaque attachment source: raw bytes and parser inputs

Candidate implementation stage; no public download route or runtime change.

## Design

Reuse the existing owner-hashed attachment directory and descriptor-relative
reader for raw bytes. A raw read can return any bounded media type and an empty
file. It must not imply that the isolated parser accepts that format. Keep the
existing parser-facing locator and load method restricted to qualified media
types and nonempty files. A separate frozen raw result type prevents accidental
substitution for the exact parser input type.

The trusted raw locator derives paths from owner, message and filename metadata
using the existing restricted reader. No raw_path grant, fallback to shared
legacy paths, caller path or MIME selection is introduced. Bind the declared
size to the opened file. Both paths require regular single-link files, reject
symlinks and traversal, read at most 10 MiB, detect changes during reading, and
close held descriptors on cancellation. Media labels are bounded ASCII tokens;
missing labels on raw files use application/octet-stream. No host decoding.

## Implementation sequence

1. Add failing generic/empty raw, parser rejection, size mismatch and owner/path
   regression tests.
2. Add distinct raw result and raw locator/load entry points over shared checks.
3. Rerun existing parser-source and real owner locator tests, then independently
   review. Production resources remain untouched.

Authenticated binary transfer into a private guest workspace, capability
configuration, model-facing tool mapping and parser format/paging qualification
are subsequent stages. These source methods alone do not provide download or
full attachment-tool parity.

## Candidate verification

Five new raw-entry-point tests failed before implementation. The source and real
owner locator suite now passes 37 tests, including generic/empty files,
metadata-size mismatch, both modes' owner/path/link protections, changed-file
and cancellation descriptor cleanup, and MIME/size rejection before os.open.
Independent review found no implementation blocker and requested the stronger
before-open assertions, which were added. The broader source/locator/parser
service/HTTP run passed 44 tests before the final five size cases were added.
These counts overlap; no real mail or production filesystem was used.
