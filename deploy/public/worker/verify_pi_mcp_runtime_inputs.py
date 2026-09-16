#!/usr/bin/env python3
"""Verify qualified public bytes before packing; CLI never accepts new pins.

Tree digest v1: SHA256 of sorted, UTF-8 JSON-array records followed by LF.
Records contain relative POSIX path (root is '.'), kind, executable permission
bits, then file SHA256 or literal relative symlink target. Directories have no
payload. Ownership and write permissions are checked separately so extracting
the qualified read-only image and removing group/world write is permitted.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys


MANIFEST = Path(__file__).with_name('pi-mcp-runtime-inputs.json')
QUALIFIED_IMAGE = '478461755f2f344572bb0784685205fb173795bb5e3acae6fe523fb73dc62bbc'


def trusted_metadata(info, label, *, ancestor=False):
    mode = info.st_mode
    sticky_parent = ancestor and stat.S_ISDIR(mode) and bool(mode & stat.S_ISVTX)
    if (info.st_uid != 0 or (not stat.S_ISLNK(mode) and
            (mode & 0o6000 or (mode & 0o022 and not sticky_parent)))):
        raise ValueError(f'Untrusted owner or permissions: {label}')


def trusted_path(path):
    # Do not resolve aliases first: every ancestor must itself be trusted and
    # must not be a symlink. Root-owned sticky /tmp is safe for root-owned children.
    for current in reversed((path, *path.parents)):
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f'Input path contains a symlink: {current}')
        trusted_metadata(info, current, ancestor=current != path)


def tree_fingerprint(root, *, trusted=True):
    root = Path(os.path.abspath(root))
    if trusted:
        trusted_path(root)
    if root.is_symlink():
        raise ValueError('Input root is a symlink')
    records = []

    def walk(path, relative):
        info = path.lstat()
        if trusted:
            trusted_metadata(info, path)
        mode = info.st_mode
        if stat.S_ISDIR(mode):
            records.append([relative, 'directory', mode & 0o111])
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                walk(child, child.relative_to(root).as_posix())
        elif stat.S_ISREG(mode):
            digest = hashlib.sha256()
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                    raise ValueError('Input changed while hashing')
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            records.append([relative, 'file', mode & 0o111, digest.hexdigest()])
        elif stat.S_ISLNK(mode):
            target = os.readlink(path)
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError('Invalid or dangling symlink') from exc
            if Path(target).is_absolute() or not resolved.is_relative_to(root):
                raise ValueError('Input symlink escapes its verified tree')
            records.append([relative, 'symlink', 0, target])
        else:
            raise ValueError(f'Unsupported input file type: {path}')

    walk(root, '.')
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item[0]):
        digest.update((json.dumps(record, ensure_ascii=True, separators=(',', ':')) + '\n').encode())
    return {'sha256': digest.hexdigest(), 'entries': len(records)}


def fingerprints(runtime, packages, *, trusted=True):
    runtime, packages = Path(runtime), Path(packages)
    return {name: tree_fingerprint(path, trusted=trusted) for name, path in {
        'bin': runtime / 'bin', 'lib': runtime / 'lib',
        'node_modules': packages / 'node_modules',
        'package.json': packages / 'package.json',
        'package-lock.json': packages / 'package-lock.json',
    }.items()}


def verify(runtime, packages, expected, *, trusted=True):
    if fingerprints(runtime, packages, trusted=trusted) != expected:
        raise ValueError('Qualified runtime input fingerprint mismatch')


def snapshot(runtime, packages, destination):
    """Copy content into a fresh private directory without ACLs or xattrs."""
    runtime, packages, destination = Path(runtime), Path(packages), Path(destination)
    destination.mkdir(mode=0o700)

    def copy(source, target):
        info = source.lstat()
        if stat.S_ISLNK(info.st_mode):
            target.symlink_to(os.readlink(source))
        elif stat.S_ISDIR(info.st_mode):
            target.mkdir(mode=0o700)
            for child in source.iterdir():
                copy(child, target / child.name)
            target.chmod(0o644 | (info.st_mode & 0o111))
        elif stat.S_ISREG(info.st_mode):
            shutil.copyfile(source, target, follow_symlinks=False)
            target.chmod(0o644 | (info.st_mode & 0o111))
        else:
            raise ValueError('Unsupported input file type during copy')

    for name in ('bin', 'lib'):
        copy(runtime / name, destination / name)
    (destination / 'pi-pkgs').mkdir(mode=0o755)
    for name in ('node_modules', 'package.json', 'package-lock.json'):
        copy(packages / name, destination / 'pi-pkgs' / name)


def main():
    if os.geteuid() != 0:
        raise ValueError('Builder requires root-owned trusted staging and execution')
    if len(sys.argv) != 4:
        raise ValueError('Usage: verify_pi_mcp_runtime_inputs.py RUNTIME PACKAGES NEW_SNAPSHOT')
    manifest = json.loads(MANIFEST.read_text())
    if manifest['format'] != 'sha256-tree-v1' or manifest['qualified_image_sha256'] != QUALIFIED_IMAGE:
        raise ValueError('Unsupported public runtime input manifest')
    runtime, packages, destination = (Path(os.path.abspath(value)) for value in sys.argv[1:])
    trusted_path(destination.parent)
    verify(runtime, packages, manifest['trees'])
    snapshot(runtime, packages, destination)
    # Only this verified private copy reaches mksquashfs. No source is copied
    # again afterwards, closing the verification-to-packing gap.
    verify(destination, destination / 'pi-pkgs', manifest['trees'])
    print('Qualified public runtime inputs and private snapshot verified')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
