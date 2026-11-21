"""Extract the MOT17-10-FRCNN development sequence via HTTP byte ranges."""

import argparse
import io
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

URL = 'https://motchallenge.net/data/MOT17.zip'
DATA = Path(__file__).resolve().parents[1] / 'data' / 'p2' / 'gt'


class RemoteZipReader(io.RawIOBase):
    """Seek through the archive while downloading only the required intervals."""

    def __init__(self):
        with urlopen(Request(URL, method='HEAD'), timeout=30) as response:
            self.size = int(response.headers['Content-Length'])
            self.etag = response.headers['ETag']
        self.position = 0
        self.start = 0
        self.buffer = b''

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origins = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        self.position = origins[whence] + offset
        if self.position < 0:
            raise OSError('Negative archive offset')
        return self.position

    def read(self, size=-1):
        remaining = max(0, self.size - self.position)
        size = remaining if size < 0 else min(size, remaining)
        if not size:
            return b''
        end = self.position + size
        if not self.start <= self.position or end > self.start + len(self.buffer):
            stop = min(self.size, max(end, self.position + 16 * 1024**2)) - 1
            headers = {'Range': f'bytes={self.position}-{stop}', 'If-Match': self.etag}
            with urlopen(Request(URL, headers=headers), timeout=60) as response:
                expected = f'bytes {self.position}-{stop}/{self.size}'
                if response.status != 206 or response.headers['Content-Range'] != expected:
                    raise OSError(f'Expected byte range {expected}')
                self.buffer = response.read()
            if len(self.buffer) != stop - self.position + 1:
                raise OSError('Incomplete archive range')
            self.start = self.position
        data = self.buffer[self.position - self.start:end - self.start]
        self.position = end
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', choices=tuple(f'MOT17-{i:02d}-FRCNN' for i in (2, 4, 5, 9, 10)),
                        default='MOT17-10-FRCNN')
    prefix = f'MOT17/train/{parser.parse_args().sequence}/'
    with RemoteZipReader() as reader, ZipFile(reader) as archive:
        members = [member for member in archive.infolist()
                   if member.filename.startswith(prefix) and not member.is_dir()]
        if not members:
            raise RuntimeError(f'No files found under {prefix}')
        for member in sorted(members, key=lambda item: item.header_offset):
            target = DATA / member.filename.removeprefix('MOT17/train/')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
        print(f'Extracted {len(members)} files from {prefix} (ZIP CRC verified).')


if __name__ == '__main__':
    main()
