"""Source-equivalent64MiB HTTP block cache; preserve the original reader."""

from collections import OrderedDict

from scripts.ranged_gzip_v21 import RangeFile


class CachedRangeFile(RangeFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = OrderedDict()
        self.cache_hits = 0
        self.cache_limit = 64 * 1024**2

    def _block(self, start):
        if start in self.blocks:
            self.cached_start, self.cached = start, self.blocks.pop(start)
            self.blocks[start] = self.cached
            self.cache_hits += 1
            return
        super()._block(start)
        self.blocks[start] = self.cached
        while sum(len(v) for v in self.blocks.values()) > self.cache_limit:
            self.blocks.popitem(last=False)
