from types import SimpleNamespace

from scripts.grid_phone_alignment_v162 import copy_alignment


def test_children_are_copied_before_native_word_iterator_advances():
    class BorrowedWord:
        def __init__(self, name, start, phone):
            self.name, self.start, self.duration = name, start, 6
            self.phone = SimpleNamespace(name=phone, start=start, duration=6)
            self.valid = True

        def __iter__(self):
            if not self.valid:
                raise RuntimeError("Native word iterator already advanced")
            yield self.phone

    def native_alignment():
        for name, start, phone in (("bin", 1, "B"), ("<sil>", 7, "SIL"), ("blue", 13, "B")):
            word = BorrowedWord(name, start, phone)
            yield word
            word.valid = False

    words, phones = copy_alignment(native_alignment(), 16_000)
    assert words == [("bin", 1, 7), ("blue", 13, 19)]
    assert phones == [
        {"phone": "B", "sample_range": [160, 1120], "release_sample": None},
        {"phone": "B", "sample_range": [2080, 3040], "release_sample": None},
    ]
