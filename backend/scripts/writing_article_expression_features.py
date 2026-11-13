"""Measure exact sentence expression against its neighbors and document background."""

from __future__ import annotations

import math
import re
from collections import Counter

from app.detectors.text.extract import from_string
from app.detectors.text.segment import segment
from scripts.writing_local_lexical_transfer import COMMON

TOKEN = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*|[,;:()!?—–-]", re.UNICODE)
GRAMMAR = (
    ("article", "a an the this that these those"),
    (
        "aux",
        "am are been being be can could did do does had has have is may might must shall should was were will would",
    ),
    (
        "pronoun",
        "he her hers him his i it its me mine our ours she their theirs them they us we who whom whose you your yours",
    ),
    (
        "join",
        "and as because but even for if neither nor or since so than that though unless until when where whereas whether while yet",
    ),
    (
        "prep",
        "about above across after against around before between both by during from in inside into near of off on onto out outside over through to under upon via with within without",
    ),
    (
        "qualifier",
        "almost also already always any each either enough ever every few just less little many more most much never no not often only other rather same several some still such very",
    ),
)
TAG = {word: label for label, words in GRAMMAR for word in words.split()}
PUNCT = frozenset(",;:()!?—–-")
MIN_BACKGROUND = 6
ALPHA = 0.5


def tokens(text):
    return TOKEN.findall(text.casefold().replace("’", "'"))


def tag(t):
    if t in PUNCT:
        return t
    if t[0].isdigit():
        return "#"
    if t in TAG:
        return TAG[t]
    if t in COMMON:
        return "function"
    return "open"


def counts(line, n):
    full = ["^"] + line + ["$"]
    return Counter(tuple(full[i : i + n]) for i in range(len(full) - n + 1))


def pool_features(sequence):
    base = []
    for row in sequence:
        token = tokens(row.text)
        tags = [tag(t) for t in token]
        fixed = [t if t in COMMON or t in PUNCT else tag(t) for t in token]
        open_runs = []
        run = 0
        for t in tags:
            if t == "open":
                run += 1
                continue
            if run:
                open_runs.append(run)
                run = 0
        if run:
            open_runs.append(run)
        if not tags:
            tags = ["open"]
            fixed = ["open"]
            open_runs = [1]
        n = len(tags)
        # Preserve the documented category order. Content words are represented solely as open-class placeholders.
        base.append(
            (
                tags,
                fixed,
                [
                    sum(t != "open" and t not in PUNCT and t != "#" for t in tags) / n,
                    sum(t in PUNCT for t in tags) / n,
                    sum(t == "#" for t in tags) / n,
                    len(open_runs) / n,
                    max(open_runs, default=0) / n,
                    tags[0] in TAG.values(),
                    sum(t in ("join", "prep") for t in tags) / n,
                ],
            )
        )
    return base


def js(left, right):
    keys = left.keys() | right.keys()
    n, m = sum(left.values()), sum(right.values())
    result = 0.0
    for k in sorted(keys):
        a, b = left[k] / n, right[k] / m
        mid = (a + b) / 2
        if a:
            result += a * math.log(a / mid) / 2
        if b:
            result += b * math.log(b / mid) / 2
    return result


def feature_rows(text):
    units = segment(from_string(text))
    measurements = pool_features(units)
    result = []
    for i in range(1, len(units) - 1):
        left, center, right = units[i - 1 : i + 2]
        if len({x.paragraph for x in (left, center, right)}) != 1:
            continue
        if not (12 <= center.n_words <= 65 and all(7 <= v.n_words <= 80 for v in (left, right))):
            continue
        if not all(u.text.endswith((".", "?", "!")) for u in (left, center, right)):
            continue
        neighbor = [k for k, u in enumerate(units) if abs(k - i) > 1 and 7 <= u.n_words <= 80]
        if len(neighbor) < MIN_BACKGROUND:
            continue
        row = measurements[i]
        bk_tags = Counter()
        bk_grams = Counter()
        bk_fixed = Counter()
        for j in neighbor:
            seq, fix, _ = measurements[j]
            bk_tags.update(seq)
            bk_grams.update(counts(seq, 2))
            bk_fixed.update(counts(fix, 2))
        seq, fix, scalars = row
        near = [measurements[k] for k in (i - 1, i + 1)]
        centered = [
            float(x) - (float(a) + float(b)) / 2
            for x, a, b in zip(scalars, near[0][2], near[1][2], strict=True)
        ]

        def rarity(x, bg):
            n = sum(bg.values())
            return -sum(
                v * math.log((bg[k] + ALPHA) / (n + ALPHA * 1024)) for k, v in sorted(x.items())
            ) / sum(x.values())

        vector = centered + [
            js(Counter(seq), bk_tags),
            rarity(counts(seq, 2), bk_grams),
            rarity(counts(fix, 2), bk_fixed),
            js(Counter(seq), Counter(near[0][0] + near[1][0])),
            math.log1p(center.n_words) - (math.log1p(left.n_words) + math.log1p(right.n_words)) / 2,
        ]
        result.append(
            {
                "idx": i,
                "span": [center.start, center.end],
                "paragraph": center.paragraph,
                "background": len(neighbor),
                "style": vector,
            }
        )
    return units, result
