"""Stylometric tells of LLM prose: lexicon, 'not X but Y' constructions, transition-starter density,
hedges, triads, sentence-rhythm uniformity (burstiness) and lexical diversity (MATTR).

These are *style* evidence — fast, transparent, and explainable sentence by sentence — but a skilled
human writer can trip them and a prompted model can avoid them, so they carry modest weight.
"""
from __future__ import annotations

import re
from statistics import mean, pstdev

LEXICON: list[tuple[str, int, str]] = [
    # tier 1: strong tells
    (r"\bdelv(?:e|es|ed|ing)\b", 1, "delve"), (r"\btapestr(?:y|ies)\b", 1, "tapestry"), (r"\bmultifaceted\b", 1, "multifaceted"),
    (r"\ba testament to\b", 1, "a testament to"), (r"\bboasts?\b(?!\s+(?:about|of)\b)", 1, "boasts"), (r"\bbustling\b", 1, "bustling"),
    (r"\brealm(?:s)? of\b", 1, "realm of"), (r"\bseamless(?:ly)?\b", 1, "seamless"), (r"\bmeticulous(?:ly)?\b", 1, "meticulous"),
    (r"\bshowcas(?:e|es|ed|ing)\b", 1, "showcase"), (r"\bmyriad\b", 1, "myriad"), (r"\bplethora\b", 1, "plethora"),
    (r"\bembark(?:s|ed|ing)?\b", 1, "embark"), (r"\bbeacon\b", 1, "beacon"), (r"\bspearhead(?:s|ed|ing)?\b", 1, "spearhead"),
    (r"\bgame[- ]chang(?:er|ing)\b", 1, "game-changer"), (r"\bcutting[- ]edge\b", 1, "cutting-edge"), (r"\bever[- ]evolving\b", 1, "ever-evolving"),
    (r"\bunwavering\b", 1, "unwavering"), (r"\bunparalleled\b", 1, "unparalleled"), (r"\brevolutioniz(?:e|es|ed|ing)\b", 1, "revolutionize"),
    (r"\bstate[- ]of[- ]the[- ]art\b", 1, "state-of-the-art"), (r"\btreasure trove\b", 1, "treasure trove"), (r"\bdouble[- ]edged sword\b", 1, "double-edged sword"),
    (r"\bactionable insights?\b", 1, "actionable insights"), (r"\bit(?:'|’)?s (?:important|worth) (?:to note|noting)\b", 1, "it's important to note"),
    (r"\bit is (?:important|worth) (?:to note|noting)\b", 1, "it is important to note"), (r"\bneedless to say\b", 1, "needless to say"),
    (r"\bin today(?:'|’)?s (?:fast[- ]paced|digital|dynamic|ever[- ]changing|rapidly|modern|competitive|data[- ]driven|business)\b", 1, "in today's fast-paced…"),
    (r"\bin the (?:world|realm|space|landscape) of\b", 1, "in the world/realm of"), (r"\bplays? a (?:crucial|vital|key|pivotal|significant) role\b", 1, "plays a crucial role"),
    (r"\blet(?:'|’)?s (?:dive|delve|unpack|explore)\b", 1, "let's dive/unpack"), (r"\bin an era (?:of|where)\b", 1, "in an era of"),
    (r"\bwhether you(?:'|’)?re a\b", 1, "whether you're a…"), (r"\blook no further\b", 1, "look no further"), (r"\bwithout further ado\b", 1, "without further ado"),
    (r"\bunlock(?:s|ed|ing)? the (?:full )?(?:potential|power|value)\b", 1, "unlock the potential"), (r"\belephant in the room\b", 1, "elephant in the room"),
    (r"\bgroundbreaking\b", 1, "groundbreaking"), (r"\bpivotal\b", 1, "pivotal"), (r"\bintricate\b|\bintricacies\b", 1, "intricate"),
    # tier 2: moderate
    (r"\bfoster(?:s|ed|ing)?\b", 2, "foster"), (r"\bvibrant\b", 2, "vibrant"), (r"\bunderscor(?:e|es|ed|ing)\b", 2, "underscore"),
    (r"\bharness(?:es|ed|ing)?\b", 2, "harness"), (r"\bempower(?:s|ed|ing|ment)?\b", 2, "empower"), (r"\belevate (?:your|the|our)\b", 2, "elevate"),
    (r"\btransformative\b", 2, "transformative"), (r"\bsynerg(?:y|ies|istic)\b", 2, "synergy"), (r"\bparadigm(?:s)?\b", 2, "paradigm"),
    (r"\bcrucial\b", 2, "crucial"), (r"\bvital\b", 2, "vital"), (r"\bmoreover\b", 2, "moreover"), (r"\bfurthermore\b", 2, "furthermore"),
    (r"\bthe (?:\w+ )?landscape\b", 2, "the … landscape"), (r"\b(?:our|the|this|a) journey\b", 2, "journey (metaphor)"), (r"\bserves? as an?\b", 2, "serves as a"),
    (r"\ba wide (?:range|array|variety) of\b", 2, "a wide range of"), (r"\bwhen it comes to\b", 2, "when it comes to"), (r"\bat its core\b", 2, "at its core"),
    (r"\bin essence\b", 2, "in essence"), (r"\bfirst and foremost\b", 2, "first and foremost"), (r"\blast but not least\b", 2, "last but not least"),
    (r"\bdiv(?:e|es|ing) (?:in|into|deeper)\b", 2, "dive into"), (r"\bkey takeaways?\b", 2, "key takeaways"), (r"\bin conclusion\b", 2, "in conclusion"),
    (r"\bcommendable\b", 2, "commendable"), (r"\binvaluable\b", 2, "invaluable"), (r"\bnavigat(?:e|es|ed|ing) the\b", 2, "navigate the…"),
    # tier 3: weak
    (r"\bleverag(?:e|es|ed|ing)\b(?!\s+ratios?)", 3, "leverage"), (r"\bcomprehensive\b(?!\s+income\b)", 3, "comprehensive"), (r"\bholistic(?:ally)?\b", 3, "holistic"),
    (r"\brobust\b", 3, "robust"), (r"\bstreamlin(?:e|es|ed|ing)\b", 3, "streamline"), (r"\butiliz(?:e|es|ed|ing|ation)\b", 3, "utilize"),
    (r"\bfacilitat(?:e|es|ed|ing)\b", 3, "facilitate"), (r"\badditionally\b", 3, "additionally"), (r"\bnotably\b", 3, "notably"), (r"\bultimately\b", 3, "ultimately"),
    (r"\bessentially\b", 3, "essentially"), (r"\bdeep dive\b", 3, "deep dive"), (r"\bin summary\b|\bto summarize\b", 3, "in summary"), (r"\bin order to\b", 3, "in order to"),
    (r"\bensur(?:e|es|ing) that\b", 3, "ensure that"), (r"\bsignificant(?:ly)?\b", 3, "significant"), (r"\bvarious\b", 3, "various"), (r"\boverall\b", 3, "overall"),
]
TIER_WEIGHTS = {1: 3.0, 2: 1.5, 3: 0.6}
CONTRAST = [
    (r"\bnot (?:just|only|merely|simply)\b[^.!?\n]{0,80}?\b(?:but|rather)\b", "not just X but Y"),
    (r"\bit(?:'|’)?s not (?:just |only |merely |simply |about )?[^.!?\n]{1,60}[.;,—–] *(?:it(?:'|’)?s|it is|this is)\b", "it's not X, it's Y"),
    (r"\bmore than just\b", "more than just"), (r"\bnot because\b[^.!?\n]{1,60}\bbut because\b", "not because X but because Y"),
    (r"\bthis (?:isn(?:'|’)?t|is not) about\b", "this isn't about X"), (r"\bnot only\b[^.!?\n]{0,80}?\bbut(?: also)?\b", "not only X but also Y"),
]
HEDGES = [r"\barguably\b", r"\bundoubtedly\b", r"\bcertainly\b", r"\bone might argue\b", r"\bsome (?:might|may|would) (?:argue|say)\b",
          r"\bit (?:seems|appears) that\b", r"\bto some extent\b", r"\bin many ways\b", r"\b(?:generally|broadly) speaking\b", r"\bit could be argued\b", r"\bit is worth considering\b"]
TRANSITION = re.compile(r"^(?:Additionally|Furthermore|Moreover|Notably|Importantly|Interestingly|Crucially|Ultimately|Essentially|Overall|Consequently|Subsequently|Similarly|Likewise|Indeed|Firstly|Secondly|Thirdly|Lastly|In conclusion|In summary)\b[,:]?", re.IGNORECASE)
TRIAD = re.compile(r"\b[a-z]+, [a-z]+, and [a-z]+\b", re.IGNORECASE)
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")

_LEX = [(re.compile(p, re.IGNORECASE), t, lab) for p, t, lab in LEXICON]
_CON = [(re.compile(p, re.IGNORECASE), lab) for p, lab in CONTRAST]
_HED = [re.compile(p, re.IGNORECASE) for p in HEDGES]


def sentence_flags(text: str) -> list[str]:
    flags: list[str] = []
    for rx, tier, lab in _LEX:
        if rx.search(text):
            flags.append(f"lexicon T{tier}: “{lab}”")
    for rx, lab in _CON:
        if rx.search(text):
            flags.append(f"contrast construction: {lab}")
    if TRANSITION.match(text.strip()):
        flags.append("transition-word opener")
    if TRIAD.search(text):
        flags.append("rule-of-three list")
    if text.count("—") >= 2:
        flags.append("multiple em-dashes")
    return flags


def sentence_style_probability(flags: list[str]) -> float | None:
    """0.5 baseline; each tell nudges toward AI; capped. None when no tells (uninformative)."""
    if not flags:
        return None
    score = 0.5
    for f in flags:
        if "T1" in f:
            score += 0.14
        elif "T2" in f or "contrast" in f:
            score += 0.08
        else:
            score += 0.04
    return min(score, 0.85)


def _mattr(words: list[str], window: int = 50) -> float | None:
    if len(words) < window:
        return None
    vals = [len(set(words[i : i + window])) / window for i in range(0, len(words) - window + 1, max(1, window // 5))]
    return float(mean(vals))


def document_metrics(sentences: list[str]) -> dict:
    text = " ".join(sentences)
    words = [w.lower() for w in WORD.findall(text)]
    n = max(len(words), 1)
    per1000 = 1000 / n
    lex_hits = {}
    lex_weight = 0.0
    for rx, tier, lab in _LEX:
        c = len(rx.findall(text))
        if c:
            lex_hits[lab] = c
            lex_weight += c * TIER_WEIGHTS[tier]
    contrast = sum(len(rx.findall(text)) for rx, _ in _CON)
    hedges = sum(len(rx.findall(text)) for rx in _HED)
    lens = [len(WORD.findall(s)) for s in sentences if WORD.findall(s)]
    cv = (pstdev(lens) / mean(lens)) if len(lens) >= 2 and mean(lens) > 0 else None
    transitions = sum(1 for s in sentences if TRANSITION.match(s.strip()))
    triads = len(TRIAD.findall(text))
    return {
        "words": len(words), "sentences": len(sentences),
        "mean_sentence_len": round(mean(lens), 1) if lens else 0, "sentence_len_cv": round(cv, 3) if cv is not None else None,
        "mattr": round(_mattr(words), 3) if _mattr(words) is not None else None,
        "lexicon_weighted_per_1000": round(lex_weight * per1000, 2), "lexicon_hits": dict(sorted(lex_hits.items(), key=lambda kv: -kv[1])[:12]),
        "contrast_per_1000": round(contrast * per1000, 2), "hedges_per_1000": round(hedges * per1000, 2),
        "transition_opener_pct": round(100 * transitions / max(len(sentences), 1), 1), "triads_per_1000": round(triads * per1000, 2),
        "emdash_per_1000": round(text.count("—") * per1000, 2),
    }


def _ramp(x: float | None, lo: float, hi: float) -> float:
    if x is None:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def document_style_probability(m: dict) -> tuple[float, list[str]]:
    """Combine metrics into a conservative [0.2, 0.8] probability with reasons."""
    reasons: list[str] = []
    parts: list[tuple[float, float]] = []  # (score 0..1, weight)
    s = _ramp(m["lexicon_weighted_per_1000"], 2.0, 14.0)
    parts.append((s, 1.0))
    if s > 0.3:
        top = ", ".join(f"“{k}”×{v}" for k, v in list(m["lexicon_hits"].items())[:5])
        reasons.append(f"LLM-favoured vocabulary density {m['lexicon_weighted_per_1000']}/1000 words ({top})")
    s = _ramp(m["contrast_per_1000"], 1.0, 6.0)
    parts.append((s, 0.7))
    if s > 0.3:
        reasons.append(f"'not X but Y' contrast constructions: {m['contrast_per_1000']}/1000 words")
    s = _ramp(m["transition_opener_pct"], 8.0, 30.0)
    parts.append((s, 0.7))
    if s > 0.3:
        reasons.append(f"{m['transition_opener_pct']}% of sentences open with a discourse transition (Additionally/Moreover/…)")
    if m["sentence_len_cv"] is not None and m["sentences"] >= 8:
        s = 1 - _ramp(m["sentence_len_cv"], 0.30, 0.65)
        parts.append((s, 0.8))
        if s > 0.5:
            reasons.append(f"Uniform sentence rhythm (length CV {m['sentence_len_cv']}; human prose is typically burstier, > 0.5)")
    if m["mattr"] is not None:
        s = 1 - _ramp(m["mattr"], 0.60, 0.78)
        parts.append((s, 0.4))
    s = _ramp(m["triads_per_1000"], 2.0, 9.0)
    parts.append((s, 0.4))
    if s > 0.4:
        reasons.append(f"Rule-of-three lists: {m['triads_per_1000']}/1000 words")
    score = sum(v * w for v, w in parts) / sum(w for _, w in parts)
    p = 0.2 + 0.6 * score
    if not reasons:
        reasons.append("No prominent stylistic tells; vocabulary and rhythm are within human ranges")
    return round(p, 4), reasons
