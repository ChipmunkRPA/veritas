"""Prepare 24 new stock-voice utterances using the qualified v240 renderer."""

from __future__ import annotations

import argparse
import json

from scripts import audio_okur_turkish_v240 as previous
from scripts.video_appearance_corrected_v112 import scoped

ROOT = previous.ROOT
MODULE = "scripts.audio_okur_turkish_followup_v245"
TRIAL = previous.TRIAL.with_name("audio-okur-turkish-followup-v245")
STAGE = TRIAL / "authored"
OUTPUT = STAGE / "output"
PUBLIC = previous.PUBLIC.with_name("audio-okur-turkish-followup-freeze-v245.json")
PREVIOUS_RESULT = previous.PUBLIC.with_name("audio-okur-turkish-complete-v240.json")
PREVIOUS_FREEZE_SHA256 = "73fdfbbe4e7d081ed85119accfc37ddd2cbefeece5d0eae495c178cfc5209549"
PREVIOUS_RESULT_SHA256 = "37e697d698ff4be034aa8631505c081e83185aa2a39e1ed32f8cf9ae1e11b6f8"
PREVIOUS_JOURNAL = previous.OUTPUT / "private-cases.jsonl"
PHRASES = (
    "ince bulutlar dağın üzerinden sessizce geçip gitti.",
    "küçük serçeler çitin üstüne konup neşeyle öttüler.",
    "durakta bekleyen yolcular kısa bir süre sonra dağıldı.",
    "kumdaki ayak izlerini kıyıya gelen dalgalar sildi.",
    "koridordaki uzun masanın üstünde açık bir kitap durdu.",
    "arka bahçede büyüyen ağaç yaz sonunda meyve verdi.",
    "köy meydanında yaşlı bir adam uzunca süre bekledi.",
    "yuvarlak taşın altında minik bir kertenkele gizlendi.",
    "karşı apartmanın penceresinde mavi bir ışık yandı.",
    "yemekten sonra herkes mutfağın etrafında biraz oturdu.",
    "dik yokuştan inen bisikletli çocuk dikkatlice durdu.",
    "kütüphaneden aldığım kitabı akşam eve dönünce açtım.",
    "kış gelince tepedeki çamların dalları karla kaplandı.",
    "vapurdan inen yolcular dar sokağa doğru yürüdüler.",
    "iki kardeş sabahtan beri bahçede sessizce oynuyordu.",
    "uzaktaki tepenin ardında güneş ağır ağır kayboldu.",
    "yağmur başlayınca meydandaki satıcı hemen örtüsünü açtı.",
    "su kenarındaki ince kamışlar rüzgârla birlikte sallandı.",
    "çizgili çantanın içinde temiz bir mendil kalmıştı.",
    "çocuklar ormanın içinden gelen sesi dikkatle dinledi.",
    "kalın perdeler açılınca odaya hafif bir aydınlık doldu.",
    "küçük kız kapının önünde annesini sabırla bekledi.",
    "kırmızı elmalar pazardaki eski tezgâha dikkatle dizildi.",
    "sıcak çayla birlikte masaya ince dilimli ekmek geldi.",
)
FINGERPRINTS = ("native_wav_sha256", "native_sample_sha256", "pcm_sha256", "wav_sha256")
CODE = (
    "backend/scripts/audio_okur_turkish_followup_v245.py",
    "backend/tests/test_audio_okur_turkish_followup_v245.py",
    "docs/audio-okur-turkish-followup-v245-method.md",
)
read, write, sha, require = previous.read, previous.write, previous.sha, previous.require


def plan():
    require(
        len(PHRASES) == len(set(PHRASES)) == 24
        and all(50 <= len(phrase) <= 56 and 7 <= len(phrase.split()) <= 8 for phrase in PHRASES),
        "Exact distinct authored sentences in the prior qualified length range required",
    )
    rows = [
        {
            "slot": f"synthetic-{index:02d}",
            "phrase": phrase,
            "seed": 245_000 + index,
            "token_sha256": previous.digest(previous.token_ids(phrase).astype("<i8").tobytes()),
        }
        for index, phrase in enumerate(PHRASES)
    ]
    prior_rows = previous.plan()
    for key in ("phrase", "token_sha256", "seed"):
        values = {row[key] for row in rows}
        require(
            len(values) == len(rows) and values.isdisjoint(row[key] for row in prior_rows),
            "Follow-up text, letter encoding and seed must be separate from v240",
        )
    return rows


def previous_result():
    require(
        sha(previous.PUBLIC) == PREVIOUS_FREEZE_SHA256 and sha(PREVIOUS_RESULT) == PREVIOUS_RESULT_SHA256,
        "Exact published v240 freeze and result required",
    )
    return read(PREVIOUS_RESULT)


def identity():
    old_result = previous_result()
    require(previous.readout() == old_result, "The published v240 result and private receipts must replay")
    identity = read(previous.PUBLIC)
    old_plan = identity.pop("private_plan_sha256")
    return {
        **identity,
        "schema": "chipmunk.audio-okur-turkish-followup-freeze/245",
        "code_sha256": {**identity["code_sha256"], **{name: sha(ROOT / name) for name in CODE}},
        "previous_freeze_sha256": PREVIOUS_FREEZE_SHA256,
        "previous_result_sha256": PREVIOUS_RESULT_SHA256,
        "previous_private_plan_sha256": old_plan,
        "previous_resource_profile_sha256": old_result["resource_profile_sha256"],
        "previous_private_journal_sha256": old_result["private_case_journal_sha256"],
        "previous_private_repeat_sha256": old_result["private_repeat_sha256"],
        "shared_acquisition_sha256": old_result["acquisition_sha256"],
        "authored_character_bounds": [50, 56],
        "authored_word_count_bounds": [7, 8],
        "planned_slots": len(plan()),
    }


def freeze():
    require(not any(path.exists() for path in (TRIAL, PUBLIC)), "Preserve earlier follow-up attempts")
    locked, rows = identity(), plan()
    TRIAL.mkdir(mode=0o700)
    private = TRIAL / "private-plan.json"
    write(private, rows)
    locked["private_plan_sha256"] = sha(private)
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return locked


def reject_previous_waveforms(rows, prior_rows):
    for key in FINGERPRINTS:
        old = {row[key] for row in prior_rows if row["status"] == "qualified"}
        new = {row[key] for row in rows if row["status"] == "qualified"}
        require(new.isdisjoint(old), "A qualified follow-up waveform matches a v240 fingerprint")


def summarize(rows, repeat, planned):
    require(
        sha(PREVIOUS_JOURNAL) == previous_result()["private_case_journal_sha256"],
        "Exact v240 journal required for exposure comparison",
    )
    prior_rows = [json.loads(line) for line in PREVIOUS_JOURNAL.read_text().splitlines()]
    reject_previous_waveforms(rows, prior_rows)
    return previous.summarize(rows, repeat, planned)


verify = scoped(previous.verify, {"TRIAL": TRIAL, "PUBLIC": PUBLIC, "identity": identity, "plan": plan})
SCOPE = {"STAGE": STAGE, "OUTPUT": OUTPUT, "PUBLIC": PUBLIC, "verify": verify, "summarize": summarize}
worker = scoped(previous.worker, {**SCOPE, "prepare_case": scoped(previous.prepare_case, {"OUTPUT": OUTPUT})})
run = scoped(
    previous.run, {"TRIAL": TRIAL, "OUTPUT": OUTPUT, "verify": verify, "MODULE": MODULE, "__file__": __file__}
)


def readout():
    observed = scoped(
        previous.readout,
        {
            **SCOPE,
            "__file__": __file__,
            "read_prepared_pcm": scoped(previous.read_prepared_pcm, {"OUTPUT": OUTPUT}),
        },
    )()
    return {
        **observed,
        "schema": "chipmunk.audio-okur-turkish-followup-result/245",
        "previous_result_sha256": PREVIOUS_RESULT_SHA256,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"preflight": identity, "freeze": freeze, "run": run, "worker": worker, "readout": readout}
    parser.add_argument("command", choices=commands)
    result = commands[parser.parse_args().command]()
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
