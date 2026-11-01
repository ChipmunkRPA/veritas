"""Prepare fresh, traceable v3 audio inputs without importing any classifier.

Full source bytes and metadata are retained. No serving or fusion file is written.
DFADD has no public per-file transcript mapping: its complete LJ Speech source
text corpus is checked against prior data and excluded from new development.
This cannot establish exact DFADD prompt independence without that mapping.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import evaluate_audio_candidates as base

ROOT = base.ROOT
OUT = ROOT / 'data/independent-evaluation/audio-v3'
RESEARCH = OUT / 'research'
SEED = 'chipmunk-audio-confirmation-20261101-v3'
REVISIONS = dict(base.REVISIONS, **{
    'isjwdu/DFADD': 'dfc1eeab3cb0068db8e87a2b89a1ebd103665b1f',
    'realnetworks-kontxt/fleurs-hs': '9e6cc6c908e34d77f6fbd46d665ad31b33dd9369',
})
base.OUT, base.SEED, base.REVISIONS = OUT, SEED, REVISIONS
save, sha, stable = base.save_json, base.digest, base.stable
CONDITIONS = ['clean', 'noise_0', 'noise_5', 'noise_15', 'music_5', 'music_15']
FAMILIES = ['PflowTTS', 'StyleTTS2', 'NaturalSpeech2']


def old_manifests():
    return [json.loads((ROOT / f'data/independent-evaluation/{d}/manifest.json').read_bytes())
            for d in ('audio', 'audio-v2')]


def norm(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def exclusions():
    rows = [r for m in old_manifests() for r in m['base_files']]
    return {k: {r[k] for r in rows if r.get(k) is not None} for k in
            ('speaker', 'source_group', 'transcript_sha256', 'sha256', 'decoded_pcm_sha256')}


def lj_texts():
    rows = list(csv.reader((RESEARCH / 'lj-speech-original-metadata.csv').open(), delimiter='|'))
    if len(rows) != 13100 or len({r[0] for r in rows}) != 13100:
        raise ValueError('Expected the complete original 13,100-entry LJ Speech transcript list')
    return sorted({norm(t) for r in rows for t in r[1:]})


def text_overlaps_lj(text, corpus):
    normalized = norm(text)
    # Compare complete passages and all five-word excerpts: DFADD selects 5–10
    # word LJ Speech sentences; this deliberately rejects more than exact matches.
    words = normalized.split()
    return normalized in corpus['full'] or any(' '.join(words[i:i+5]) in corpus['fivegrams']
                                               for i in range(max(0, len(words)-4)))


def lj_index():
    texts = lj_texts()
    return {'full': set(texts), 'fivegrams': {' '.join(t.split()[i:i+5]) for t in texts
                                           for i in range(max(0, len(t.split())-4))}}


def url(repo, name):
    return f'https://huggingface.co/datasets/{repo}/resolve/{REVISIONS[repo]}/{name}'


def archive(family):
    import fsspec
    import zipfile
    stream = fsspec.open(url('isjwdu/DFADD', f'DATASET_{family}.zip'), 'rb', block_size=65536).open()
    return stream, zipfile.ZipFile(stream)


def metadata():
    import bz2
    import requests
    import fsspec
    import pyarrow.parquet as pq
    OUT.mkdir(parents=True, exist_ok=True)
    RESEARCH.mkdir(exist_ok=True)
    (OUT / 'metadata').mkdir(exist_ok=True)
    (OUT / 'sources').mkdir(exist_ok=True)
    for repo in ('realnetworks-kontxt/fleurs-hs','openslr/librispeech_asr','isjwdu/DFADD'):
        publisher_path=RESEARCH/(repo.replace('/','_')+'-publisher.json')
        if not publisher_path.exists():
            publisher=base.get('https://huggingface.co/api/datasets/'+repo).json()
            if publisher['sha']!=REVISIONS[repo] or publisher.get('gated'):
                raise ValueError('Pinned public dataset revision or access changed')
            save(publisher_path,publisher)
        card_path=RESEARCH/(repo.replace('/','_')+'-README.md')
        if not card_path.exists():card_path.write_bytes(base.get(url(repo,'README.md')).content)
    for name in ('recording-metadata','recording-transcripts','voice-distribution','voice-metadata'):
        path=RESEARCH/f'realnetworks-kontxt_fleurs-hs-data_en_us_{name}.csv'
        if not path.exists():
            path.write_bytes(base.get(url('realnetworks-kontxt/fleurs-hs',f'data/en_us/{name}.csv')).content)
    lj_path=RESEARCH/'lj-speech-original-metadata.csv'
    if not lj_path.exists():
        # The official archive begins with metadata.csv. Bound the read rather
        # than downloading its multi-gigabyte audio collection.
        with requests.get('https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
                          headers={'Range':'bytes=0-8388607'},stream=True,timeout=60) as response:
            response.raise_for_status();parts=[];size=0
            for chunk in response.iter_content(65536):
                parts.append(chunk);size+=len(chunk)
                if size>=8388608:break
        raw=bz2.BZ2Decompressor().decompress(b''.join(parts)[:8388608])
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tar:
            lj_path.write_bytes(tar.extractfile('LJSpeech-1.1/metadata.csv').read())
    # Preserve source pages and license declarations next to our metadata.
    refs = {
        'DFADD-primary-paper.html': 'https://arxiv.org/html/2409.08731v1',
        'DFADD-primary-README.md': 'https://raw.githubusercontent.com/isjwdu/DFADD/main/README.md',
        'DFADD-MIT-LICENSE.txt': 'https://raw.githubusercontent.com/isjwdu/DFADD/main/LICENSE',
        'LJ-Speech-primary.html': 'https://keithito.com/LJ-Speech-Dataset/',
        'LibriSpeech-primary.html': 'https://www.openslr.org/12/',
    }
    def page(item):
        name, link = item
        path = RESEARCH/name
        if not path.exists(): path.write_bytes(base.get(link).content)
    with ThreadPoolExecutor(5) as pool: list(pool.map(page, refs.items()))
    def zip_index(family):
        path = OUT / 'metadata' / f'dfadd-{family}-index.json'
        if path.exists(): return
        stream, z = archive(family)
        try:
            save(path, [dict(name=i.filename, bytes=i.file_size, compressed_bytes=i.compress_size,
                             crc32=i.CRC, offset=i.header_offset) for i in z.infolist()])
        finally: z.close(); stream.close()
        print('Indexed DFADD', family, flush=True)
    with ThreadPoolExecutor(4) as pool: list(pool.map(zip_index, FAMILIES + ['VCTK_BONAFIDE']))
    repo = 'openslr/librispeech_asr'
    publisher = json.loads((RESEARCH/'openslr_librispeech_asr-publisher.json').read_bytes())
    names = sorted(r['rfilename'] for r in publisher['siblings']
                   if r['rfilename'].startswith('all/train.clean.100/') and r['rfilename'].endswith('.parquet'))
    def libri(name):
        path = OUT / 'metadata' / ('libri-' + Path(name).name + '.json')
        if not path.exists():
            with fsspec.open(url(repo, name), 'rb', block_size=65536) as s:
                rows = pq.ParquetFile(s).read(columns=['speaker_id', 'chapter_id', 'id', 'text']).to_pylist()
            save(path, rows)
        return json.loads(path.read_bytes())
    with ThreadPoolExecutor(4) as pool: tables = list(pool.map(libri, names))
    rows = []
    for name, table in zip(names, tables, strict=True):
        offset = len(rows)
        rows.extend(dict(r, row_index=offset+i, parquet=name) for i, r in enumerate(table))
    save(OUT/'metadata/libri-train100-index.json', rows)
    prior = [r for m in old_manifests() for r in m['base_files']]
    corpus = lj_index()
    overlaps = [{'id': r['id'], 'transcript': r['transcript']}
                for r in prior if text_overlaps_lj(r['transcript'], corpus)]
    containment = [{'id': r['id'], 'matching_lj_passages': [t for t in corpus['full']
        if t == norm(r['transcript']) or t in norm(r['transcript']) or norm(r['transcript']) in t]}
        for r in prior]
    containment = [r for r in containment if r['matching_lj_passages']]
    save(OUT/'metadata/lj-prior-exclusion-audit-v2.json', dict(
        corpus_entries=13100, corpus_sha256=base.file_digest(RESEARCH/'lj-speech-original-metadata.csv'),
        original_url='https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
        archive_member='LJSpeech-1.1/metadata.csv', acquisition='First 8 MiB HTTP range; original bzip2/tar metadata member',
        prior_originals=len(prior), normalized_five_word_overlap=overlaps,
        normalized_complete_passage_or_containment_overlap=containment,
        five_word_note='Generic five-word overlap is recorded; it is not a claim of shared utterances. Complete LJ passages and containment are separately checked.',
        exact_prompt_mapping_available=False,
        primary_membership_evidence='DFADD paper arXiv:2409.08731v1 §3.1 states all 300 text prompts selected from LJ Speech.'))
    print('Metadata ready. Prior LJ five-word overlaps:', len(overlaps), flush=True)


def source_ref(raw, **metadata):
    path = OUT/'sources'/(sha(raw)+'.audio')
    if not path.exists(): path.write_bytes(raw)
    pcm = base.decode(path)
    import soundfile as sf
    return dict(metadata, path=str(path.relative_to(OUT)), sha256=sha(raw), bytes=len(raw),
                decoded_pcm_sha256=sha(pcm.astype('<f4').tobytes()), decoded_samples=len(pcm),
                original_sample_rate=sf.info(path).samplerate)


def parquet_group(repo, name, group, columns):
    """Bounded direct range fallback; never downloads the full corpus."""
    import fsspec
    import pyarrow.parquet as pq
    key=sha((repo+name+str(group)).encode())
    cache=OUT/'metadata'/f'parquet-audio-group-{key}.json'
    if cache.exists():return json.loads(cache.read_bytes())
    with fsspec.open(url(repo,name),'rb',block_size=65536) as stream:
        reader=pq.ParquetFile(stream)
        compressed=sum(reader.metadata.row_group(group).column(i).total_compressed_size
                       for i in range(reader.metadata.row_group(group).num_columns))
        consumed=sum(json.loads(p.read_bytes())['compressed_bytes']
                     for p in (OUT/'metadata').glob('parquet-audio-group-*.json'))
        if consumed+compressed>3_000_000_000:raise ValueError('3 GB direct range audio fallback budget exceeded')
        rows=reader.read_row_group(group,columns=['audio']+columns).to_pylist()
    result=[]
    for offset,row in enumerate(rows):
        raw=row.pop('audio')['bytes']
        ref=source_ref(raw,dataset=repo,revision=REVISIONS[repo],url=url(repo,name),
                       parquet=name,parquet_row_group=group,parquet_group_row=offset)
        result.append(dict(row,ref=ref))
    value=dict(compressed_bytes=compressed,rows=result)
    save(cache,value);print('Range fallback',repo,name,group,compressed,'cumulative',consumed+compressed,flush=True)
    return value


def download_fleurs():
    """Download two bounded public archives (382,528,254 bytes total)."""
    import requests
    paths = []
    for split, expected in [('dev',142437887), ('test',240090367)]:
        path = OUT/'sources'/f'fleurs-en-{split}.tar.gz'
        if not path.exists() or path.stat().st_size != expected:
            with requests.get(url('realnetworks-kontxt/fleurs-hs', f'data/en_us/splits/{split}.tar.gz'),
                              stream=True, timeout=90) as response:
                response.raise_for_status()
                with path.open('wb') as file:
                    for chunk in response.iter_content(1024*1024): file.write(chunk)
        if path.stat().st_size != expected: raise ValueError('Unexpected FLEURS archive length')
        paths.append(dict(upstream_split=split, path=str(path.relative_to(OUT)), bytes=expected,
                          sha256=base.file_digest(path)))
        print('Downloaded FLEURS', split, expected, flush=True)
    save(OUT/'metadata/fleurs-archive-receipts.json', paths)


def fleurs_alias(r):
    vendor, name = r['tts_vendor'], r['tts_name']
    if vendor == 'amazon': return 'amazon:' + name
    if vendor == 'azure': return 'azure:' + name.replace('Multilingual', '')
    if vendor == 'google':
        return 'google:en-US-voice-' + name.rsplit('-',1)[-1]
    return vendor + ':' + name


def prepare_fleurs(used, corpus):
    def read(name):
        return list(csv.DictReader((RESEARCH/f'realnetworks-kontxt_fleurs-hs-data_en_us_{name}.csv').open()))
    transcripts = {r['id']: r['transcript'] for r in read('recording-transcripts')}
    recordings = {r['name']: r for r in read('recording-metadata')}
    groups = defaultdict(list)
    for row in read('voice-distribution'):
        rec = recordings[row['name']]
        if rec['split'] not in ('dev','test'): continue
        text = transcripts[row['id']]
        key = base.transcript_key(text)
        if key in used['transcript_sha256'] or text_overlaps_lj(text, corpus): continue
        alias = fleurs_alias(row)
        split = 'fit' if row['tts_vendor'] == 'google' and ('Standard' in row['tts_name'] or 'News' in row['tts_name']) else 'cal'
        groups[(split,alias)].append(dict(row, upstream_split=rec['split'], transcript=text, transcript_sha256=key))
    print('FLEURS eligible groups', Counter(k[0] for k in groups), flush=True)
    durations_path=OUT/'metadata/fleurs-source-durations.json'
    if durations_path.exists():durations=json.loads(durations_path.read_bytes())
    else:
        import soundfile as sf
        durations={}
        for upstream in ('dev','test'):
            with tarfile.open(OUT/'sources'/f'fleurs-en-{upstream}.tar.gz','r:gz') as tar:
                for member in tar:
                    if '/synthetic/' in '/'+member.name and member.name.endswith('.wav'):
                        info=sf.info(io.BytesIO(tar.extractfile(member).read()))
                        durations[Path(member.name).name]=info.frames/info.samplerate
        save(durations_path,durations)
    selected = []
    # Round-robin voices before extra files, with aliases kept entirely in one split.
    for split, target in [('fit',20),('cal',40)]:
        keys = sorted([k for k in groups if k[0]==split],key=lambda k:stable(k[1]))
        if len(keys) != (8 if split=='fit' else 35): raise ValueError('Unexpected FLEURS alias group count')
        queues = {k:sorted(groups[k],key=lambda r:stable(r['name'])) for k in keys}
        for n in range(target):
            key = keys[n % len(keys)]
            for row in queues[key]:
                if durations.get(row['name'],0)<[2.,3.,4.][n%3]:continue
                group = 'fleurs:transcript:' + row['id']
                if row['transcript_sha256'] in used['transcript_sha256'] or group in used['source_group']: continue
                used['transcript_sha256'].add(row['transcript_sha256']); used['source_group'].add(group)
                selected.append(dict(row, split=split, speaker=key[1], source_group=group, seconds=[2.,3.,4.][n%3]))
                break
            else: raise ValueError('Insufficient fresh FLEURS transcript options')
    refs = {}
    for upstream in ('dev','test'):
        wanted = {r['name'] for r in selected if r['upstream_split']==upstream}
        with tarfile.open(OUT/'sources'/f'fleurs-en-{upstream}.tar.gz','r:gz') as tar:
            for m in tar:
                if '/synthetic/' in '/' + m.name and Path(m.name).name in wanted:
                    raw = tar.extractfile(m).read()
                    refs[Path(m.name).name] = source_ref(raw, dataset='realnetworks-kontxt/fleurs-hs',
                        revision=REVISIONS['realnetworks-kontxt/fleurs-hs'],
                        url=url('realnetworks-kontxt/fleurs-hs',f'data/en_us/splits/{upstream}.tar.gz'),
                        archive_member=m.name, upstream_split=upstream)
    sources=[]
    for r in selected:
        ref=refs[r['name']]
        if ref['decoded_samples'] < r['seconds']*16000: raise ValueError('Selected FLEURS source unexpectedly short')
        if ref['sha256'] in used['sha256'] or ref['decoded_pcm_sha256'] in used['decoded_pcm_sha256']: raise ValueError('Duplicate FLEURS input')
        row=dict(ref,id='fake:fleurs:'+r['name'],split=r['split'],label=1,speaker=r['speaker'],cluster_id=r['speaker'],
            generator='fleurs-'+r['tts_vendor'],generator_voice=r['tts_name'],generator_engine=r['tts_engine'],
            source_group=r['source_group'],transcript=r['transcript'],transcript_sha256=r['transcript_sha256'],
            seconds=r['seconds'],license='CC-BY-4.0',license_url='https://huggingface.co/datasets/realnetworks-kontxt/fleurs-hs',
            label_source='KONTXT/RealNetworks publisher synthetic directory and per-file vendor/voice distribution',
            voice_identity_limit='Documented voice names/known aliases; no biometric identity verification across vendors.')
        for k in used:
            if row.get(k) is not None: used[k].add(row[k])
        sources.append(row)
    return sources


def prepare_dfadd(used):
    denied=exclusions()
    all_speakers=set()
    indexes={}
    for family in FAMILIES:
        entries=json.loads((OUT/'metadata'/f'dfadd-{family}-index.json').read_bytes())
        candidates=[]
        for entry in entries:
            match=re.search(r'/((?:p\d+|s\d+))_(\d+)_',entry['name'])
            if match and entry['name'].endswith(('.flac','.wav')):
                speaker='vctk:'+match[1]
                if speaker not in denied['speaker']:
                    candidates.append(dict(entry,speaker=speaker,prompt_index=int(match[2])))
                    all_speakers.add(speaker)
        indexes[family]=candidates
    common=set.intersection(*[{r['speaker'] for r in indexes[family]} for family in FAMILIES])
    ordered=sorted(common,key=stable)
    sources=[]; used_indices=set(); rejected=[]
    for fi,family in enumerate(FAMILIES):
        stream,z=archive(family)
        try:
            for n in range(50):
                speaker=ordered[(fi*50+n)%len(ordered)]
                choices=sorted([r for r in indexes[family] if r['speaker']==speaker],key=lambda r:stable(str(r['prompt_index'])))
                for r in choices:
                    if r['prompt_index'] in used_indices: continue
                    cache=OUT/'metadata'/('dfadd-source-'+family+'-'+speaker.replace(':','-')+'-'+str(r['prompt_index'])+'.json')
                    if cache.exists(): ref=json.loads(cache.read_bytes())
                    else:
                        ref=source_ref(z.read(r['name']),dataset='isjwdu/DFADD',revision=REVISIONS['isjwdu/DFADD'],
                            url=url('isjwdu/DFADD',f'DATASET_{family}.zip'),archive_member=r['name'],
                            archive_crc32=r['crc32'],archive_compressed_bytes=r['compressed_bytes'])
                        save(cache,ref)
                    seconds=min(4.,np.floor(ref['decoded_samples']/16000*4)/4)
                    if seconds<2 or ref['sha256'] in used['sha256'] or ref['decoded_pcm_sha256'] in used['decoded_pcm_sha256']:
                        rejected.append(dict(member=r['name'],reason='shorter than 2 s or duplicate bytes/PCM'));continue
                    pcm=base.decode(OUT/ref['path'])[:int(seconds*16000)]
                    if np.sqrt(np.mean(pcm**2))<1e-4:
                        rejected.append(dict(member=r['name'],reason='silent source interval'));continue
                    row=dict(ref,id='fake:dfadd:'+family+':'+Path(r['name']).stem,split='test',label=1,
                        speaker=speaker,cluster_id=speaker,generator=family,source_group='dfadd:'+family+':'+Path(r['name']).stem,
                        transcript=None,transcript_sha256=None,prompt_index=r['prompt_index'],
                        text_corpus='LJ-Speech-1.1',text_group='dfadd-ljspeech-300-prompts',
                        transcript_provenance='Actual per-file text is unpublished; full LJ Speech corpus excluded from development and previous study text. Prompt indices are recorded, not treated as verified cross-family transcript identities.',
                        seconds=float(seconds),license='MIT; underlying VCTK CC-BY-4.0; LJ Speech public-domain',
                        license_url='https://github.com/isjwdu/DFADD',
                        label_source='Official DFADD April 2025 revised archive; synthetic family/source speaker filename and paper §3.1–3.2')
                    sources.append(row);used_indices.add(r['prompt_index'])
                    for k in used:
                        if row.get(k) is not None: used[k].add(row[k])
                    print('Prepared DFADD',family,n+1,seconds,flush=True);break
                else: raise ValueError('Insufficient eligible distinct-index DFADD files')
        finally:z.close();stream.close()
    save(OUT/'metadata/dfadd-pre-score-rejections.json',rejected)
    return sources


def prepare_reals(used, corpus, generated):
    vctk=json.loads((ROOT/'data/independent-evaluation/audio-v2/metadata/vctk-index.json').read_bytes())
    libri=json.loads((OUT/'metadata/libri-train100-index.json').read_bytes())
    result=[]
    denied=exclusions()
    # Authentic duration counts are matched exactly to generated counts per split.
    durations={s:[r['seconds'] for r in sorted([r for r in generated if r['split']==s],key=lambda r:stable(r['id']))]
               for s in ('fit','cal','test')}
    vctk_speakers=sorted({r['speaker'] for r in generated if r['split']=='test'},key=stable)[:50]
    entries=json.loads((OUT/'metadata/dfadd-VCTK_BONAFIDE-index.json').read_bytes())
    members={Path(r['name']).stem:r for r in entries if r['name'].endswith('.wav')}
    stream,z=archive('VCTK_BONAFIDE')
    try:
        for n,speaker in enumerate(vctk_speakers):
            seconds=durations['test'][n]
            choices=sorted([r for r in vctk if 'vctk:'+r['speaker_id']==speaker and r['mic_id']=='mic1'],key=lambda r:stable(r['id']))
            for r in choices:
                rid=r['id'].rsplit('_mic',1)[0];tk=base.transcript_key(r['text']);group='vctk:'+rid
                if rid not in members or tk in used['transcript_sha256'] or group in used['source_group'] or text_overlaps_lj(r['text'],corpus):continue
                cache=OUT/'metadata'/('vctk-source-'+rid+'.json')
                if cache.exists():ref=json.loads(cache.read_bytes())
                else:
                    ref=source_ref(z.read(members[rid]['name']),dataset='isjwdu/DFADD',revision=REVISIONS['isjwdu/DFADD'],
                        url=url('isjwdu/DFADD','DATASET_VCTK_BONAFIDE.zip'),archive_member=members[rid]['name'],
                        transcript_dataset='jspaulsen/vctk',transcript_revision='fb74847570d78d2b23e83193d8e55df80e6271b2',
                        transcript_row_index=r['row_idx'],microphone='not documented by archive publisher')
                    save(cache,ref)
                if ref['decoded_samples']<seconds*16000 or ref['sha256'] in used['sha256'] or ref['decoded_pcm_sha256'] in used['decoded_pcm_sha256']:continue
                row=dict(ref,id='real:'+rid,split='test',label=0,speaker=speaker,cluster_id=speaker,generator='human',
                    source_group=group,transcript=r['text'],transcript_sha256=tk,seconds=seconds,license='CC-BY-4.0',
                    license_url='https://datashare.ed.ac.uk/handle/10283/3443',
                    label_source='CSTR VCTK bona fide recording in DFADD; transcript matched by exact VCTK utterance ID')
                result.append(row)
                for k in used:used[k].add(row[k])
                print('Prepared VCTK',n+1,seconds,flush=True);break
            else:raise ValueError('No eligible authentic VCTK recording for '+speaker)
    finally:z.close();stream.close()
    groups=defaultdict(list)
    shard_offsets={}
    for r in libri:
        shard_offsets.setdefault(r['parquet'],r['row_index'])
        r['local_row']=r['row_index']-shard_offsets[r['parquet']]
        r['row_group']=r['local_row']//100
        speaker='librispeech:'+str(r['speaker_id'])
        if speaker not in denied['speaker']:groups[speaker].append(r)
    ordered_speakers=sorted(groups,key=stable)[:160]
    # Metadata-only set cover reduces direct range transfer while preserving the
    # preselected speakers. Selection never observes detector output.
    coverage=defaultdict(set)
    for speaker in ordered_speakers:
        for r in groups[speaker]:coverage[(r['parquet'],r['row_group'])].add(speaker)
    uncovered=set(ordered_speakers);preferred={}
    while uncovered:
        key=min(coverage,key=lambda k:(-len(coverage[k]&uncovered),stable(str(k))))
        for speaker in coverage[key]&uncovered:preferred[speaker]=key
        uncovered-=coverage[key]
    save(OUT/'metadata/libri-range-selection-plan.json',dict(
        speakers=ordered_speakers,selected_row_groups=len(set(preferred.values())),
        preferred={s:list(k) for s,k in preferred.items()},
        selection='Stable hash speaker cohort; metadata-only greedy row-group coverage; source duration and transcript eligibility only.'))
    available=iter(ordered_speakers)
    for split,seconds_list in [('fit',durations['fit']),('cal',durations['cal']),('test',durations['test'][50:])]:
        for n,seconds in enumerate(seconds_list):
            speaker=next(available)
            for r in sorted(groups[speaker],key=lambda r:((r['parquet'],r['row_group'])!=preferred[speaker],stable(r['id']))):
                tk=base.transcript_key(r['text']);group='librispeech:'+r['id']
                if tk in used['transcript_sha256'] or group in used['source_group'] or text_overlaps_lj(r['text'],corpus):continue
                data=parquet_group('openslr/librispeech_asr',r['parquet'],r['row_group'],['id','text'])['rows'][r['local_row']%100]
                if data['id']!=r['id']:raise ValueError('LibriSpeech metadata/audio row mismatch')
                ref=data['ref'];ph=ref['decoded_pcm_sha256']
                if ref['decoded_samples']<seconds*16000 or ref['sha256'] in used['sha256'] or ph in used['decoded_pcm_sha256']:continue
                row=dict(ref,id='real:'+r['id'],split=split,label=0,speaker=speaker,cluster_id=speaker,generator='human',
                    source_group=group,transcript=r['text'],transcript_sha256=tk,decoded_pcm_sha256=ph,
                    seconds=seconds,license='CC-BY-4.0',
                    license_url='https://www.openslr.org/12/',label_source='LibriSpeech publisher: genuine LibriVox audiobook recording')
                result.append(row)
                for k in used:used[k].add(row[k])
                print('Prepared LibriSpeech',split,n+1,seconds,flush=True);break
            else:raise ValueError('No eligible fresh LibriSpeech source for '+speaker)
    return result


def prepare_sources():
    path=OUT/'source-preparation.json'
    if path.exists():return json.loads(path.read_bytes())['base_files']
    used=exclusions();corpus=lj_index()
    audit=json.loads((OUT/'metadata/lj-prior-exclusion-audit-v2.json').read_bytes())
    if any(len(t.split())>=5 for r in audit['normalized_complete_passage_or_containment_overlap']
           for t in r['matching_lj_passages']):
        raise ValueError('Prior transcript overlaps LJ source corpus; review before selecting sources')
    fake=prepare_fleurs(used,corpus)+prepare_dfadd(used)
    sources=fake+prepare_reals(used,corpus,fake)
    save(path,dict(seed=SEED,base_files=sources,counts={s:dict(Counter(r['label'] for r in sources if r['split']==s))
        for s in ('fit','cal','test')},prior_manifest_hashes=[m['sha256'] for m in old_manifests()]))
    return sources


def prepare_backgrounds():
    path=OUT/'backgrounds.json'
    if path.exists():return json.loads(path.read_bytes())
    corpus=[]
    denied_hashes={b['sha256'] for m in old_manifests() for b in m['backgrounds']}
    denied_ids={b.get('source_id') for m in old_manifests() for b in m['backgrounds']}
    denied_names={Path(str(s)).stem for s in denied_ids if s}
    old_bg_cache=OUT/'metadata/prior-background-pcm-hashes.json'
    if old_bg_cache.exists():old_pcm=json.loads(old_bg_cache.read_bytes())
    else:
        old_pcm=[dict(id=b.get('source_id'),sha256=b['sha256'],decoded_pcm_sha256=sha(
            base.decode(ROOT/f'data/independent-evaluation/{directory}'/b['path']).astype('<f4').tobytes()))
            for directory in ('audio','audio-v2')
            for b in json.loads((ROOT/f'data/independent-evaluation/{directory}/manifest.json').read_bytes())['backgrounds']]
        save(old_bg_cache,old_pcm)
    denied_pcm={b['decoded_pcm_sha256'] for b in old_pcm}
    configs=[('noise','free-sound',{'fit':6,'cal':6,'test':8},'Public-domain'),
             ('music','rfm',{'fit':2,'cal':2,'test':4},'CC-BY-3.0')]
    for kind,subset,targets,license_name in configs:
        license_url=f'https://huggingface.co/datasets/corypaik/musan/raw/{REVISIONS["corypaik/musan"]}/licenses/{kind}/{subset}/LICENSE'
        declaration=base.get(license_url).text
        if ('Public Domain' if kind=='noise' else 'CC BY 3.0') not in declaration:
            raise ValueError('Expected commercial-use-compatible background license')
        (RESEARCH/f'musan-{kind}-{subset}-LICENSE.txt').write_text(declaration)
        name='noise/train-00000-of-00002.parquet' if kind=='noise' else 'music/train-00009-of-00010.parquet'
        rows=parquet_group('corypaik/musan',name,0,['path','source'])['rows']
        available=sorted([r for r in rows if subset in r['path']
                          and r['path'] not in denied_ids and Path(r['path']).stem not in denied_names],key=lambda r:stable(r['path']))
        for split,target in targets.items():
            count=0
            while count<target:
                if not available:raise ValueError('Insufficient fresh licensed backgrounds')
                r=available.pop(0);rid=r['path']
                filename=Path(rid).stem
                if kind=='music' and filename not in declaration:raise ValueError('Background missing from per-file license')
                ref=r['ref'];pcm=base.decode(OUT/ref['path'])
                ph=sha(pcm.astype('<f4').tobytes())
                if ref['sha256'] in denied_hashes or ph in denied_pcm or len(pcm)<8*16000 or np.sqrt(np.mean(pcm**2))<1e-4:continue
                denied_hashes.add(ref['sha256']);denied_ids.add(rid);denied_names.add(filename);denied_pcm.add(ph)
                corpus.append(dict(ref,kind=kind,split=split,source_id=rid,decoded_pcm_sha256=ph,
                    license=license_name,license_url=license_url,
                    parent_identity_limit='MUSAN recording path and decoded audio checked; original FreeSound contributor/session IDs are unpublished.' if kind=='noise' else 'Distinct named musical works by one composer; creator identity is shared across splits.',
                    attribution='Kevin MacLeod, Incompetech, via MUSAN' if kind=='music' else 'MUSAN / FreeSound per-file public-domain credits in retained LICENSE'))
                count+=1;print('Prepared background',kind,split,count,flush=True)
    save(path,corpus)
    return corpus


def production_source_audit(sources):
    from app.detectors.audio.decode import decode_audio
    prior_path=OUT/'metadata/prior-production-source-hashes.json'
    if not prior_path.exists():
        jobs=[(ROOT/f'data/independent-evaluation/{directory}',r) for directory in ('audio','audio-v2')
              for r in json.loads((ROOT/f'data/independent-evaluation/{directory}/manifest.json').read_bytes())['base_files']]
        def old(job):
            directory,r=job;d=decode_audio(directory/r['path'],source_sha256=r['sha256'])
            return dict(id=r['id'],sha256=r['sha256'],pcm_sha256=d.mapping['pcm_sha256'],
                prefixes={str(n):sha(d.pcm[:int(n*16000)].astype('<f4').tobytes())
                          for n in [2+i*.25 for i in range(9)] if len(d.pcm)>=n*16000})
        with ThreadPoolExecutor(6) as pool:old_rows=list(pool.map(old,jobs))
        save(prior_path,old_rows)
    old_rows=json.loads(prior_path.read_bytes())
    full={r['pcm_sha256'] for r in old_rows}
    crops={h for r in old_rows for h in r['prefixes'].values()}
    def current(r):
        d=decode_audio(OUT/r['path'],source_sha256=r['sha256'])
        prefix=sha(d.pcm[:int(r['seconds']*16000)].astype('<f4').tobytes())
        if d.mapping['pcm_sha256'] in full or prefix in crops:
            raise ValueError('Production-decoded original/prefix overlaps a previous source')
        return dict(r,production_pcm_sha256=d.mapping['pcm_sha256'],source_prefix_pcm_sha256=prefix,
                    source_production_input=d.mapping)
    with ThreadPoolExecutor(6) as pool:result=list(pool.map(current,sources))
    if len({r['production_pcm_sha256'] for r in result})!=len(result):raise ValueError('Duplicate current production decoded source')
    if len({r['source_prefix_pcm_sha256'] for r in result})!=len(result):raise ValueError('Duplicate current production decoded prefix')
    return result


def prepare_variants():
    import soundfile as sf
    from app.detectors.audio.decode import decode_audio
    if (OUT/'manifest.json').exists():return
    sources=production_source_audit(prepare_sources())
    backgrounds=prepare_backgrounds()
    (OUT/'inputs').mkdir(exist_ok=True)
    decoded_bg={b['sha256']:base.decode(OUT/b['path']).astype(np.float64) for b in backgrounds}
    counters=Counter();jobs=[]
    for r in sorted(sources,key=lambda r:stable(r['id'])):
        key=(r['split'],r['label']);ordinal=counters[key];counters[key]+=1;jobs.append((r,ordinal))
    def one(job):
        source,ordinal=job;n=int(source['seconds']*16000)
        clean=decode_audio(OUT/source['path'],source_sha256=source['sha256']).pcm[:n].astype(np.float64)
        rms=np.sqrt(np.mean(clean**2))
        if rms<1e-4:raise ValueError('Silent source')
        clean*=.1/rms
        mixtures={'clean':clean};selected={}
        for kind in ('noise','music'):
            pool=[b for b in backgrounds if b['split']==source['split'] and b['kind']==kind]
            b=pool[ordinal%len(pool)];full_bg=decoded_bg[b['sha256']]
            start=int(stable(source['id']+kind+':offset')[:8],16)%(len(full_bg)-n+1)
            bg=full_bg[start:start+n];rms_bg=np.sqrt(np.mean(bg**2))
            if rms_bg<1e-5:raise ValueError('Silent fixed background interval')
            for snr in ((0,5,15) if kind=='noise' else (5,15)):
                name=f'{kind}_{snr}';mixtures[name]=clean+bg*(.1/rms_bg/10**(snr/20));selected[name]=(b,start,snr)
        gain=min(1.,.98/max(np.max(np.abs(v)) for v in mixtures.values()))
        result=[];clean_pcm=None
        for condition,mixture in mixtures.items():
            pcm=(mixture*gain).astype(np.float32);p=OUT/'inputs'/(sha(pcm.tobytes())+'.wav')
            sf.write(p,pcm,16000,subtype='FLOAT');digest=base.file_digest(p)
            d=decode_audio(p,source_sha256=digest)
            if condition=='clean':clean_pcm=d.pcm
            b=selected.get(condition)
            result.append(dict(id=stable(source['id']+condition),base_id=source['id'],split=source['split'],
                label=source['label'],speaker=source['speaker'],cluster_id=source['cluster_id'],generator=source['generator'],
                dataset=source['dataset'],seconds=source['seconds'],condition=condition,path=str(p.relative_to(OUT)),
                sha256=digest,pcm_sha256=d.mapping['pcm_sha256'],production_input=d.mapping,samples=n,
                sample_rate=16000,source_interval_samples=[0,n],background_sha256=b[0]['sha256'] if b else None,
                background_id=b[0]['source_id'] if b else None,background_kind=b[0]['kind'] if b else None,
                background_start_sample=b[1] if b else None,snr_db=b[2] if b else None,common_peak_gain=gain,
                measured_snr_db=None if not b else float(20*np.log10(np.sqrt(np.mean(clean_pcm.astype(float)**2))/
                    np.sqrt(np.mean((d.pcm-clean_pcm).astype(float)**2))))))
        return result
    with ThreadPoolExecutor(6) as pool:variants=[v for result in pool.map(one,jobs) for v in result]
    manifest=dict(schema='chipmunk.audio-confirmation/3',seed=SEED,deployment_eligible=False,
        scope='New source files and documented voice identities; 2–4-second English prefixes with six controlled noise/music conditions. DFADD exact transcript mapping is unpublished.',
        prior_manifest_hashes=[m['sha256'] for m in old_manifests()],base_files=sources,backgrounds=backgrounds,
        variants=variants,source_normalization='Production-decoded clean speech RMS -20 dBFS; common peak attenuation <=0.98 across all six conditions.',
        background_identity_limit='MUSAN recording paths, bytes and decoded PCM are disjoint. Original FreeSound contributor/session identities are unpublished. RFM tracks are distinct named works by Kevin MacLeod, whose creator identity spans splits.',
        text_overlap_limit='No complete LJ passage of at least five words matches or is contained in prior transcripts. Nine prior passages share generic five-word phrases and one contains a three-word LJ passage. Unknown per-file DFADD text prevents a guarantee of exact prior prompt independence. New fitting/calibration texts exclude all LJ five-word overlaps.',
        preparation_protocol_sha256=base.file_digest(ROOT.parent/'docs/audio-calibration-v3-protocol.md'))
    manifest['audit']=dict(original_counts={s:dict(Counter(r['label'] for r in sources if r['split']==s)) for s in ('fit','cal','test')},
        voice_counts={s:{str(label):len({r['cluster_id'] for r in sources if r['split']==s and r['label']==label})
                         for label in (0,1)} for s in ('fit','cal','test')},
        duration_counts={s:{str(label):dict(Counter(str(r['seconds']) for r in sources if r['split']==s and r['label']==label))
                            for label in (0,1)} for s in ('fit','cal','test')},
        variants=len(variants),models_scored=False,exact_dfadd_transcript_identity_verified=False)
    manifest['sha256']=sha(base.canonical(manifest));save(OUT/'manifest.json',manifest)
    print(json.dumps(manifest['audit'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['metadata','download-fleurs','sources','backgrounds','variants'])
    args=parser.parse_args()
    {'metadata':metadata,'download-fleurs':download_fleurs,'sources':prepare_sources,
     'backgrounds':prepare_backgrounds,'variants':prepare_variants}[args.command]()
