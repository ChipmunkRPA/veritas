# SPDX-License-Identifier: Apache-2.0
"""Build or check the public native decision ledger without project imports."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'chipmunk.video-mouth-luma-condition/native-only/1'
ARMS = ('plain','conditioned')
FLOOR = .04
SPLITS = {'development':56,'held':146}

def need(test, message):
    if not test:
        raise ValueError(message)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_bytes())

def overlap(left, right):
    a,b=map(Fraction,left)
    c,d=map(Fraction,right)
    return max(0,min(b,d)-max(a,c))/(max(b,d)-min(a,c))

def point(rows, arm):
    scores=sorted((r['arms'][arm]['score'] for r in rows if r['arms'][arm]['score'] is not None), reverse=True)
    return max(FLOOR,scores[2] if len(scores)>=3 else 0)

def totals(rows, arm, threshold, split):
    counts, by_role=Counter(),{}
    positives, ordinary=[],[]
    for row in rows:
        d=row['arms'][arm]
        score=d['score']
        observed=d['measured']
        proposed=score is not None
        need(proposed==bool(d['range']) and (not proposed or math.isfinite(score) and score>0 and observed),
             'Invalid proposed range')
        flag=proposed and score>threshold
        useful=bool(proposed and row['truth'] and overlap(d['range'],row['truth'])>=Fraction(1,2))
        proof=d.get('anchors')
        need(bool(proof)==(split=='held' and (flag or useful)), 'Native proof scope differs')
        if proof:
            start,end=d['frames']
            clock=Fraction(d['time_base'])
            need([a['frame'] for a in proof]==[start-1,(start+end)//2,end]
                 and proof[-1]['pts']*clock==Fraction(d['range'][-1])
                 and all(len(a['rgb_sha256'])==len(a['crop_sha256'])==64 for a in proof),
                 'Selected native timestamps differ')
        role=f"{row['family']}:{row['role']}"
        kind='neural' if row['truth'] else 'ordinary'
        marks={'files':1,'measured':observed,'flagged':flag,'overlap_before_point':useful,'localized':flag and useful}
        by_role.setdefault(role,Counter()).update(marks)
        counts.update({f'{kind}_{k}':v for k,v in marks.items()})
        if row['aligned']:
            counts.update({f"strict_{row['family']}_{k}":v for k,v in marks.items()})
            if observed:
                positives.append(score or 0)
        if kind=='ordinary' and observed:
            ordinary.append(score or 0)
    pairs=len(positives)*len(ordinary)
    auc=sum(2*(p>n)+(p==n) for p in positives for n in ordinary)
    return {'counts':dict(counts),'by_role':{k:dict(v) for k,v in by_role.items()},
            'auc_strict_vs_ordinary':str(Fraction(auc,2*pairs)) if pairs else None}

def assemble(run):
    lock_file=run/'freeze.json'
    lock=read(lock_file)
    audit=read(run/'audit_direct.json')
    points=read(run/'points.json')
    need(points['development_sha256']==digest(run/'development.json'), 'Point selection file differs')
    result={'schema':SCHEMA,'freeze_sha256':digest(lock_file),'protocol':lock['protocol'],
            'prior_roster_sha256':lock['prior_rosters'],'points':points['points'],'splits':{},
            'stress_clock_reopened_pictures':audit['stress_clock_pictures'],
            'stress_scoring':'pending due to workspace runtime startup failure'}
    groups={}
    for split,count in SPLITS.items():
        file=run/f'{split}.json'
        scored=read(file)
        roster={(r['family'],r['id'],r['role']):r for r in lock['rows'] if r['split']==split}
        need(scored['freeze_sha256']==digest(lock_file) and len(scored['rows'])==count,
             'Observed scope differs')
        rows=[]
        for row in scored['rows']:
            frozen=roster.pop((row['family'],row['id'],row['role']))
            need(all(row[k]==v for k,v in frozen.items()),'Pinned source differs')
            group=('fresh' if row['family']=='derived_ordinary' else row['family'], row['group'])
            reduced={k:row[k] for k in ('family','role','truth','native_frames','frame_support')}
            reduced.update(aligned=row.get('aligned',False),group=groups.setdefault(group,len(groups)+1),arms={})
            for arm in ARMS:
                score=row['arms'][arm]
                proposal=score['proposal']
                detail={'score':score['score'],'range':proposal['seconds'] if proposal else None,
                        'measured':score['measured']}
                if split=='held' and proposal and (score['score']>result['points'][arm] or
                    row['truth'] and overlap(proposal['seconds'],row['truth'])>=Fraction(1,2)):
                    detail.update({k:proposal[k] for k in ('time_base','frames','anchors')})
                reduced['arms'][arm]=detail
            rows.append(reduced)
        need(not roster,'Missing frozen files')
        result['splits'][split]={'sha256':digest(file),'rows':rows,'native_audit':audit.get(split,{})}
        result['splits'][split]['totals']={arm:totals(rows,arm,result['points'][arm],split) for arm in ARMS}
    return result

def check(result):
    need(result['schema']==SCHEMA and set(result['splits'])==set(SPLITS), 'Scope differs')
    need(result['points']=={arm:point(result['splits']['development']['rows'],arm) for arm in ARMS}, 'Points differ')
    for split,count in SPLITS.items():
        block=result['splits'][split]
        rows=block['rows']
        need(len(rows)==count and block['totals']=={arm:totals(rows,arm,result['points'][arm],split) for arm in ARMS},
             'Outcome differs')
        print(split,json.dumps({arm:d['counts'] for arm,d in block['totals'].items()},sort_keys=True))
    held=result['splits']['held']
    native=held['native_audit']
    need(native['scored_timestamps']==sum(row['native_frames'] for row in held['rows'])
         and native['ranges']==sum('anchors' in row['arms'][arm] for row in held['rows'] for arm in ARMS)
         and native['frame_crop_pairs']==sum(len({a['frame'] for arm in ARMS for a in row['arms'][arm].get('anchors',[])})
                                            for row in held['rows'])
         and native['files']==sum(any('anchors' in row['arms'][arm] for arm in ARMS) for row in held['rows']),
         'Native audit counts differ')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt',type=Path)
    parser.add_argument('--assemble',type=Path)
    args=parser.parse_args()
    if args.assemble:
        with args.receipt.open('x') as stream:
            json.dump(assemble(args.assemble),stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write('\n')
    check(read(args.receipt))
