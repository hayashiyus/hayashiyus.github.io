#!/usr/bin/env python3
"""Prepare the exact frozen inputs; no inference or GPU work is performed."""
import argparse, hashlib, json, urllib.request
from pathlib import Path
COMMIT='4d09ba6d6a7e891dd8063974dfe81b3d78e3494f'
REVISION='2fc06364715b967f1860aea9cf38778875588b17'
ROOT=f'https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{COMMIT}/exp8_math'
FILES={'code/apply_edit.py':'d2b627a6cf344f309cca4bcab4ff6ccc6ad1c8f29cabda78d11f99c19d1b1c5d','code/profiles.py':'ba4be365e44c3e165b9c3982af9dbcd8ec94f7fd0430e5fd9de1d57f249a4ae4','code/mathqa.py':'f0d576a1adc625dede7bff749b3bf8c9a5a8f6bae43d51e2d0ee36d694e89843','code/evaluate.py':'c93f477c02b3ae0d6b37abe3afad8bb7353e273322ff963eec7ff2270012035d','code/st_numpy.py':'3c4e532fb8edf374e1a4636aa806a04d062f824943eaf4bca0d3af5b0e7dca40','runs/p50c/edit/manifest.json':'83185c1db46e3dafb34af645f8c5ce033c09cc99d4512b9e1b8e3aba9d269225','runs/p50c/edit/edit.npz':'e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7'}
CANONICAL_MATH_SHA='06dd3b4208ad8399004c9367f194de82418e4f545319fa23c02c002665228798'
CONTRAST_SHA='60ab01cb23bcfe1820b618c71fcf0128eefe6a20f59e039c0adf40da039182c3'
SUFFIX='Please reason step by step, and put your final answer within \\boxed{}.'
def sha(b):return hashlib.sha256(b).hexdigest()
def put(p,b):
    if p.exists() and p.read_bytes()!=b:raise RuntimeError('refusing to overwrite differing bytes: '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def put_json(p,obj):put(p,(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode())
def get(url):
    with urllib.request.urlopen(url,timeout=60) as r:return r.read()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('experiments/data/repro_inputs'))
    ap.add_argument('--offline',action='store_true');ap.add_argument('--skip-model',action='store_true')
    ap.add_argument('--prepare-inputs',action='store_true');ap.add_argument('--math500',type=Path)
    ap.add_argument('--model-dir',type=Path);a=ap.parse_args()
    for rel,want in FILES.items():
        p=a.out/rel;b=p.read_bytes() if (a.offline or p.exists()) else get(ROOT+'/'+rel)
        if sha(b)!=want:raise RuntimeError('sha256 mismatch: '+rel)
        put(p,b);print('verified',rel)
    contrast=Path(__file__).with_name('contrast_texts.json').read_bytes()
    if sha(contrast)!=CONTRAST_SHA:raise RuntimeError('bundled contrast checksum failed')
    put(a.out/'contrast_texts.json',contrast)
    model_dir=a.model_dir or a.out/'model'
    if not a.skip_model and not a.offline:
        from huggingface_hub import snapshot_download
        snapshot_download('Qwen/Qwen3.5-0.8B',revision=REVISION,local_dir=model_dir)
    if a.prepare_inputs:
        if a.math500:rows=json.loads(a.math500.read_text())
        elif a.offline:rows=json.loads((a.out/'math500.json').read_text())
        else:
            rows=[]
            for offset in range(0,500,100):
                obj=json.loads(get(f'https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4%2FMATH-500&config=default&split=test&offset={offset}&length=100'))
                rows.extend(r['row'] for r in obj['rows'])
        canonical=json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        if sha(canonical)!=CANONICAL_MATH_SHA:raise RuntimeError('MATH-500 content differs from the frozen dataset')
        put_json(a.out/'math500.json',rows)
        import numpy as np
        order=np.random.default_rng(20261105).permutation(500).tolist()
        assert order[:8]==[434,176,147,141,145,395,494,295]
        ordered=[dict(rows[i],id=str(rows[i]['unique_id'])) for i in order]
        put_json(a.out/'math500_ordered.json',ordered);put_json(a.out/'math500_pilot.json',ordered[:8])
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
        def prefix(r):return tok.apply_chat_template([{'role':'user','content':r['problem']+'\n'+SUFFIX}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        cal=[prefix(r) for r in ordered[:16]];evaluation=[prefix(r) for r in ordered[16:48]]
        put_json(a.out/'math_prefixes_cal.json',cal);put_json(a.out/'math_prefixes_eval.json',evaluation)
        put_json(a.out/'input_manifest.json',{'order':order,'seed':20261105,'rng':'numpy.default_rng.permutation',
            'math_canonical_sha256':CANONICAL_MATH_SHA,'calibration_prefix_sha256':[sha(s.encode()) for s in cal],
            'evaluation_prefix_sha256':[sha(s.encode()) for s in evaluation]})
    put_json(a.out/'verification.json',{'repo_commit':COMMIT,'model_revision':REVISION,'edit_sha256':FILES['runs/p50c/edit/edit.npz']})
if __name__=='__main__':main()
