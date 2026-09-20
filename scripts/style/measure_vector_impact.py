"""
Measure whether the steering VECTOR or the control LAW moves the image more.

The main sweep varies the control law on one set of steering vectors; this
answers the prior question of whether that is even the dominant axis. Three
distances, all on the same prompts at the same seeds:

  A) how far each arm sits from the unsteered baseline, split into the erased
     band (should be far) and the preserved band (should be near);
  B) LPIPS between the same law's images under the two vector families -- the
     direct cost of swapping the .pt file;
  C) LPIPS between two laws under one vector family -- the cost of swapping the
     controller.

If (B) exceeds (C), conclusions drawn about control laws are smaller than the
variation hiding in vector construction, and the vector has to be reported as a
factor rather than assumed as a constant.

Run after scripts/style/run_vector_ablation.sh has produced the *_teca_* arms:

    PYTHONPATH=. python scripts/style/measure_vector_impact.py
"""
import os
import sys

import lpips
import numpy as np
import torch
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

fn = lpips.LPIPS(net='alex').cuda()
def load(p):
    a=np.asarray(Image.open(p).convert('RGB').resize((512,512))).astype(np.float32)/127.5-1.0
    return torch.tensor(a).permute(2,0,1).unsqueeze(0).cuda()
def d(a,b):
    with torch.no_grad(): return fn(load(a),load(b)).item()
ROOT='results/sd14/style'   # baseline lives here, vector-set-independent
SETS={'Van Gogh':('vangogh',[20,21,22,23,24],[0,40,60,80,81]),
      'Kelly McKernan':('kelly',[60,61,62,63,64],[0,20,40,80,81])}
ARMS=['casteer','pid','adaptive_kg']
def dirn(arm,slug,teca):
    sub = 'teca_vectors' if teca else 'casteer_vectors'
    return f"{ROOT}/{sub}/{arm}{'_teca' if teca else ''}_{slug}/all"
for artist,(slug,er,ke) in SETS.items():
    print(f"\n{'='*76}\n{artist}\n{'='*76}")
    print("A) Distance from the unsteered baseline (LPIPS, mean over 5 cases per band)")
    print(f"{'':16}{'--- orig sv ---':>24}{'--- TECA sv ---':>24}")
    print(f"{'arm':<16}{'erased':>12}{'preserved':>12}{'erased':>12}{'preserved':>12}")
    for arm in ARMS:
        cells=[]
        for teca in (False,True):
            dd=dirn(arm,slug,teca)
            for grp in (er,ke):
                s=[d(f"{ROOT}/baseline_{slug}/all/{c}.png", f"{dd}/{c}.png") for c in grp if os.path.exists(f"{dd}/{c}.png")]
                cells.append(np.mean(s) if s else float('nan'))
        print(f"{arm:<16}{cells[0]:>12.3f}{cells[1]:>12.3f}{cells[2]:>12.3f}{cells[3]:>12.3f}")
    print("\nB) VECTOR SWAP: LPIPS(orig sv image, TECA sv image) -- same law, same seed")
    print(f"{'arm':<16}{'erased':>12}{'preserved':>12}")
    vsw=[]
    for arm in ARMS:
        vals=[]
        for grp in (er,ke):
            s=[d(f"{dirn(arm,slug,False)}/{c}.png", f"{dirn(arm,slug,True)}/{c}.png") for c in grp
               if os.path.exists(f"{dirn(arm,slug,True)}/{c}.png")]
            vals.append(np.mean(s) if s else float('nan'))
        vsw += vals
        print(f"{arm:<16}{vals[0]:>12.3f}{vals[1]:>12.3f}")
    print("\nC) LAW SWAP: LPIPS between two control laws -- same vector, same seed")
    print(f"{'pair':<26}{'orig sv':>12}{'TECA sv':>12}")
    lsw=[]
    for a,b in [('casteer','pid'),('casteer','adaptive_kg'),('pid','adaptive_kg')]:
        out=[]
        for teca in (False,True):
            da,db=dirn(a,slug,teca),dirn(b,slug,teca)
            s=[d(f"{da}/{c}.png", f"{db}/{c}.png") for c in er+ke
               if os.path.exists(f"{da}/{c}.png") and os.path.exists(f"{db}/{c}.png")]
            out.append(np.mean(s) if s else float('nan'))
        lsw += out
        print(f"{a+' vs '+b:<26}{out[0]:>12.3f}{out[1]:>12.3f}")
    print(f"\n  >> mean vector-swap distance {np.nanmean(vsw):.3f}   vs   mean law-swap distance {np.nanmean(lsw):.3f}"
          f"   ({np.nanmean(vsw)/max(np.nanmean(lsw),1e-9):.1f}x)")
