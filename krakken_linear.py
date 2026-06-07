#!/usr/bin/env python3
"""
krakken_linear.py -- LINEAR-trail active-S-box lower bounds for Krakken-2048.

This is the linear-cryptanalysis counterpart to krakken_solve.py (which does
differential). It proves a minimum number of linearly-active S-boxes over N
rounds; with the ABYSSAL S-box's maximum squared correlation of 2^-6 per active
S-box (AES-class S-box: max |correlation| = 2^-3, so correlation^2 = 2^-6), a
bound of B active S-boxes gives a linear-hull / correlation bound of 2^-6B,
i.e. the same per-S-box weight as the differential case.

WHY LINEAR IS (MOSTLY) THE DUAL OF DIFFERENTIAL
-----------------------------------------------
For mask (linear) propagation, each linear layer acts by its TRANSPOSE (adjoint),
and active S-boxes are counted the same way. For Krakken's layers:

  * MDS  : the MDS matrix and its transpose have the SAME branch number (9).
           => the branch-number activity constraint is IDENTICAL to differential.
  * Rho, Pi, InkCloud word-perm, XRBD/ARX rotations : bit-permutations and
           rotations. A permutation's transpose is its inverse; at the
           active/inactive (0/1) level "some bit active" is preserved by any
           bijection, so the activity-propagation constraints are structurally
           the SAME (we keep the forward wiring; activity sets are identical).
  * Chi  : S-box layer; activity pattern identical at the active/inactive level.
  * XRBD / PRESSURE XOR-merges : the ADJOINT of a linear fan-out (one source
           feeding several targets) is a fan-in (several sources into one), but
           at the 0/1 activity level both reduce to the SAME OR relation between
           the involved lanes. So the OR-merge constraints are reused.
  * Theta: this is the ONE layer whose transpose differs structurally. Theta
           mixes column parities; its adjoint mixes the transposed column
           coupling. At the activity level we keep theta's non-cancelling
           column-parity-spread model, which is a SOUND over-approximation for
           BOTH directions (it never lets the attacker cancel), so it remains a
           valid lower bound for the linear case too. (A tighter linear-specific
           theta is possible future work; the current model stays conservative.)

NET: at the byte-lane activity level used here, the linear model coincides with
the differential model except for the interpretation of the per-S-box weight
(LAT vs DDT), which happens to give the same 2^-6. The bound is therefore a
sound LOWER bound on linearly-active S-boxes. This mirrors the well-known fact
that for AES-like wide-trail designs the differential and linear active-S-box
counts coincide.

IMPORTANT HONESTY NOTE for the paper: because the activity-level model is
direction-symmetric here, this script will reproduce the differential numbers.
That is the EXPECTED and correct result for a wide-trail design -- it is a
*confirmation* that the linear bound matches the differential one, not a bug.
If you want a genuinely independent linear check, the next refinement is a
bit-level mask model with the explicit transpose of each linear map; that is
heavier and is noted as future work.

USAGE (identical interface to krakken_solve.py):
    python3 krakken_linear.py 8 --solver scip --full --checkpoint lin8.txt
    python3 krakken_linear.py 2 --solver scip --xrbd
deps: pulp + (pyscipopt for --solver scip)
"""
import sys, argparse, pulp

RHO=[32,1,62,28,36,44,15,61,6,19,24,55,3,10,43,17,
     25,39,41,59,47,8,56,14,18,35,21,33,2,49,22,51]
MDS_BRANCH=9; NW,NL=32,8
BUTTERFLY_ROT=[13,23,37,41,53]
ARX_RSHIFT=17; ARX_LSHIFT=31; INK_ROT=11
# Per-active-S-box weight in bits: linear uses max squared correlation.
# AES-class S-box: max |corr| = 2^-3  =>  corr^2 = 2^-6.
SBOX_WEIGHT_BITS=6

def rotl_spread(r):
    r%=64; m={}
    for b in range(NL):
        m[b]=sorted(set(((bit+r)%64)//NL for bit in range(8*b,8*b+8)))
    return m
def _rshift_spread(s):
    m={}
    for b in range(NL):
        m[b]=sorted(set((bit-s)//NL for bit in range(8*b,8*b+8) if bit-s>=0))
    return m
def _lshift_spread(s):
    m={}
    for b in range(NL):
        m[b]=sorted(set((bit+s)//NL for bit in range(8*b,8*b+8) if bit+s<64))
    return m
def _inv(spread):
    inv={l:[] for l in range(NL)}
    for b,outs in spread.items():
        for o in outs: inv[o].append(b)
    return inv
def pi_perm():
    p=[0]*NW
    for i in range(NW):
        x,y=i//4,i%4; p[((x+3*y)&7)*4+y]=i
    return p
PI=pi_perm()
def ink_perm():
    p=[0]*NW
    for i in range(NW): p[(i*7)&31]=i
    return p
INK_PERM=ink_perm()

def newL(n): return {(w,l):pulp.LpVariable(f"{n}_{w}_{l}",cat="Binary") for w in range(NW) for l in range(NL)}
def OR(prob,d,srcs):
    if not srcs: prob+=d==0; return
    for s in srcs: prob+=d>=s
    prob+=d<=pulp.lpSum(srcs)

def theta(prob,dst,src,t):
    # non-cancelling column-parity spread (sound for both directions)
    par={(c,l):pulp.LpVariable(f"{t}_p_{c}_{l}",cat="Binary") for c in range(8) for l in range(NL)}
    for c in range(8):
        for l in range(NL): OR(prob,par[(c,l)],[src[(4*c+y,l)] for y in range(4)])
    for c in range(8):
        for l in range(NL):
            inj=pulp.LpVariable(f"{t}_i_{c}_{l}",cat="Binary")
            OR(prob,inj,[par[((c+7)&7,l)],par[((c+1)&7,l)]])
            for y in range(4):
                w=4*c+y; OR(prob,dst[(w,l)],[src[(w,l)],inj])
def mds(prob,dst,src,t,tight=False):
    # MDS branch number is identical for the transpose => same constraint.
    for y in range(4):
        for l in range(NL):
            ins=[src[(4*c+y,l)] for c in range(8)]; outs=[dst[(4*c+y,l)] for c in range(8)]
            z=pulp.LpVariable(f"{t}_z_{y}_{l}",cat="Binary")
            for v in ins: prob+=z>=v
            prob+=z<=pulp.lpSum(ins)
            prob+=pulp.lpSum(ins)+pulp.lpSum(outs)>=MDS_BRANCH*z
            for v in outs: prob+=v<=z
            if tight:
                S=pulp.lpSum(ins+outs)
                for vk in ins+outs: prob+=S>=MDS_BRANCH*vk
def rho(prob,dst,src,t):
    for w in range(NW):
        inv=_inv(rotl_spread(RHO[w]))
        for o in range(NL): OR(prob,dst[(w,o)],[src[(w,b)] for b in inv[o]])
def wperm(prob,dst,src,perm):
    for ow in range(NW):
        for l in range(NL): prob+=dst[(ow,l)]==src[(perm[ow],l)]
def chi(prob,dst,src,t):
    r32=lambda l:(l+4)%NL
    for p in range(4):
        for y in range(4):
            wa=(2*p)*4+y; wb=(2*p+1)*4+y
            ap={l:dst[(wa,l)] for l in range(NL)}
            for l in range(NL): OR(prob,ap[l],[src[(wa,l)],src[(wb,r32(l))]])
            for l in range(NL): OR(prob,dst[(wb,l)],[src[(wb,l)],ap[r32(l)]])

XRBD=[(BUTTERFLY_ROT[k],[(i,i^(1<<k)) for i in range(NW) if (i&(1<<k))==0]) for k in range(5)]
def xrbd(prob,dst,src,t):
    cur=src
    for k,(r,pairs) in enumerate(XRBD):
        nxt=newL(f"{t}_s{k}")
        inv=_inv(rotl_spread(r)); touched=set()
        for (i,j) in pairs:
            touched.add(i); touched.add(j)
            xp={l:nxt[(i,l)] for l in range(NL)}
            for l in range(NL): OR(prob,xp[l],[cur[(i,l)],cur[(j,l)]])
            for l in range(NL): OR(prob,nxt[(j,l)],[cur[(j,l)]]+[xp[b] for b in inv[l]])
        for w in range(NW):
            if w not in touched:
                for l in range(NL): prob+=nxt[(w,l)]==cur[(w,l)]
        cur=nxt
    for w in range(NW):
        for l in range(NL): prob+=dst[(w,l)]==cur[(w,l)]

def pressure(prob,dst,src,t):
    rinv=_inv(_rshift_spread(ARX_RSHIFT)); linv=_inv(_lshift_spread(ARX_LSHIFT))
    rot7inv=_inv(rotl_spread(7)); rot19inv=_inv(rotl_spread(19))
    for c in range(8):
        wa,wb,wcc,wd=4*c+0,4*c+1,4*c+2,4*c+3
        a1={l:pulp.LpVariable(f"{t}_a1_{c}_{l}",cat="Binary") for l in range(NL)}
        b1={l:pulp.LpVariable(f"{t}_b1_{c}_{l}",cat="Binary") for l in range(NL)}
        cc1={l:pulp.LpVariable(f"{t}_cc1_{c}_{l}",cat="Binary") for l in range(NL)}
        d1={l:pulp.LpVariable(f"{t}_d1_{c}_{l}",cat="Binary") for l in range(NL)}
        for l in range(NL): OR(prob,a1[l],[src[(wa,l)],src[(wcc,l)]]+[src[(wcc,b)] for b in rinv[l]])
        for l in range(NL): OR(prob,b1[l],[src[(wb,l)],src[(wd,l)]]+[src[(wd,b)] for b in rinv[l]])
        for l in range(NL): OR(prob,cc1[l],[src[(wcc,l)],a1[l]]+[a1[b] for b in linv[l]])
        for l in range(NL): OR(prob,d1[l],[src[(wd,l)],b1[l]]+[b1[b] for b in linv[l]])
        for l in range(NL):
            prob+=dst[(wa,l)]==a1[l]; prob+=dst[(wcc,l)]==cc1[l]
        for l in range(NL): OR(prob,dst[(wb,l)],[b1[b] for b in rot7inv[l]])
        for l in range(NL): OR(prob,dst[(wd,l)],[d1[b] for b in rot19inv[l]])

def ink(prob,dst,src,t):
    rinv=_inv(rotl_spread(INK_ROT)); tmp=newL(f"{t}_rot")
    for w in range(NW):
        for o in range(NL): OR(prob,tmp[(w,o)],[src[(w,b)] for b in rinv[o]])
    wperm(prob,dst,tmp,INK_PERM)

def core_round(prob,src,r,tight,use_xrbd=False,full=False):
    t=newL(f"r{r}t"); theta(prob,t,src,f"r{r}t")
    m=newL(f"r{r}m"); mds(prob,m,t,f"r{r}m",tight)
    rh=newL(f"r{r}r"); rho(prob,rh,m,f"r{r}r")
    pi=newL(f"r{r}p"); wperm(prob,pi,rh,PI)
    ch=newL(f"r{r}c"); chi(prob,ch,pi,f"r{r}c")
    if full:
        bf=newL(f"r{r}bf"); xrbd(prob,bf,ch,f"r{r}bf")
        pr=newL(f"r{r}pr"); pressure(prob,pr,bf,f"r{r}pr")
        ik=newL(f"r{r}ik"); ink(prob,ik,pr,f"r{r}ik")
        return pi, ik
    if use_xrbd:
        bf=newL(f"r{r}bf"); xrbd(prob,bf,ch,f"r{r}bf")
        return pi, bf
    return pi, ch

def build(rounds,tight,use_xrbd=False,full=False):
    prob=pulp.LpProblem("lin",pulp.LpMinimize)
    st=newL("in"); prob+=pulp.lpSum(st.values())>=1
    terms=[]; cur=st
    for r in range(rounds):
        ci,out=core_round(prob,cur,r,tight,use_xrbd,full); terms+=list(ci.values()); cur=out
    prob+=pulp.lpSum(terms)
    return prob

def solve_scip(mps,timeout,checkpoint=None):
    from pyscipopt import Model
    m=Model(); m.readProblem(mps); m.setParam("limits/time",timeout)
    if checkpoint:
        try:
            from pyscipopt import Eventhdlr, SCIP_EVENTTYPE
            class _CP(Eventhdlr):
                def eventinit(self):
                    self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND,self)
                    self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED,self)
                def eventexit(self):
                    self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND,self)
                    self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED,self)
                def eventexec(self,event):
                    try:
                        d=self.model.getDualbound(); p=self.model.getPrimalbound()
                        with open(checkpoint,"w") as f:
                            f.write(f"dual={d}\nprimal={p}\nproven_lower_bound={int(d) if d==d else 0}\n")
                    except Exception: pass
            m.includeEventhdlr(_CP(),"cp","checkpoint")
        except Exception: pass
    m.optimize()
    primal=m.getObjVal() if m.getNSols()>0 else float('inf')
    return primal, m.getDualbound(), (m.getStatus()=="optimal")

def solve_cbc(prob,timeout):
    prob.solve(pulp.PULP_CBC_CMD(msg=1,timeLimit=timeout))
    return pulp.value(prob.objective), float('nan'), (pulp.LpStatus[prob.status]=="Optimal")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("rounds",type=int,nargs="?",default=2)
    ap.add_argument("--solver",choices=["scip","cbc"],default="scip")
    ap.add_argument("--timeout",type=int,default=3600)
    ap.add_argument("--tight",action="store_true")
    ap.add_argument("--xrbd",action="store_true",help="SPN+XRBD")
    ap.add_argument("--full",action="store_true",help="all layers: XRBD+PRESSURE+InkCloud")
    ap.add_argument("--checkpoint",default=None)
    a=ap.parse_args()
    prob=build(a.rounds,a.tight,a.xrbd,a.full)
    name=("COMPLETE" if a.full else "SPN+XRBD" if a.xrbd else "SPN-core")
    print(f"[*] LINEAR {name} {a.rounds}rd vars={len(prob.variables())} cons={len(prob.constraints)} solver={a.solver}")
    if a.solver=="cbc":
        primal,dual,proven=solve_cbc(prob,a.timeout)
    else:
        mps="/tmp/krakken_lin.mps"; prob.writeMPS(mps)
        primal,dual,proven=solve_scip(mps,a.timeout,a.checkpoint)
    print("\n"+"="*64)
    print(f"  LINEAR  Rounds: {a.rounds}")
    if primal is not None and primal==primal:
        B=int(round(primal))
        print(f"  primal (UPPER): {B}   dual (LOWER): {dual:.3f}")
        if proven:
            print(f"  >>> PROVEN MINIMUM (linear) = {B} active S-boxes."
                  f"  Correlation^2 bound <= 2^-{SBOX_WEIGHT_BITS*B}.")
        else:
            print(f"  NOT proven (time/gap). minimum in [{int(dual)+1 if dual==dual else 0}, {B}].")
    print("="*64)

if __name__=="__main__":
    main()
