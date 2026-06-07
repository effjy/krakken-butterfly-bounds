#!/usr/bin/env python3
"""
krakken_solve.py -- solve the Krakken SPN-core active-S-box MILP with a choice of
open-source solvers, reporting the TRUE lower bound and gap (no status-string lies).

Solvers (install as needed):
    --solver scip     pip install pyscipopt     (STRONGEST cuts -> best gap-closing; recommended)
    --solver highs    pip install highspy        (fastest LP; lighter cuts)
    --solver ortools   pip install ortools        (wraps SCIP; similar to scip)
    --solver cbc      (PuLP default; weakest, included for comparison)

The model (theta -> MDS -> rho -> pi -> chi) is identical to krakken_spn_core.py.
We build it in PuLP, export to MPS, and hand the MPS to the chosen solver's
native API so we get reliable bound/gap numbers.

WHAT TO READ IN THE OUTPUT
  - "primal (best feasible)"  = upper bound on the minimum (a real trail exists at this)
  - "dual (best bound)"       = lower bound proven so far
  - PROVEN iff dual rounds up to primal (gap effectively 0). Otherwise: minimum
    is in [ceil(dual), primal], unproven.

usage:
    python3 krakken_solve.py 2 --solver scip --gate     # must yield 27
    python3 krakken_solve.py 3 --solver scip --timeout 7200
deps: pulp + one of the solver packages
"""
import sys, argparse, pulp

RHO=[32,1,62,28,36,44,15,61,6,19,24,55,3,10,43,17,
     25,39,41,59,47,8,56,14,18,35,21,33,2,49,22,51]
MDS_BRANCH=9; NW,NL=32,8

def rotl_spread(r):
    r%=64; m={}
    for b in range(NL):
        m[b]=sorted(set(((bit+r)%64)//NL for bit in range(8*b,8*b+8)))
    return m
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
    par={(c,l):pulp.LpVariable(f"{t}_p_{c}_{l}",cat="Binary") for c in range(8) for l in range(NL)}
    for c in range(8):
        for l in range(NL): OR(prob,par[(c,l)],[src[(4*c+y,l)] for y in range(4)])
    for c in range(8):
        for l in range(NL):
            inj=pulp.LpVariable(f"{t}_i_{c}_{l}",cat="Binary")
            OR(prob,inj,[par[((c+7)&7,l)],par[((c+1)&7,l)]])
            for y in range(4):
                w=4*c+y; OR(prob,dst[(w,l)],[src[(w,l)],inj])
def mds(prob,dst,src,t,tight=True):
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
                for vk in ins+outs: prob += S >= MDS_BRANCH*vk
def rho(prob,dst,src,t):
    for w in range(NW):
        sp=rotl_spread(RHO[w]); inv={l:[] for l in range(NL)}
        for b,o in sp.items():
            for oo in o: inv[oo].append(b)
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

# --- XRBD butterfly (optional, --xrbd). 5 stages; rotations 13,23,37,41,53.
# Byte-lane OR-merge model, validated earlier against the C reference's
# single-lane propagation. Each crossover (i,j) with rotation r:
#   x' = x ^ y               -> i-lanes = OR(x lanes, y lanes)
#   y' = y ^ rotl(x', r)     -> j-lanes = OR(y lanes, spread_r(x' lanes))
BUTTERFLY_ROT=[13,23,37,41,53]
def _xrbd_stages():
    out=[]
    for k in range(5):
        d=1<<k
        out.append((BUTTERFLY_ROT[k],[(i,i^d) for i in range(NW) if (i&d)==0]))
    return out
XRBD=_xrbd_stages()
def xrbd(prob,dst,src,t):
    cur=src
    for k,(r,pairs) in enumerate(XRBD):
        nxt=newL(f"{t}_s{k}")
        spread=rotl_spread(r); inv={l:[] for l in range(NL)}
        for b,o in spread.items():
            for oo in o: inv[oo].append(b)
        touched=set()
        for (i,j) in pairs:
            touched.add(i); touched.add(j)
            xp={l:nxt[(i,l)] for l in range(NL)}
            for l in range(NL):
                OR(prob,xp[l],[cur[(i,l)],cur[(j,l)]])
            for l in range(NL):
                contrib=[cur[(j,l)]]+[xp[b] for b in inv[l]]
                OR(prob,nxt[(j,l)],contrib)
        for w in range(NW):
            if w not in touched:
                for l in range(NL): prob+=nxt[(w,l)]==cur[(w,l)]
        cur=nxt
    for w in range(NW):
        for l in range(NL): prob+=dst[(w,l)]==cur[(w,l)]

# --- PRESSURE (ARX) and InkCloud (optional, --full). Byte-lane OR-spread model,
# ported from the validated bytelane prover. '+=' modeled as OR (addition only
# spreads activity; ignoring carry keeps the bound a sound lower bound).
ARX_RSHIFT=17; ARX_LSHIFT=31; INK_ROT=11
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

def pressure(prob,dst,src,t):
    rinv=_inv(_rshift_spread(ARX_RSHIFT)); linv=_inv(_lshift_spread(ARX_LSHIFT))
    rot7inv=_inv(rotl_spread(7)); rot19inv=_inv(rotl_spread(19))
    for c in range(8):
        wa,wb,wcc,wd=4*c+0,4*c+1,4*c+2,4*c+3
        a1={l:pulp.LpVariable(f"{t}_a1_{c}_{l}",cat="Binary") for l in range(NL)}
        b1={l:pulp.LpVariable(f"{t}_b1_{c}_{l}",cat="Binary") for l in range(NL)}
        cc1={l:pulp.LpVariable(f"{t}_cc1_{c}_{l}",cat="Binary") for l in range(NL)}
        d1={l:pulp.LpVariable(f"{t}_d1_{c}_{l}",cat="Binary") for l in range(NL)}
        for l in range(NL):
            OR(prob,a1[l],[src[(wa,l)],src[(wcc,l)]]+[src[(wcc,b)] for b in rinv[l]])
        for l in range(NL):
            OR(prob,b1[l],[src[(wb,l)],src[(wd,l)]]+[src[(wd,b)] for b in rinv[l]])
        for l in range(NL):
            OR(prob,cc1[l],[src[(wcc,l)],a1[l]]+[a1[b] for b in linv[l]])
        for l in range(NL):
            OR(prob,d1[l],[src[(wd,l)],b1[l]]+[b1[b] for b in linv[l]])
        # post rotations: b=rotl(b1,7), d=rotl(d1,19); a,cc unchanged
        for l in range(NL):
            prob+=dst[(wa,l)]==a1[l]
            prob+=dst[(wcc,l)]==cc1[l]
        for l in range(NL):
            OR(prob,dst[(wb,l)],[b1[b] for b in rot7inv[l]])
        for l in range(NL):
            OR(prob,dst[(wd,l)],[d1[b] for b in rot19inv[l]])

def ink(prob,dst,src,t):
    rinv=_inv(rotl_spread(INK_ROT))
    tmp=newL(f"{t}_rot")
    for w in range(NW):
        for o in range(NL):
            OR(prob,tmp[(w,o)],[src[(w,b)] for b in rinv[o]])
    wperm(prob,dst,tmp,INK_PERM)

def core_round(prob,src,r,tight,use_xrbd=False,full=False):
    t=newL(f"r{r}t"); theta(prob,t,src,f"r{r}t")
    m=newL(f"r{r}m"); mds(prob,m,t,f"r{r}m",tight)
    rh=newL(f"r{r}r"); rho(prob,rh,m,f"r{r}r")
    pi=newL(f"r{r}p"); wperm(prob,pi,rh,PI)
    ch=newL(f"r{r}c"); chi(prob,ch,pi,f"r{r}c")
    # S-boxes are counted at the Chi input (pi); post-Chi layers only change how
    # activity propagates to the NEXT round.
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
    prob=pulp.LpProblem("spn",pulp.LpMinimize)
    st=newL("in"); prob+=pulp.lpSum(st.values())>=1
    terms=[]; cur=st
    for r in range(rounds):
        ci,out=core_round(prob,cur,r,tight,use_xrbd,full); terms+=list(ci.values()); cur=out
    prob+=pulp.lpSum(terms)
    return prob

def report(rounds,primal,dual,proven):
    print("\n"+"="*64)
    print(f"  Rounds: {rounds}")
    print(f"  primal (best feasible, UPPER bound): {primal}")
    print(f"  dual   (best bound,    LOWER bound): {dual:.3f}")
    if proven:
        B=int(round(primal))
        print(f"  >>> PROVEN MINIMUM = {B} active S-boxes. DP <= 2^-{6*B}.")
    else:
        lo=int(dual) if dual==dual else 0
        print(f"  >>> NOT PROVEN. minimum is in [{lo+1 if lo<primal else primal}, {int(primal)}].")
        print(f"      (Need dual to reach primal. Bigger solver / tighter model required.)")
    print("="*64)

# ---------- solver backends ----------
def solve_scip(mps, timeout, emphasis=False, checkpoint=None):
    from pyscipopt import Model
    m=Model(); m.readProblem(mps); m.setParam("limits/time",timeout)
    if emphasis:
        # Spend effort PROVING the bound, not re-finding known trails:
        #  - optimality emphasis biases toward dual/cut work
        #  - crank cut generation (cuts are what close this kind of gap)
        #  - throttle primal heuristics (they keep rediscovering the 55-trail)
        try:
            from pyscipopt import SCIP_PARAMEMPHASIS as E
            m.setEmphasis(E.OPTIMALITY)
        except Exception:
            pass
        try:
            m.setParam("separating/maxroundsroot", -1)  # unlimited root cut rounds
            m.setParam("separating/maxrounds", -1)       # unlimited cut rounds at nodes
            m.setHeuristics(0)                           # heuristics OFF (0 = SCIP_PARAMSETTING.OFF)
        except Exception:
            pass
    if checkpoint:
        # write best bounds to disk on every improvement, so Ctrl-C still leaves a record
        try:
            from pyscipopt import Eventhdlr, SCIP_EVENTTYPE
            class _CP(Eventhdlr):
                def eventinit(self):
                    self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
                    self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)
                def eventexit(self):
                    self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
                    self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)
                def eventexec(self, event):
                    try:
                        d=self.model.getDualbound(); p=self.model.getPrimalbound()
                        with open(checkpoint,"w") as f:
                            f.write(f"dual={d}\nprimal={p}\n"
                                    f"proven_lower_bound={int(d) if d==d else 0}\n"
                                    f"range=[{int(d)+1 if d==d else 0}, {int(p) if p==p else 'inf'}]\n")
                    except Exception:
                        pass
            m.includeEventhdlr(_CP(), "cp", "checkpoint best bounds")
        except Exception:
            pass
    m.optimize()
    primal=m.getObjVal() if m.getNSols()>0 else float('inf')
    dual=m.getDualbound()
    proven=(m.getStatus()=="optimal")
    return primal,dual,proven
def solve_highs(mps, timeout):
    import highspy
    h=highspy.Highs(); h.readModel(mps)
    h.setOptionValue("time_limit",float(timeout)); h.run()
    info=h.getInfo()
    primal=info.objective_function_value if h.getNumSol()>0 else float('inf')
    dual=info.mip_dual_bound if hasattr(info,'mip_dual_bound') else float('nan')
    proven=(str(h.getModelStatus())=="HighsModelStatus.kOptimal")
    return primal,dual,proven
def solve_ortools(mps, timeout):
    from ortools.linear_solver import pywraplp
    solver=pywraplp.Solver.CreateSolver("SCIP")
    with open(mps) as f: solver.ImportModelFromMps(f.read()) if hasattr(solver,'ImportModelFromMps') else None
    # OR-Tools mps import is limited; fall back to note
    raise NotImplementedError("Use --solver scip (OR-Tools' backend is SCIP anyway).")
def solve_cbc(prob, timeout):
    prob.solve(pulp.PULP_CBC_CMD(msg=1,timeLimit=timeout))
    primal=pulp.value(prob.objective)
    return primal, float('nan'), (pulp.LpStatus[prob.status]=="Optimal")  # CBC gap unreliable here

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("rounds",type=int,nargs="?",default=2)
    ap.add_argument("--solver",choices=["scip","highs","ortools","cbc"],default="scip")
    ap.add_argument("--timeout",type=int,default=3600)
    ap.add_argument("--tight",action="store_true",help="add convex-hull MDS lifting")
    ap.add_argument("--xrbd",action="store_true",
                    help="include the XRBD butterfly layer (SPN+XRBD model)")
    ap.add_argument("--full",action="store_true",
                    help="include ALL post-Chi layers: XRBD + PRESSURE (ARX) + InkCloud")
    ap.add_argument("--gate",action="store_true",help="assert 2-round result == 27")
    ap.add_argument("--emphasis",action="store_true",
                    help="SCIP: prioritize proving the bound (cuts up, heuristics off)")
    ap.add_argument("--checkpoint",default=None,
                    help="SCIP: file to write best dual/primal to on every improvement")
    a=ap.parse_args()

    prob=build(a.rounds,a.tight,a.xrbd,a.full)
    nv=len(prob.variables()); nc=len(prob.constraints)
    model_name = ("COMPLETE round (SPN+XRBD+PRESSURE+InkCloud)" if a.full
                  else "SPN+XRBD" if a.xrbd else "SPN-core")
    print(f"[*] {model_name} {a.rounds}rd  vars={nv} cons={nc}  solver={a.solver} tight={a.tight}")

    if a.solver=="cbc":
        primal,dual,proven=solve_cbc(prob,a.timeout)
    elif a.solver=="scip":
        mps="/tmp/krakken_spn.mps"; prob.writeMPS(mps)
        primal,dual,proven=solve_scip(mps,a.timeout,emphasis=a.emphasis,checkpoint=a.checkpoint)
    else:
        mps="/tmp/krakken_spn.mps"; prob.writeMPS(mps)
        primal,dual,proven={"highs":solve_highs,"ortools":solve_ortools}[a.solver](mps,a.timeout)

    report(a.rounds,primal,dual,proven)
    if a.gate and a.rounds==2:
        ok = abs(primal-27)<0.5
        print("[GATE] 2-round == 27:", "PASS" if ok else f"*** FAIL got {primal} ***")

if __name__=="__main__":
    main()
