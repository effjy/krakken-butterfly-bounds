# Krakken‑2048 – Provable Differential & Linear Bounds (XRBD Layer)

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper.pdf)  
**📄 Click here to read the full paper:** [`paper.pdf`](paper.pdf)

This repository contains the **complete, reproducible MILP models** and proofs for the paper  

> **Provable Differential and Linear Bounds for a Butterfly‑Diffusion Permutation: The XRBD Layer and Krakken‑2048**  

The paper introduces the XOR‑Rotation Butterfly Diffusion (XRBD) layer and proves – with a closed optimality gap – a minimum of **229 active S‑boxes** over the eight‑round Krakken‑2048 permutation, bounding any single differential or linear characteristic by **2⁻¹³⁷⁴**.

## Repository contents

| File | Description |
|------|-------------|
| [`paper.pdf`](paper.pdf) | The full paper (IACR preprint style, all proofs and specifications) |
| `krakken_solve.py` | MILP for **differential** trails (active‑S‑box lower bound) |
| `krakken_linear.py` | MILP for **linear** trails (active‑S‑box lower bound) |
| *(other files)* | Supplementary artifacts (test vectors, constant verification) |

## The Krakken‑2048 round layers (explained)

The permutation operates on a **2048‑bit** state: 32 words of 64 bits, arranged as 8 columns × 4 rows. One round applies the following steps in order:

| Layer | Type | What it does (byte‑lane activity level) |
|-------|------|-------------------------------------------|
| **Theta** | Linear diffusion | Spreads activity across columns using column parities (non‑cancelling over‑approximation). |
| **MDS** | Linear diffusion (GF(2⁸)) | Circulant matrix with branch number **9** – the core of the wide‑trail strategy. Activity constraint: *zero or ≥9 active bytes per row*. |
| **Rho** | Word rotation | Each 64‑bit word rotated left by a round‑dependent amount. At byte‑lane level it spreads bits across byte lanes. |
| **Pi** | Word permutation | Permutes the 32 words (bijection, row‑preserving). |
| **Chi** | Non‑linear (S‑box) | Applies the **ABYSSAL** 8‑bit S‑box (Δ‑uniformity 4, max squared correlation 2⁻⁶) with cross‑lane coupling. **S‑boxes are counted here.** |
| **XRBD** | Butterfly diffusion | 5 stages of XOR‑rotate crossovers (distances 1,2,4,8,16) that make every output word depend on every input word (optimal depth). Destroys lane‑aligned trails. |
| **Pressure** | ARX | Intra‑column addition, shift, and rotation (modelled as activity‑spreading OR – sound lower bound). |
| **InkCloud** | Word shuffle & rotation | Rotates each word left by 11 bits, then scatters via multiplier 7 (bijection). |

The MILP models capture the **propagation of activity** (active / inactive) through these layers. The objective is to minimise the total number of active S‑boxes under the constraint that the input difference/mask is non‑zero.

## Reproducing the bounds

### Prerequisites

- Python 3.8+
- `pulp` (PuLP linear programming)
- A solver – **SCIP** is strongly recommended (closes the gap to proven optimality).  
  Install: `pip install pyscipopt`

---

## 1. Differential bounds (`krakken_solve.py`)

This script models the **differential** case. It returns the **minimum number of active S‑boxes** for a given number of rounds and layer configuration.

### All command‑line switches

| Switch | Description |
|--------|-------------|
| `rounds` (positional) | Number of rounds (e.g., `2`, `3`, `8`). Default = 2. |
| `--solver {scip,highs,ortools,cbc}` | Choose solver. **SCIP** is recommended for proving optimality. Default = `scip`. |
| `--timeout N` | Time limit in seconds. Default = 3600. |
| `--tight` | Add convex‑hull MDS lifting (stronger constraints). Always use for proofs. |
| `--xrbd` | Include the XRBD butterfly layer (SPN+XRBD model). |
| `--full` | Include **all** post‑Chi layers: XRBD + Pressure + InkCloud. This is the full permutation. |
| `--gate` | For 2‑round SPN core, assert that the result equals 27 (used for validation). |
| `--emphasis` | (SCIP only) Prioritise proving the bound: increase cut generation, disable primal heuristics. |
| `--checkpoint FILE` | (SCIP only) Write best dual/primal bounds to `FILE` on every improvement (useful for long runs). |

### Examples – reproducing the paper’s tables

#### Table 1: SPN core vs SPN+XRBD (2 and 3 rounds)

```bash
# 2 rounds, SPN core (no XRBD) → expects 27
python3 krakken_solve.py 2 --solver scip --tight

# 2 rounds, SPN + XRBD → expects 37
python3 krakken_solve.py 2 --solver scip --tight --xrbd

# 3 rounds, SPN core → expects 55
python3 krakken_solve.py 3 --solver scip --tight

# 3 rounds, SPN + XRBD → expects 69
python3 krakken_solve.py 3 --solver scip --tight --xrbd
```

#### Table 2: Full eight‑round bound

```bash
# 8 rounds, all layers (XRBD + Pressure + InkCloud) → expects 229
python3 krakken_solve.py 8 --solver scip --tight --full --timeout 7200
```

#### Using the checkpoint and emphasis options (for hard instances)

```bash
# 7 rounds, full permutation, with checkpoint and optimality emphasis
python3 krakken_solve.py 7 --solver scip --tight --full --timeout 14400 --emphasis --checkpoint bounds7.txt
```

### Understanding the output

- **`primal (best feasible)`** = upper bound on the minimum (a real trail exists with this many active S‑boxes).
- **`dual (best bound)`** = lower bound proven so far.
- **PROVEN** if the dual rounds up to the primal → gap = 0.00% → the number is the **true minimum** active S‑box count.

Example proven output:
```
>>> PROVEN MINIMUM = 229 active S-boxes. DP <= 2^-1374.
```

---

## 2. Linear bounds (`krakken_linear.py`)

The linear model uses the **same activity‑propagation constraints** (because for wide‑trail designs the differential and linear counts coincide at the byte‑lane level). The per‑S‑box weight is taken as the **maximum squared correlation** = 2⁻⁶, so the bound on any single linear characteristic is 2⁻⁶ᵇ.

### All command‑line switches (linear version)

| Switch | Description |
|--------|-------------|
| `rounds` (positional) | Number of rounds. Default = 2. |
| `--solver {scip,cbc}` | Solver choice. SCIP strongly recommended. Default = `scip`. |
| `--timeout N` | Time limit in seconds. Default = 3600. |
| `--tight` | Add convex‑hull MDS lifting. |
| `--xrbd` | Include XRBD butterfly layer (SPN+XRBD model). |
| `--full` | Include **all** post‑Chi layers (XRBD + Pressure + InkCloud). |
| `--checkpoint FILE` | (SCIP only) Write best bounds to `FILE` periodically. |

> **Note**: The linear script does **not** have `--gate` or `--emphasis` (it uses a simpler cut strategy, but still proves the bound with SCIP).

### Examples – linear bounds

```bash
# 2 rounds, SPN + XRBD → expects 37 (same as differential)
python3 krakken_linear.py 2 --solver scip --tight --xrbd

# 8 rounds, full permutation → expects 229
python3 krakken_linear.py 8 --solver scip --tight --full --timeout 7200

# 3 rounds, SPN core only (no XRBD) → expects 55
python3 krakken_linear.py 3 --solver scip --tight
```

### Linear output interpretation

The output is identical to the differential script, except the bound is stated as a **correlation² bound**:

```
>>> PROVEN MINIMUM (linear) = 229 active S-boxes. Correlation^2 bound <= 2^-1374.
```

---

## Verifying the paper’s claims

The paper reports the following exact values (all proven with gap 0.00%):

| Rounds | SPN core (differential) | SPN+XRBD | Full permutation (all layers) |
|--------|------------------------|----------|-------------------------------|
| 1      | 5                      | 5        | 5                             |
| 2      | 27                     | **37**   | –                             |
| 3      | 55                     | **69**   | –                             |
| 4      | –                      | –        | 101                           |
| 5      | –                      | –        | 133                           |
| 6      | –                      | –        | 165                           |
| 7      | –                      | –        | 197                           |
| 8      | –                      | –        | **229**                       |

To confirm each entry, run the corresponding command with `--solver scip --tight` (and `--xrbd` or `--full` as needed). The solver output will show `PROVEN MINIMUM = X` when the gap closes.

## Citation

If you use this repository or the results in your own work, please cite the paper:

```bibtex
@unpublished{lachance2026krakken,
  author = {Jean-François Lachance-Caumartin},
  title = {Provable Differential and Linear Bounds for a Butterfly-Diffusion Permutation: The XRBD Layer and Krakken-2048},
  note = {Figshare preprint},
  year = {2026},
  doi = {10.6084/m9.figshare.32599689}
}
```

**DOI:** [10.6084/m9.figshare.32599689](https://doi.org/10.6084/m9.figshare.32599689)

## License

The code and documentation are provided under the **MIT License** (see `LICENSE`). The paper is under **CC BY 4.0**.

## Contact & further work

Open issues for questions or to report any discrepancy. Independent cryptanalysis, division‑property MILP, and differential‑cluster estimates are welcome – see the “Open problems” section of the paper.
