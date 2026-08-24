# 🇮🇳 IndSolve — Indigenous Mathematical Optimization Engine
Problem Statement ID: 26119 (Mangalore Refinery and Petrochemicals Limited)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Math](https://img.shields.io/badge/Math-First%20Principles-orange.svg)]()
[![SIH-2026](https://img.shields.io/badge/SIH-Problem%2026119-green.svg)]()
[![Industry](https://img.shields.io/badge/Industry-MRPL%20Refinery-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

> **A from-scratch mathematical optimization solver core designed for India's strategic industrial infrastructure.**  
> Solves Linear Programming (LP) and Mixed-Integer Linear Programming (MILP) models arising in refinery crude blending, power grid economic dispatch, and supply chain logistics without any dependency on foreign commercial solvers.

---

## ⏱️ 60–90 Second SIH Judge Demo Script

| Time | Action on Screen | Spoken Script |
|---|---|---|
| **00:00 – 00:15** | Open **Scenario Lab** (Home view) | *"Welcome to IndSolve, an indigenous mathematical optimization workbench built for MRPL Problem Statement 26119. We are demonstrating a 100,000 bpd refinery procurement optimization subject to quality, supplier quotas, and coker limits."* |
| **00:15 – 00:35** | Point to the 3 KPI cards & baseline mix | *"At the baseline operating point, our custom Simplex engine finds the optimal crude slate in under 5 ms: $7.139M daily cost ($71.39/bbl), hitting the 1.20% blend sulfur ceiling exactly with zero slack."* |
| **00:35 – 00:55** | Select preset: **Middle East Escalation** | *"Now we simulate a market shock: Arabian Light spikes +$15/bbl. The solver automatically reallocates 15,000 bpd into alternative sweet and sour grades, absorbing the price increase while guaranteeing feed sulfur feasibility."* |
| **00:55 – 01:15** | Expand **Independent Constraint Audit** | *"To prove zero false claims, every solution passes our independent constraint audit, verifying all 8 physical bounds and recalculating $c^T x$ with zero violations."* |
| **01:15 – 01:30** | Switch to **Validation** / **Import Model** | *"Finally, our 23-test verification suite confirms exact agreement against published rational optimums on Netlib AFIRO and SciPy HiGHS."* |

---

## 🏛️ System Architecture

```
                                  ┌────────────────────────┐
                                  │   Industrial Problem   │
                                  │  (Refinery, Grid, MPS) │
                                  └───────────┬────────────┘
                                              │
                                  ┌───────────▼────────────┐
                                  │    Transparent Presolve│
                                  │ (Fixed Var, Row Prune) │
                                  └───────────┬────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     │                                                 │
          ┌──────────▼──────────┐                           ┌──────────▼──────────┐
          │  Continuous LP Core │                           │      MILP Engine    │
          │  (Tableau Simplex)  │                           │  (Branch-and-Bound) │
          └──────────┬──────────┘                           └──────────┬──────────┘
                     │                                                 │
      ┌──────────────┴──────────────┐                   ┌──────────────┴──────────────┐
      │ • Big-M Artificial Vars     │                   │ • Most Fractional Branching │
      │ • Bland's Anti-Cycling      │                   │ • Pruning by Bound/Infeas   │
      │ • Affine Bounds Shift       │                   │ • Node Limit Status Guard   │
      └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              │
                                  ┌───────────▼────────────┐
                                  │  Execution Acceleration│
                                  │  (Native CPU Numba JIT)│
                                  └───────────┬────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     │                                                 │
          ┌──────────▼──────────┐                           ┌──────────▼──────────┐
          │  Interactive Web UI │                           │ Verification Suite  │
          │ (Streamlit Dashboard│                           │ (SciPy / Koch 2004) │
          └─────────────────────┘                           └─────────────────────┘
```

---

## 🛣️ Phased Engineering & Performance Roadmap

| Phase | Milestone Name | Architecture & Linear Algebra | Target Scale & Hardware |
|---|---|---|---|
| **Phase 1 (Delivered)** | **Current Core Engine** | Dense Tableau Simplex, DFS Branch-and-Bound, 3 Safe Presolve Reductions, Bland's Tie-Breaking. | Small-to-medium dense models ($< 250$ vars/rows), CPU with Numba JIT. |
| **Phase 2 (Q4 2026)** | **Sparse Revised Simplex** | Compressed Sparse Column (CSC) matrices, Sparse LU Factorization with Forest-Tomlin updates, Gomory cuts. | $10,000+$ variables with $<1\%$ density. |
| **Phase 3 (2027)** | **GPU-Assisted Acceleration** | CUDA cuSPARSE, SYCL, and ROCm kernels for sparse matrix-vector products and Interior Point (IPM) Cholesky solves. | Refinery-scale enterprise models ($100,000+$ constraints), Heterogeneous GPUs. |

---

## 🚀 Quickstart & Reproduction

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/IndSolve.git
cd IndSolve
pip install -r requirements.txt
```

### 2. Run Mathematical Verification Suite (CLI)
```bash
python run_verification.py
```
*Executes all 23 unit tests across 6 problem families and exits with code 0 on complete pass.*

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📊 Illustrative Industrial Case Studies Included

1. **MRPL 2D Refinery Blending Walkthrough**: Interactive geometric polytope visualization showing corner-to-corner simplex pivots under feed sulfur limits (illustrative parameters).
2. **MRPL Multi-Crude 100,000 bpd Slate**: Feedstock optimization across Arabian Light, Brent, Bonny Light, Dubai Sour, and Basra Heavy crudes subject to blend sulfur, coker limits, and supplier quotas (illustrative parameters).
3. **National Power Grid Dispatch**: Merit-order economic dispatch across Coal, Solar, Hydro, and Gas units satisfying 1,200 MW demand with regional carbon emission cap constraints.
4. **Indian Multi-City Facility Location & Logistics (MILP)**: Classic Fixed-Charge Network Design deciding binary opening of Mumbai, Delhi, and Chennai hubs with continuous freight allocation under capacity and demand constraints.

---

## ⚖️ Verification vs External Reference Solvers & Standard Benchmarks

| Problem / Benchmark | IndSolve Objective | External Reference (SciPy HiGHS / milp) | Published Rational Optimum (Koch 2004) | Status / Agreement |
|---|---|---|---|---|
| **MRPL 2D Blend** | `$75,300.00` | `$75,300.00` | `$75,300.00` | ✅ Exact Match (`diff < 1e-7`) |
| **MRPL 5-Crude Slate** | `$7,143,333.33` | `$7,143,333.33` | `$7,143,333.33` | ✅ Exact Match (`diff < 1e-7`) |
| **Power Grid Dispatch** | `₹26,75,000.00` | `₹26,75,000.00` | `₹26,75,000.00` | ✅ Exact Match (`diff = 0.000`) |
| **Facility Location MILP**| `₹38,500.00` | `₹38,500.00` | `₹38,500.00` | ✅ Exact Integer Optimum (`diff = 0.000`) |
| **Netlib AFIRO (MPS)** | `-464.75314286` | `-464.75314286` | `-464.75314286` | ✅ Exact Agreement (`diff = 1.15e-9`) |

### 📁 Validated Continuous-LP MPS Subset Specification
IndSolve v0.2 implements a safe, validated parser for continuous-LP MPS models:
- **Supported Sections**: `NAME`, `ROWS`, `COLUMNS`, `RHS`, `BOUNDS`, `ENDATA`.
- **Supported Row Types**: `N` (Objective), `L` ($\le$), `G` ($\ge$), `E` ($=$).
- **Supported Bounds**: `LO` (Lower Bound), `UP` (Upper Bound), `FX` (Fixed Variable), `FR` (Free Variable $-\infty < x < \infty$).
- **Explicitly Rejected with Actionable Errors**:
  - Integer markers: `INTORG` / `INTEND` (discrete variables).
  - Discrete bound types: `BV` (Binary), `LI` (Integer lower), `UI` (Integer upper), `SC`, `SI`.
  - Advanced sections: `RANGES`, `SOS`, `QUADOBJ`, `QCMATRIX`, `INDICATORS`.

### 📁 Netlib AFIRO Benchmark Provenance
- **Official Source**: Netlib LP Problem Library ([netlib.sandia.gov/lp/data/afiro](https://netlib.sandia.gov/lp/data/afiro)) & HiGHS test suite.
- **SHA-256 Checksum**: `9cd304f02717cbd6f85068cb777b69d28539b22a4868ae0f0fb425f514f0eea5`
- **Model Structure**: 27 constraint rows (8 Equality, 19 Upper-Bound), 32 structural variables, 83 constraint non-zeros (9.61% matrix density; 88 total non-zeros including the 5 objective coefficients).
- **Exact Reference**: Thorsten Koch (2004), *The Final NETLIB-LP Results*, Operations Research Letters.

---

## 👥 Team & Submission Info
- **Event**: Smart India Hackathon (SIH) 2026 — Pre-Qualifier
- **Problem Statement ID**: 26119
- **Organization**: Mangalore Refinery and Petrochemicals Limited (MRPL)
- **Category**: Software / Mathematical Optimization Engine
