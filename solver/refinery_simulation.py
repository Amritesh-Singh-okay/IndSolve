"""
IndSolve — Refinery What-If Decision Support Simulation Engine
Simulates dynamic crude slate procurement under price volatility, availability reductions,
coker/heavy-feed limits, and environmental feed sulfur limits.
References MRPL LP procurement modeling methodologies (Mangalore Refinery and Petrochemicals Limited).
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from .tableau_simplex import SimplexSolver


# Baseline Crude Feedstock Parameters (Illustrative Model Parameters)
BASE_CRUDES = {
    "Arabian_Light": {"price": 72.0, "sulfur": 1.77, "api": 33.4, "max_avail": 40000.0, "type": "Medium Sour"},
    "Brent":         {"price": 78.0, "sulfur": 0.37, "api": 38.3, "max_avail": 35000.0, "type": "Light Sweet"},
    "Bonny_Light":   {"price": 75.0, "sulfur": 0.13, "api": 33.4, "max_avail": 30000.0, "type": "Ultra-Sweet"},
    "Dubai_Sour":    {"price": 68.0, "sulfur": 2.00, "api": 31.0, "max_avail": 45000.0, "type": "Heavy Sour"},
    "Basra_Heavy":   {"price": 58.0, "sulfur": 3.50, "api": 24.7, "max_avail": 25000.0, "type": "High-Sulfur Heavy"},
}


def build_refinery_model(
    throughput: float,
    max_sulfur_pct: float,
    price_adjustments: Dict[str, float],
    avail_adjustments: Dict[str, float],
    carbon_penalty_per_bbl_sour: float = 0.0,
    heavy_limit_bpd: float = 25000.0,
) -> Dict[str, Any]:
    """
    Constructs the exact canonical LP matrices for MRPL refinery crude procurement.
    """
    crude_keys = list(BASE_CRUDES.keys())
    n = len(crude_keys)

    # Cost vector c (Base price + spot adjustment + hypothetical policy carbon penalty on sour crude)
    c = np.zeros(n, dtype=np.float64)
    for i, key in enumerate(crude_keys):
        base = BASE_CRUDES[key]
        p = base["price"] + price_adjustments.get(key, 0.0)
        if base["sulfur"] > 1.0:
            p += carbon_penalty_per_bbl_sour
        c[i] = p

    # Constraints A_ub @ x <= b_ub
    # 1. Sulfur ceiling: sum(sulfur_i * x_i) <= max_sulfur_pct * throughput
    # 2. Coker / Heavy crude limit: Basra_Heavy <= heavy_limit_bpd
    # 3..7. Supplier availability limits: x_i <= avail_i
    A_ub_rows = []
    b_ub_rows = []
    constraint_labels_ub = []

    # Row 0: Sulfur ceiling
    sulfur_coeffs = [BASE_CRUDES[k]["sulfur"] for k in crude_keys]
    A_ub_rows.append(sulfur_coeffs)
    b_ub_rows.append(max_sulfur_pct * throughput)
    constraint_labels_ub.append(f"Blended Feed Sulfur Ceiling ({max_sulfur_pct:.2f}% max)")

    # Row 1: Heavy crude coker limit
    heavy_row = [0.0] * n
    heavy_idx = crude_keys.index("Basra_Heavy")
    heavy_row[heavy_idx] = 1.0
    A_ub_rows.append(heavy_row)
    b_ub_rows.append(heavy_limit_bpd)
    constraint_labels_ub.append(f"Basra Heavy Coker Unit Limit ({heavy_limit_bpd:,.0f} bpd)")

    # Rows 2..N+1: Supplier availability limits
    for i, key in enumerate(crude_keys):
        row = [0.0] * n
        row[i] = 1.0
        max_v = avail_adjustments.get(key, BASE_CRUDES[key]["max_avail"])
        A_ub_rows.append(row)
        b_ub_rows.append(max_v)
        constraint_labels_ub.append(f"{key.replace('_', ' ')} Supplier Quota ({max_v:,.0f} bpd)")

    # Equality constraint A_eq @ x == b_eq: Total crude throughput == throughput
    A_eq = np.ones((1, n), dtype=np.float64)
    b_eq = np.array([float(throughput)], dtype=np.float64)
    constraint_labels_eq = [f"Total Distillation Target ({throughput:,.0f} bpd)"]

    # Variable bounds [(0, avail_i)]
    bounds = [(0.0, avail_adjustments.get(k, BASE_CRUDES[k]["max_avail"])) for k in crude_keys]

    return {
        "c": c,
        "A_ub": np.array(A_ub_rows, dtype=np.float64),
        "b_ub": np.array(b_ub_rows, dtype=np.float64),
        "A_eq": A_eq,
        "b_eq": b_eq,
        "bounds": bounds,
        "var_names": crude_keys,
        "crude_keys": crude_keys,
        "throughput": throughput,
        "max_sulfur_pct": max_sulfur_pct,
        "heavy_limit_bpd": heavy_limit_bpd,
        "constraint_labels_ub": constraint_labels_ub,
        "constraint_labels_eq": constraint_labels_eq,
    }


def run_what_if_simulation(
    base_throughput: float,
    base_sulfur: float,
    what_if_throughput: float,
    what_if_sulfur: float,
    price_adjustments: Dict[str, float],
    avail_adjustments: Dict[str, float],
    carbon_penalty: float = 0.0,
    heavy_limit_bpd: float = 25000.0,
    tolerance: float = 1e-7,
    max_iter: int = 5000,
) -> Dict[str, Any]:
    """
    Executes Baseline vs What-If Refinery LP models and generates decision-support analytics.
    Returns exact solved model dictionaries (base_model, whatif_model) for zero-trust auditing.
    """
    solver = SimplexSolver(tol=tolerance, max_iter=max_iter)

    # 1. Baseline Model
    base_model = build_refinery_model(
        throughput=base_throughput,
        max_sulfur_pct=base_sulfur,
        price_adjustments={},
        avail_adjustments={},
        carbon_penalty_per_bbl_sour=0.0,
        heavy_limit_bpd=25000.0,
    )
    res_base = solver.solve(
        c=base_model["c"],
        A_ub=base_model["A_ub"],
        b_ub=base_model["b_ub"],
        A_eq=base_model["A_eq"],
        b_eq=base_model["b_eq"],
        bounds=base_model["bounds"],
        maximize=False,
    )

    # 2. What-If Model
    whatif_model = build_refinery_model(
        throughput=what_if_throughput,
        max_sulfur_pct=what_if_sulfur,
        price_adjustments=price_adjustments,
        avail_adjustments=avail_adjustments,
        carbon_penalty_per_bbl_sour=carbon_penalty,
        heavy_limit_bpd=heavy_limit_bpd,
    )
    res_whatif = solver.solve(
        c=whatif_model["c"],
        A_ub=whatif_model["A_ub"],
        b_ub=whatif_model["b_ub"],
        A_eq=whatif_model["A_eq"],
        b_eq=whatif_model["b_eq"],
        bounds=whatif_model["bounds"],
        maximize=False,
    )

    crude_keys = base_model["crude_keys"]

    # 3. Allocation Comparison Table
    comparison_rows = []
    for i, key in enumerate(crude_keys):
        base_alloc = res_base.x[i] if res_base.success else 0.0
        whatif_alloc = res_whatif.x[i] if res_whatif.success else 0.0
        delta_alloc = whatif_alloc - base_alloc
        pct_change = (delta_alloc / base_alloc * 100.0) if base_alloc > 1e-4 else (100.0 if whatif_alloc > 1e-4 else 0.0)
        quota_val = avail_adjustments.get(key, BASE_CRUDES[key]["max_avail"])
        util_pct = (whatif_alloc / quota_val * 100.0) if quota_val > 0 else 0.0

        comparison_rows.append({
            "Crude Variety": key.replace("_", " "),
            "Crude Quality": BASE_CRUDES[key]["type"],
            "Sulfur %": f"{BASE_CRUDES[key]['sulfur']:.2f}%",
            "Price ($/bbl)": f"${whatif_model['c'][i]:.2f}",
            "Quota (bpd)": f"{quota_val:,.0f}",
            "Baseline (bpd)": f"{base_alloc:,.0f}",
            "Recommended (bpd)": f"{whatif_alloc:,.0f}",
            "Delta (bpd)": f"{'+' if delta_alloc >= 0 else ''}{delta_alloc:,.0f}",
            "Utilization": f"{util_pct:.0f}%",
            "Daily Cost ($)": f"${whatif_alloc * whatif_model['c'][i]:,.0f}",
            "base_val": base_alloc,
            "whatif_val": whatif_alloc,
        })

    df_comp = pd.DataFrame(comparison_rows)

    # 4. Computed Physical KPIs
    sulfur_coeffs = np.array([BASE_CRUDES[k]["sulfur"] for k in crude_keys])
    if res_whatif.success and what_if_throughput > 0:
        actual_blend_sulfur_pct = float(np.dot(sulfur_coeffs, res_whatif.x) / what_if_throughput)
        sulfur_headroom_pct_pts = what_if_sulfur - actual_blend_sulfur_pct
        sulfur_headroom_weighted_bbl = float(what_if_sulfur * what_if_throughput - np.dot(sulfur_coeffs, res_whatif.x))
        cost_whatif = res_whatif.fun
        cost_per_bbl = cost_whatif / what_if_throughput
    else:
        actual_blend_sulfur_pct = 0.0
        sulfur_headroom_pct_pts = 0.0
        sulfur_headroom_weighted_bbl = 0.0
        cost_whatif = 0.0
        cost_per_bbl = 0.0

    if res_base.success and base_throughput > 0:
        cost_base = res_base.fun
        cost_base_per_bbl = cost_base / base_throughput
    else:
        cost_base = 0.0
        cost_base_per_bbl = 0.0

    cost_delta = cost_whatif - cost_base
    cost_pct = (cost_delta / cost_base * 100.0) if cost_base > 0 else 0.0

    # 5. Identify Binding Constraints dynamically against exact model RHS
    binding_constraints = []
    if res_whatif.success:
        A_ub = whatif_model["A_ub"]
        b_ub = whatif_model["b_ub"]

        # Check sulfur ceiling (Row 0)
        sulfur_lhs = float(np.dot(A_ub[0], res_whatif.x))
        sulfur_rhs = float(b_ub[0])
        if abs(sulfur_lhs - sulfur_rhs) < 1.0 or abs(sulfur_headroom_pct_pts) < 1e-4:
            binding_constraints.append(
                f"🔴 **Blended Feed Sulfur Ceiling ({what_if_sulfur:.2f}%)**: BINDING at 100% capacity "
                f"(Blend Sulfur = {actual_blend_sulfur_pct:.3f}%)."
            )
        else:
            binding_constraints.append(
                f"🟢 **Blended Feed Sulfur Ceiling**: Non-binding (Headroom: +{sulfur_headroom_pct_pts:.3f}% pts / +{sulfur_headroom_weighted_bbl:,.1f} %·bbl)."
            )

        # Check Heavy crude / Coker limit (Row 1)
        heavy_rhs = float(b_ub[1])
        heavy_lhs = float(res_whatif.x[crude_keys.index("Basra_Heavy")])
        if abs(heavy_lhs - heavy_rhs) < 1.0:
            binding_constraints.append(
                f"🔴 **Basra Heavy Coker Capacity**: BINDING at max {heavy_rhs:,.0f} bpd limit."
            )
        else:
            binding_constraints.append(
                f"🟢 **Basra Heavy Coker Capacity**: Non-binding (Utilization: {heavy_lhs:,.0f} / {heavy_rhs:,.0f} bpd)."
            )

        # Check Supplier Limits (Rows 2..N+1)
        for i, k in enumerate(crude_keys):
            row_idx = 2 + i
            quota_rhs = float(b_ub[row_idx])
            quota_lhs = float(res_whatif.x[i])
            if abs(quota_lhs - quota_rhs) < 1.0 and quota_rhs > 0:
                binding_constraints.append(
                    f"🟡 **{k.replace('_', ' ')} Supplier Availability**: BINDING at max quota ({quota_rhs:,.0f} bpd)."
                )

    # 6. Concise Decision Intelligence Explanation (Max 3 clear bullets)
    explanations = []
    if res_whatif.success and res_base.success:
        # Bullet 1: Procurement Cost Impact
        if cost_delta > 0:
            explanations.append(f"**Procurement Cost Impact**: Daily acquisition cost increases by **${cost_delta:,.0f} (+{cost_pct:.1f}%)** to satisfy tighter quality or supply constraints.")
        elif cost_delta < 0:
            explanations.append(f"**Procurement Cost Impact**: Daily acquisition cost decreases by **${abs(cost_delta):,.0f} ({cost_pct:.1f}%)** from cheaper crude availability.")
        else:
            explanations.append("**Procurement Cost Impact**: Daily crude procurement expenditure remains unchanged at baseline optimal parity.")

        # Bullet 2: Quality / Sulfur Shift
        sweet_shift = (res_whatif.x[1] + res_whatif.x[2]) - (res_base.x[1] + res_base.x[2])
        if what_if_sulfur < base_sulfur:
            explanations.append(
                f"**Quality Regulation Driver**: Tightening the feed sulfur ceiling from {base_sulfur:.2f}% to {what_if_sulfur:.2f}% shifts **{abs(sweet_shift):,.0f} bpd** into Light Sweet grades (Brent/Bonny Light) to replace sour barrels."
            )
        elif carbon_penalty > 0:
            explanations.append(
                f"**Policy Incentive Response**: A hypothetical ${carbon_penalty:.2f}/bbl penalty on sour crude shifts economics toward low-sulfur crudes despite higher FOB spot prices."
            )
        elif any(abs(v) > 0.1 for v in price_adjustments.values()):
            shocked_crudes = [k.replace('_', ' ') for k, v in price_adjustments.items() if abs(v) > 0.1]
            explanations.append(f"**Spot Price Rebalancing**: Price adjustments on {', '.join(shocked_crudes)} prompted the LP solver to re-optimize margin trade-offs across supplier quotas.")
        else:
            explanations.append(f"**Feed Quality Baseline**: Blend sulfur is balanced at {actual_blend_sulfur_pct:.3f}% ({sulfur_headroom_pct_pts:+.3f}% pts headroom vs {what_if_sulfur:.2f}% limit).")

        # Bullet 3: Supply Bottleneck Driver
        binding_supplier_names = [k.replace('_', ' ') for i, k in enumerate(crude_keys) if abs(res_whatif.x[i] - whatif_model['b_ub'][2+i]) < 1.0]
        if binding_supplier_names:
            explanations.append(f"**Active Supply Bottlenecks**: Procurement of **{', '.join(binding_supplier_names)}** is constrained at 100% of supplier quota limits.")
        else:
            explanations.append("**Supply Buffer**: All supplier quotas have spare buffer capacity with no individual supply bottlenecks.")

    elif not res_whatif.success:
        total_avail = sum(avail_adjustments.get(k, BASE_CRUDES[k]["max_avail"]) for k in crude_keys)
        if what_if_throughput > total_avail:
            explanations.append(f"**Throughput Deficit**: Required throughput ({what_if_throughput:,.0f} bpd) exceeds total available crude supply ({total_avail:,.0f} bpd).")
        else:
            explanations.append(f"**Sulfur Infeasibility**: The requested {what_if_sulfur:.2f}% sulfur ceiling cannot be mathematically satisfied by the available crude varieties under current quotas.")
        explanations.append("**Actionable Next Step**: Relax the feed sulfur ceiling, increase supplier availability quotas, or reduce the daily distillation target.")

    return {
        "base_model": base_model,
        "whatif_model": whatif_model,
        "res_base": res_base,
        "res_whatif": res_whatif,
        "df_comp": df_comp,
        "binding_constraints": binding_constraints,
        "explanations": explanations[:3],  # Strictly capped at 3 concise bullets
        "cost_base": cost_base,
        "cost_whatif": cost_whatif,
        "cost_delta": cost_delta,
        "cost_pct": cost_pct,
        "cost_per_bbl": cost_per_bbl,
        "cost_base_per_bbl": cost_base_per_bbl,
        "actual_blend_sulfur_pct": actual_blend_sulfur_pct,
        "sulfur_headroom_pct_pts": sulfur_headroom_pct_pts,
        "sulfur_headroom_weighted_bbl": sulfur_headroom_weighted_bbl,
    }
