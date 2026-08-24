"""
IndSolve — Preloaded Industrial Problem Formulations & Benchmarks
Provides illustrative linear programming and mixed-integer case studies
using transparent, documented model parameters.
"""

from typing import Dict, Any
import numpy as np


def get_preloaded_problems() -> Dict[str, Dict[str, Any]]:
    """
    Returns pre-configured industrial optimization problems with honest,
    exact mathematical formulations matching their respective matrices.
    """
    return {
        "🛢️ Refinery 2D Visual Blend (MRPL Demo)": {
            "name": "MRPL 2-Crude Blending (Visual Walkthrough)",
            "description": "Blends Arabian Light and Brent Crude (illustrative parameters) to satisfy a minimum 1,000 bbl/day production target at minimum acquisition cost subject to supplier availability limits and a 1.0% feed sulfur ceiling.",
            "is_2d": True,
            "var_names": ["Arabian_Light (bbl)", "Brent_Crude (bbl)"],
            "maximize": False,
            "c": np.array([72.0, 78.0]),  # Illustrative price per barrel in USD
            # Inequalities A_ub @ x <= b_ub
            # 1. Arabian Light supply limit: x1 <= 700
            # 2. Brent supply limit: x2 <= 800
            # 3. Sulfur constraint: 1.77% x1 + 0.37% x2 <= 1.0% * (x1 + x2) => 0.77 x1 - 0.63 x2 <= 0
            "A_ub": np.array([
                [1.0, 0.0],
                [0.0, 1.0],
                [0.77, -0.63],
            ]),
            "b_ub": np.array([700.0, 800.0, 0.0]),
            # Inequalities A_ge @ x >= b_ge
            # 1. Total throughput >= 1000 bbl
            "A_ge": np.array([
                [1.0, 1.0],
            ]),
            "b_ge": np.array([1000.0]),
            "bounds": [(0.0, 700.0), (0.0, 800.0)],
            "integrality": [0, 0],
            "units": "USD",
            "context": "Illustrative 2-variable LP blending model demonstrating geometric feasible polytope navigation under feed sulfur limits."
        },

        "🏭 MRPL Multi-Crude Refinery Optimization (5 Crudes)": {
            "name": "MRPL Multi-Crude Slate Procurement Model",
            "description": "Optimizes procurement allocation across 5 crude varieties (illustrative model parameters) to fulfill a 100,000 bbl/day throughput target at minimum acquisition cost, subject to blend sulfur limits, coker heavy-feed capacity, light crude processing caps, and supplier availability.",
            "is_2d": False,
            "var_names": ["Arabian_Light", "Brent", "Bonny_Light", "Dubai_Sour", "Basra_Heavy"],
            "maximize": False,
            # Illustrative Prices ($/barrel): Arab Light: 72, Brent: 78, Bonny: 75, Dubai: 68, Basra: 58
            "c": np.array([72.0, 78.0, 75.0, 68.0, 58.0]),
            # A_ub <= b_ub:
            # Row 1: Sulfur limit (1.77% x1 + 0.37% x2 + 0.13% x3 + 2.00% x4 + 3.50% x5 <= 1.20% * 100000 = 120000)
            # Row 2: Heavy crude coker limit (Basra Heavy <= 25,000 bbl)
            # Row 3: Light crude processing cap (Brent + Bonny <= 60,000 bbl)
            # Rows 4..8: Individual supplier availability limits
            "A_ub": np.array([
                [1.77, 0.37, 0.13, 2.00, 3.50],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]),
            "b_ub": np.array([120000.0, 25000.0, 60000.0, 40000.0, 35000.0, 30000.0, 45000.0, 25000.0]),
            # A_eq == b_eq:
            # Total refinery throughput target = 100,000 barrels
            "A_eq": np.array([
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]),
            "b_eq": np.array([100000.0]),
            "A_ge": None,
            "b_ge": None,
            "bounds": [(0.0, 40000.0), (0.0, 35000.0), (0.0, 30000.0), (0.0, 45000.0), (0.0, 25000.0)],
            "integrality": [0, 0, 0, 0, 0],
            "units": "USD",
            "context": "Illustrative multi-crude slate LP inspired by refinery procurement planning workflows (e.g. RPMS/PIMS methodology). Prototype model using published illustrative parameters."
        },

        "⚡ Power Grid Economic Dispatch (NTPC/PowerGrid)": {
            "name": "Single-Period Electricity Merit-Order Dispatch",
            "description": "Determines single-period generation output across Coal, Solar, Hydro, and Gas units to satisfy a 1,200 MW demand target at lowest total cost subject to generation capacity bounds and a regional carbon emission cap.",
            "is_2d": False,
            "var_names": ["Coal_Unit_A (MW)", "Coal_Unit_B (MW)", "Solar_Farm (MW)", "Hydro_Plant (MW)", "Gas_Turbine (MW)"],
            "maximize": False,
            # Illustrative Marginal Cost in INR per MWh: Coal_A: 2800, Coal_B: 3100, Solar: 1900, Hydro: 1200, Gas: 4500
            "c": np.array([2800.0, 3100.0, 1900.0, 1200.0, 4500.0]),
            "A_ub": np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0],  # Coal A max 500MW
                [0.0, 1.0, 0.0, 0.0, 0.0],  # Coal B max 400MW
                [0.0, 0.0, 1.0, 0.0, 0.0],  # Solar max (daytime) 350MW
                [0.0, 0.0, 0.0, 1.0, 0.0],  # Hydro max reservoir 250MW
                [0.0, 0.0, 0.0, 0.0, 1.0],  # Gas peaker max 200MW
                # Carbon emission cap constraint: 0.9*CoalA + 0.85*CoalB + 0.4*Gas <= 700 tons CO2
                [0.9, 0.85, 0.0, 0.0, 0.4]
            ]),
            "b_ub": np.array([500.0, 400.0, 350.0, 250.0, 200.0, 700.0]),
            # Total demand equality constraint: sum(MW) == 1,200 MW
            "A_eq": np.array([
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ]),
            "b_eq": np.array([1200.0]),
            "A_ge": None,
            "b_ge": None,
            "bounds": [(100.0, 500.0), (100.0, 400.0), (0.0, 350.0), (50.0, 250.0), (0.0, 200.0)],
            "integrality": [0, 0, 0, 0, 0],
            "units": "INR",
            "context": "Merit-order economic dispatch model illustrating static multi-generator power balancing and emission cap compliance."
        },

        "📦 Multi-City Warehouse Facility Location & Logistics (Binary MILP)": {
            "name": "Regional Warehouse Hub Selection & Freight Allocation (Fixed-Charge MILP)",
            "description": "Determines optimal binary opening of 3 regional warehouse hubs (Mumbai, Delhi, Chennai) with fixed daily overheads and continuous freight routing to fulfill 650 tons of customer demand at minimum total cost.",
            "is_2d": False,
            "is_facility_milp": True,
            "var_names": [
                "Open_Mumbai_Hub (Binary y1)",
                "Open_Delhi_Hub (Binary y2)",
                "Open_Chennai_Hub (Binary y3)",
                "Ship_Mumbai_Tons (x1)",
                "Ship_Delhi_Tons (x2)",
                "Ship_Chennai_Tons (x3)"
            ],
            "facility_names": ["Mumbai Hub (West)", "Delhi Hub (North)", "Chennai Hub (South)"],
            "fixed_costs": [12000.0, 10000.0, 8000.0],
            "capacities": [500.0, 450.0, 350.0],
            "shipping_rates": [25.0, 32.0, 40.0],
            "demand_target": 650.0,
            "maximize": False,
            # Cost vector: [Fixed_Mumbai, Fixed_Delhi, Fixed_Chennai, Rate_Mumbai, Rate_Delhi, Rate_Chennai]
            "c": np.array([12000.0, 10000.0, 8000.0, 25.0, 32.0, 40.0]),
            # Capacity Linking constraints: x_i - Cap_i * y_i <= 0
            "A_ub": np.array([
                [-500.0,    0.0,    0.0, 1.0, 0.0, 0.0],
                [   0.0, -450.0,    0.0, 0.0, 1.0, 0.0],
                [   0.0,    0.0, -350.0, 0.0, 0.0, 1.0],
            ]),
            "b_ub": np.array([0.0, 0.0, 0.0]),
            # Demand satisfaction constraint: x1 + x2 + x3 >= 650 tons
            "A_ge": np.array([
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            ]),
            "b_ge": np.array([650.0]),
            "A_eq": None,
            "b_eq": None,
            "bounds": [
                (0.0, 1.0),    # y1 binary
                (0.0, 1.0),    # y2 binary
                (0.0, 1.0),    # y3 binary
                (0.0, 500.0),  # x1 continuous
                (0.0, 450.0),  # x2 continuous
                (0.0, 350.0),  # x3 continuous
            ],
            "integrality": [1, 1, 1, 0, 0, 0],  # y1, y2, y3 are strictly BINARY (0 or 1)
            "units": "INR",
            "context": "Classic Fixed-Charge Network Design (FCND) problem demonstrating mathematical necessity of Branch-and-Bound over LP relaxation."
        }
    }
