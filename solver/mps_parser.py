"""
IndSolve — Validated Continuous-LP MPS Subset Parser
Supports exact standard continuous LP sections: NAME, ROWS, COLUMNS, RHS, BOUNDS, ENDATA.
Strictly validates model dimensions and rejects unsupported constructs (INTORG/INTEND markers,
discrete bound types, quadratic/ranges/SOS sections) with clear, actionable student-friendly errors.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np


class MPSParseError(Exception):
    """Raised when an MPS file violates format syntax or contains unsupported model constructs."""
    pass


# Explicitly defined supported constructs for the Continuous-LP MPS Subset
SUPPORTED_SECTIONS = {"NAME", "ROWS", "COLUMNS", "RHS", "BOUNDS", "ENDATA"}
SUPPORTED_ROW_TYPES = {"N", "L", "G", "E"}
SUPPORTED_BOUND_TYPES = {"LO", "UP", "FX", "FR"}

UNSUPPORTED_SECTIONS = {
    "RANGES": "Double-sided range constraints (RANGES)",
    "SOS": "Special Ordered Sets (SOS)",
    "QUADOBJ": "Quadratic objective matrix (QUADOBJ)",
    "QCMATRIX": "Quadratic constraint matrix (QCMATRIX)",
    "CSECTION": "Convex quadratic section (CSECTION)",
    "INDICATORS": "Indicator constraints (INDICATORS)",
    "PWL": "Piecewise linear functions (PWL)",
    "DELAYED": "Delayed / Lazy constraints (DELAYED)",
    "USERCNSTR": "User-defined cutting planes (USERCNSTR)",
}

DISCRETE_BOUND_TYPES = {
    "BV": "Binary variable (0 or 1)",
    "LI": "Integer lower bound",
    "UI": "Integer upper bound",
    "SC": "Semi-continuous variable",
    "SI": "Semi-continuous integer variable",
}


def parse_mps_text(mps_text: str, max_vars: int = 250, max_rows: int = 250) -> Dict[str, Any]:
    """
    Parses a validated continuous-LP MPS file string into IndSolve canonical matrices.

    Supported Subset:
        - Continuous Linear Programs only (no integer markers or quadratic terms).
        - Sections: NAME, ROWS, COLUMNS, RHS, BOUNDS, ENDATA.
        - Row Types: N (Objective), L (<=), G (>=), E (==).
        - Bounds: LO (Lower Bound), UP (Upper Bound), FX (Fixed Variable), FR (Free Variable).

    Raises:
        MPSParseError: On unsupported sections, discrete/integer markers, malformed records,
                       or references to undeclared rows/variables.
    """
    lines = [line.rstrip() for line in mps_text.splitlines() if line.strip() and not line.strip().startswith("*")]
    if not lines:
        raise MPSParseError("The uploaded MPS file is empty or contains only comments.")

    current_section = None
    problem_name = "UNNAMED"
    obj_row_name = None
    row_types: Dict[str, str] = {}    # row_name -> 'N', 'L', 'G', 'E'
    row_order: List[str] = []         # Ordered constraint row names (excluding objective)

    col_dict: Dict[str, Dict[str, float]] = {}  # col_name -> {row_name: value}
    col_order: List[str] = []                   # Ordered structural variable names

    rhs_dict: Dict[str, float] = {}             # row_name -> value
    bounds_dict: Dict[str, Tuple[Optional[float], Optional[float]]] = {} # col_name -> (lb, ub)

    seen_sections = set()

    for line_num, line in enumerate(lines, start=1):
        # Section Header (Starts in Column 1 without leading whitespace)
        if not line.startswith(" ") and not line.startswith("\t"):
            parts = line.split()
            header = parts[0].upper()

            if header == "NAME":
                problem_name = parts[1] if len(parts) > 1 else "PROBLEM"
                current_section = "NAME"
                seen_sections.add("NAME")

            elif header in UNSUPPORTED_SECTIONS:
                desc = UNSUPPORTED_SECTIONS[header]
                raise MPSParseError(
                    f"Unsupported MPS Section '{header}' at line {line_num}: {desc}. "
                    f"IndSolve v0.2 supports the validated continuous-LP MPS subset only. "
                    f"Advanced '{header}' parsing is on the Phase 2 engineering roadmap."
                )

            elif header in SUPPORTED_SECTIONS:
                current_section = header
                seen_sections.add(header)

            else:
                raise MPSParseError(
                    f"Unknown or unsupported MPS Section header '{header}' at line {line_num}. "
                    f"Supported sections: {', '.join(sorted(SUPPORTED_SECTIONS))}."
                )
            continue

        parts = line.split()
        if not parts:
            continue

        # Check for integer markers in COLUMNS section
        line_upper = line.upper()
        if "'MARKER'" in line_upper or "INTORG" in line_upper or "INTEND" in line_upper:
            raise MPSParseError(
                f"Integer marker detected at line {line_num}: '{line.strip()}'. "
                f"IndSolve MPS parser supports the validated continuous-LP subset only. "
                f"For discrete MILP problems, please use IndSolve's native Branch-and-Bound solver."
            )

        # ---------------------------------------------------------------------
        # 1. ROWS SECTION
        # ---------------------------------------------------------------------
        if current_section == "ROWS":
            # Format: <TYPE> <ROW_NAME>
            if len(parts) < 2:
                raise MPSParseError(f"Malformed ROWS record at line {line_num}: '{line.strip()}'. Expected: <TYPE> <ROW_NAME>.")
            r_type = parts[0].upper()
            r_name = parts[1]

            if r_type not in SUPPORTED_ROW_TYPES:
                raise MPSParseError(
                    f"Invalid row type '{r_type}' for row '{r_name}' at line {line_num}. "
                    f"Supported row types: N (Objective), L (<=), G (>=), E (==)."
                )

            if r_type == "N":
                if obj_row_name is None:
                    obj_row_name = r_name
                # Note: Secondary N rows in MPS are ignored per standard
            else:
                if r_name in row_types:
                    raise MPSParseError(f"Duplicate row name '{r_name}' declared in ROWS section at line {line_num}.")
                row_types[r_name] = r_type
                row_order.append(r_name)

        # ---------------------------------------------------------------------
        # 2. COLUMNS SECTION
        # ---------------------------------------------------------------------
        elif current_section == "COLUMNS":
            # Format: <COL_NAME> <ROW_NAME_1> <VAL_1> [<ROW_NAME_2> <VAL_2>]
            if len(parts) < 3 or len(parts) % 2 == 0:
                raise MPSParseError(
                    f"Malformed COLUMNS record at line {line_num}: '{line.strip()}'. "
                    f"Expected: <COL_NAME> <ROW_1> <VAL_1> [<ROW_2> <VAL_2>]."
                )

            c_name = parts[0]
            if c_name not in col_dict:
                col_dict[c_name] = {}
                col_order.append(c_name)

            i = 1
            while i < len(parts):
                r_name = parts[i]
                try:
                    val = float(parts[i+1])
                except ValueError:
                    raise MPSParseError(f"Non-numeric matrix value '{parts[i+1]}' for variable '{c_name}', row '{r_name}' at line {line_num}.")

                if r_name not in row_types and r_name != obj_row_name:
                    raise MPSParseError(
                        f"COLUMNS record at line {line_num} references undefined row '{r_name}'. "
                        f"All constraint and objective rows must be declared in the ROWS section first."
                    )

                col_dict[c_name][r_name] = val
                i += 2

        # ---------------------------------------------------------------------
        # 3. RHS SECTION
        # ---------------------------------------------------------------------
        elif current_section == "RHS":
            # Format: [<RHS_NAME>] <ROW_NAME_1> <VAL_1> [<ROW_NAME_2> <VAL_2>]
            start_idx = 1 if (len(parts) % 2 == 1) else 0
            if (len(parts) - start_idx) < 2 or (len(parts) - start_idx) % 2 != 0:
                raise MPSParseError(f"Malformed RHS record at line {line_num}: '{line.strip()}'.")

            i = start_idx
            while i < len(parts):
                r_name = parts[i]
                try:
                    val = float(parts[i+1])
                except ValueError:
                    raise MPSParseError(f"Non-numeric RHS value '{parts[i+1]}' for row '{r_name}' at line {line_num}.")

                if r_name not in row_types and r_name != obj_row_name:
                    raise MPSParseError(f"RHS record at line {line_num} references undeclared row '{r_name}'.")

                rhs_dict[r_name] = val
                i += 2

        # ---------------------------------------------------------------------
        # 4. BOUNDS SECTION
        # ---------------------------------------------------------------------
        elif current_section == "BOUNDS":
            # Format: <BOUND_TYPE> <BOUND_NAME> <COL_NAME> [<VAL>]
            if len(parts) < 3:
                raise MPSParseError(f"Malformed BOUNDS record at line {line_num}: '{line.strip()}'. Expected: <TYPE> <BND_NAME> <COL_NAME> [<VAL>].")

            b_type = parts[0].upper()
            c_name = parts[2]

            if c_name not in col_dict:
                raise MPSParseError(
                    f"BOUNDS record at line {line_num} references undeclared variable '{c_name}'. "
                    f"All variables must appear in the COLUMNS section before receiving bounds."
                )

            if b_type in DISCRETE_BOUND_TYPES:
                desc = DISCRETE_BOUND_TYPES[b_type]
                raise MPSParseError(
                    f"Unsupported discrete/integer bound type '{b_type}' for variable '{c_name}' at line {line_num}: {desc}. "
                    f"This parser supports continuous-LP bounds (LO, UP, FX, FR) only."
                )

            if b_type not in SUPPORTED_BOUND_TYPES:
                raise MPSParseError(
                    f"Unsupported bound type '{b_type}' for variable '{c_name}' at line {line_num}. "
                    f"Supported continuous bound types: LO (Lower), UP (Upper), FX (Fixed), FR (Free)."
                )

            val = 0.0
            if b_type != "FR":
                if len(parts) < 4:
                    raise MPSParseError(f"Bound type '{b_type}' requires a numeric value at line {line_num}: '{line.strip()}'.")
                try:
                    val = float(parts[3])
                except ValueError:
                    raise MPSParseError(f"Non-numeric bound value '{parts[3]}' for variable '{c_name}' at line {line_num}.")

            current_lb, current_ub = bounds_dict.get(c_name, (0.0, None))
            if b_type == "UP":    # Upper bound: lb <= x <= val
                bounds_dict[c_name] = (current_lb, val)
            elif b_type == "LO":  # Lower bound: val <= x <= ub
                bounds_dict[c_name] = (val, current_ub)
            elif b_type == "FX":  # Fixed variable: x == val
                bounds_dict[c_name] = (val, val)
            elif b_type == "FR":  # Free variable: -inf < x < +inf
                bounds_dict[c_name] = (None, None)

    # -------------------------------------------------------------------------
    # VALIDATE MODEL COMPLETENESS & DIMENSIONS
    # -------------------------------------------------------------------------
    if "ROWS" not in seen_sections:
        raise MPSParseError("Missing mandatory ROWS section in MPS file.")
    if "COLUMNS" not in seen_sections:
        raise MPSParseError("Missing mandatory COLUMNS section in MPS file.")
    if obj_row_name is None:
        raise MPSParseError("No objective row (Type 'N') was declared in the ROWS section.")

    num_vars = len(col_order)
    num_rows = len(row_order)

    if num_vars == 0:
        raise MPSParseError("COLUMNS section contains zero variables.")

    if num_vars > max_vars or num_rows > max_rows:
        raise MPSParseError(
            f"MPS dimensions ({num_rows} rows × {num_vars} variables) exceed dense tableau safety limits "
            f"(max {max_rows} rows, {max_vars} variables). "
            f"IndSolve's dense core is optimized for small-to-medium models (< 250 rows/vars). "
            f"Sparse Revised-Simplex is scheduled for Phase 2."
        )

    # Build objective vector c
    c = np.zeros(num_vars, dtype=np.float64)
    for j, c_name in enumerate(col_order):
        c[j] = col_dict[c_name].get(obj_row_name, 0.0)

    # Build canonical constraint matrices
    A_ub_rows, b_ub_list = [], []
    A_ge_rows, b_ge_list = [], []
    A_eq_rows, b_eq_list = [], []

    for r_name in row_order:
        r_type = row_types[r_name]
        row_vec = np.zeros(num_vars, dtype=np.float64)
        for j, c_name in enumerate(col_order):
            if r_name in col_dict[c_name]:
                row_vec[j] = col_dict[c_name][r_name]

        rhs_val = rhs_dict.get(r_name, 0.0)

        if r_type == "L":
            A_ub_rows.append(row_vec)
            b_ub_list.append(rhs_val)
        elif r_type == "G":
            A_ge_rows.append(row_vec)
            b_ge_list.append(rhs_val)
        elif r_type == "E":
            A_eq_rows.append(row_vec)
            b_eq_list.append(rhs_val)

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_list) if b_ub_list else None
    A_ge = np.array(A_ge_rows) if A_ge_rows else None
    b_ge = np.array(b_ge_list) if b_ge_list else None
    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq_list) if b_eq_list else None

    # Variable bounds list
    bounds = [bounds_dict.get(c_name, (0.0, None)) for c_name in col_order]

    # Non-zeros count:
    # 1. Constraint Matrix Non-Zeros (Strictly A_ub, A_ge, A_eq)
    nnz_constraints = sum(
        sum(1 for r_name, val in row_map.items() if r_name in row_types and val != 0.0)
        for row_map in col_dict.values()
    )
    # 2. Total Non-Zeros (including objective row)
    total_nnz = sum(
        sum(1 for val in row_map.values() if val != 0.0)
        for row_map in col_dict.values()
    )

    total_matrix_elements = max(1, num_rows * num_vars)
    density = (nnz_constraints / total_matrix_elements) * 100.0

    return {
        "problem_name": problem_name,
        "obj_row_name": obj_row_name,
        "num_vars": num_vars,
        "num_rows": num_rows,
        "num_ub": len(b_ub_list),
        "num_ge": len(b_ge_list),
        "num_eq": len(b_eq_list),
        "nnz": nnz_constraints,
        "total_nnz": total_nnz,
        "density": density,
        "c": c,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_ge": A_ge,
        "b_ge": b_ge,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "bounds": bounds,
        "var_names": col_order,
        "row_names": row_order,
        "maximize": False,  # Standard MPS is minimization
    }
