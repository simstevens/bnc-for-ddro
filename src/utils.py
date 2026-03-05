from pathlib import Path

def gurobi_model_to_string(gurobi_model):
    if not gurobi_model:
        return "No Gurobi model loaded."

    lines = []
    lines.append("=== Gurobi Model ===")

    # Objective
    lines.append("Objective:")
    obj_expr = gurobi_model.getObjective()
    terms = [
        f"{obj_expr.getCoeff(i)}*{obj_expr.getVar(i).VarName}"
        for i in range(obj_expr.size())
    ]
    lines.append(f"  {gurobi_model.ModelSense} : " + " + ".join(terms) + "\n")

    # Variables
    lines.append("Variables:")
    for var in gurobi_model.getVars():
        lines.append(f"  {var.VarName} ∈ [{var.LB}, {var.UB}]")
    lines.append("")

    # Constraints
    lines.append("Constraints:")
    for constr in gurobi_model.getConstrs():
        expr = gurobi_model.getRow(constr)
        terms = [
            f"{expr.getCoeff(i)}*{expr.getVar(i).VarName}"
            for i in range(expr.size())
        ]
        lhs = " + ".join(terms)
        sense = constr.Sense
        rhs = constr.RHS
        lines.append(f"  {constr.ConstrName}: {lhs} {sense} {rhs}")
    lines.append("")

    return "\n".join(lines)

def find_files_by_stem(root_dir: str, stem: str):
    """
    Searches directory and all subdirectories of <root_dir> for mps/aux file pairs
    with instance name <stem>.
    """
    root = Path(root_dir)
    return [str(f) for f in root.rglob('*') if f.stem == stem or f.stem == stem + '.mps']

def base_stem(path: Path) -> str:
    # Remove all suffixes (e.g., .mps.gz -> 'filename')
    stem = path.name
    for suf in path.suffixes:
        stem = stem[: -len(suf)]
    return stem