#!/usr/bin/python3

import gurobipy as gp
from statistics import median
from mps_aux_writer import MPS_AUX_Writer
from utils import gurobi_model_to_string, find_files_by_stem, base_stem
from instance_data import InstanceData
import numpy as np
from pathlib import Path

# Use class:
# mps_aux_reader = MPS_AUX_Parser(
#         "../instances/testinstances/generalExample.mps",
#         "../instances/testinstances/generalExample.aux",
#     )
# mps_aux_reader.read()
# print(mps_aux_reader)
# model = mps_aux_reader.get_gurobi_model()
# info = mps_aux_reader.get_instance_info()
class MPS_AUX_Parser:
    def __init__(self, mps_file_name: str, aux_file_name: str):
        self._mps_file_name = mps_file_name
        self._aux_file_name = aux_file_name


        # data to be filled by readers
        self._gurobi_model = any
        self._ll_data = None
        self._instance_data = InstanceData()

        self._instance_data.name = Path(self._aux_file_name).stem

    def read(self):
        self._read_mps_file()
        self._read_aux_file()
        self._set_nrs()
        self._sanity_checks()
        self._model_to_np_arrays()  # Optional: Convert constraints to NumPy arrays if needed

    @staticmethod
    def parsers_from_directory(root_dir: str, recurse: bool=False) -> dict[str, "MPS_AUX_Parser"]:
        """
        Returns a dict mapping instance stem to MPS_AUX_Parser for all valid MPS/AUX pairs in root_dir.
        If reverse=True, then all instances in subdirectories will be returned too.
        """
        root = Path(root_dir)

        if recurse:
            mps_files = {base_stem(f): f for f in root.rglob("*.mps*")}
            aux_files = {base_stem(f): f for f in root.rglob("*.aux")}
        else:
            mps_files = {base_stem(f): f for f in root.glob("*.mps*")}
            aux_files = {base_stem(f): f for f in root.glob("*.aux")}

        parsers = {}
        for stem in mps_files.keys() & aux_files.keys():
            parser = MPS_AUX_Parser(str(mps_files[stem]), str(aux_files[stem]))
            parser.read()
            parsers[stem] = parser

        return parsers



    def _read_mps_file(self):
        self._gurobi_model = gp.read(self._mps_file_name)

    def _read_aux_file(self):
        with open(self._aux_file_name, 'r') as file:
            lines = [line.strip() for line in file if line.strip() and not line.strip().startswith("#")]

        self._process_aux_lines(lines)
        self._normalize_vars_section()
        self._store_aux_data()

    def _process_aux_lines(self, lines):
        ll_data = {}
        current_section = None
        collecting = False

        for line in lines:
            if not line or line.startswith("#"):
                continue

            if line.startswith("@"):
                if line in ["@NUMVARS", "@NUMCONSTRS", "@NAME", "@MPS", "@LP"]:
                    current_section = line[1:]
                    collecting = False
                elif line.endswith("BEGIN"):
                    current_section = line[1:-5]  # strip @ and BEGIN
                    ll_data[current_section] = []
                    collecting = True
                elif line.endswith("END"):
                    collecting = False
                    current_section = None
                continue

            if current_section:
                if collecting:
                    ll_data[current_section].append(line)
                else:
                    # Single-line entries
                    ll_data[current_section] = line
        self._ll_data = ll_data

    def _normalize_vars_section(self):
        normalized_vars = []
        for entry in self._ll_data.get("VARS", []):
            parts = entry.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid VARS entry: {entry}")
            name = parts[0]
            try:
                value = float(parts[1])
            except ValueError:
                raise ValueError(f"Invalid numeric value in VARS entry: {entry}")
            normalized_vars.append((name, value))
        self._ll_data["VARS"] = normalized_vars

    def _store_aux_data(self):
        self._set_information_on_follower()
        self._set_information_on_leader()
        self._set_information_on_bounds()

    def _set_information_on_follower(self):
        self._instance_data.nr_ll_vars = int(self._ll_data['NUMVARS'])
        self._instance_data.nr_ul_vars = (len(self._gurobi_model.getVars()) - self._instance_data.nr_ll_vars)
        self._instance_data.ll_vars = [name for (name, value) in self._ll_data['VARS']]
        self._instance_data.ll_objective = self._ll_data['VARS']
        self._instance_data.ll_constraints = self._ll_data['CONSTRS']

    def _set_information_on_leader(self):
        self._instance_data.ul_vars = [var.VarName for var in self._gurobi_model.getVars() if
                                       var.VarName not in self._instance_data.ll_vars]
        self._instance_data.ul_constraints = [constr.ConstrName for constr in self._gurobi_model.getConstrs() if
                                              constr.ConstrName not in self._instance_data.ll_constraints]
        obj_expr = self._gurobi_model.getObjective()
        self._instance_data.ul_objective = [
            (obj_expr.getVar(i).VarName, obj_expr.getCoeff(i))
            for i in range(obj_expr.size())
        ]

    def _set_information_on_bounds(self):
        for var in self._gurobi_model.getVars():
            name = var.VarName
            lb = var.LB
            ub = var.UB

            if name in self._instance_data.ul_vars:
                self._instance_data.leader_upper_bounds.append((name, ub))
                self._instance_data.leader_lower_bounds.append((name, lb))
            elif name in self._instance_data.ll_vars:
                self._instance_data.follower_upper_bounds.append((name, ub))
                self._instance_data.follower_lower_bounds.append((name, lb))
            else:
                raise ValueError(f"Variable '{name}' not found in UL or LL variable lists.")

    def _set_nrs(self):
        self._instance_data.nr_ll_constrs = len(self._instance_data.ll_constraints)
        self._instance_data.nr_ul_constrs = (len(self._gurobi_model.getConstrs()) - self._instance_data.nr_ll_constrs)

    def get_instance_data(self):
        return self._instance_data

    def get_gurobi_model(self):
        return self._gurobi_model

    def __str__(self):
        return gurobi_model_to_string(self._gurobi_model) + "\n" + str(self._instance_data)

    def _sanity_checks(self):
        assert self._instance_data.nr_ul_constrs == len(
            self._instance_data.ul_constraints), "UL constraint count mismatch"
        assert self._instance_data.nr_ll_constrs == len(
            self._instance_data.ll_constraints), "LL constraint count mismatch"

    def write(self, path, filename):
        mps_aux_writer = MPS_AUX_Writer(path, filename, self._gurobi_model, self._instance_data)
        mps_aux_writer.write()

    def _constraint_to_np_array(self, constraint_name: str) -> np.ndarray:
        """Return the coefficients of the constraint as a NumPy array, ordered by model variables."""
        # find the constraint object
        constr = next(
            (c for c in self._gurobi_model.getConstrs() if c.ConstrName == constraint_name), None
        )
        if constr is None:
            raise ValueError(f"Constraint '{constraint_name}' not found.")


        # get the constraint's linear expression
        expr = self._gurobi_model.getRow(constr)

        return self._expression_to_np_array(expr)

    def _expression_to_np_array(self, expr: gp.LinExpr) -> np.ndarray:
        """Convert a Gurobi linear expression to a NumPy array of coefficients."""

        var_list = self._gurobi_model.getVars()
        var_to_index = {v:i for i, v in enumerate(var_list)}
        coeffs = np.zeros(len(var_list))

        # Map variable indices to coefficients
        for i in range(expr.size()):
            var = expr.getVar(i)
            idx = var_to_index[var]

            coeffs[idx] = expr.getCoeff(i)

        return coeffs

    def _model_to_np_arrays(self) -> None:

        """Convert the model's constraints to a dictionary of NumPy arrays."""

        # upper level constraints: A x + B y >= a
        A = np.zeros((self._instance_data.nr_ul_constrs, self._instance_data.nr_ul_vars)) # dim: m_u x n_x
        B = np.zeros((self._instance_data.nr_ul_constrs, self._instance_data.nr_ll_vars)) # dim: m_u x n_y
        a = np.zeros(self._instance_data.nr_ul_constrs) # dim: m_u

        # lower level constraints: C x + D y >= b
        C = np.zeros((self._instance_data.nr_ll_constrs, self._instance_data.nr_ul_vars)) # dim: m_l x n_x
        D = np.zeros((self._instance_data.nr_ll_constrs, self._instance_data.nr_ll_vars)) # dim: m_l x n_y
        b = np.zeros(self._instance_data.nr_ll_constrs) # dim: m_l

        # Temporaries for duplicated '=' rows (sign = -1). Will concatenate after main pass.
        extra_A = []
        extra_B = []
        extra_a = []
        extra_A_names = []

        extra_C = []
        extra_D = []
        extra_b = []
        extra_C_names = []

        self.var_indices = self._get_var_indices()
        
        # constraint coefficients and rhs values
        for constr in self._gurobi_model.getConstrs():
            constr_name = constr.ConstrName
            constr_coeffs= self._constraint_to_np_array(constr_name)

            sense = constr.Sense # '>' for \geq, '<' for \leq, '=' for ==

            # determine the sign of constraint (we want to convert <= to >=). For '=' add a mirrored row later.
            if sense == '<':
                sign = -1
                add_mirror = False
            elif sense == '>':
                sign = 1
                add_mirror = False
            else:  # '='
                sign = 1
                add_mirror = True

            if constr_name in self._instance_data.ll_constraints: # lower level constraint
                idx = self._instance_data.ll_constraints.index(constr_name) # row index
                C[idx, :] = sign * constr_coeffs[self.var_indices['ul_var_indices']]
                D[idx, :] = sign * constr_coeffs[self.var_indices['ll_var_indices']]
                b[idx] = sign * constr.RHS
                if add_mirror:
                    extra_C.append(-constr_coeffs[self.var_indices['ul_var_indices']])
                    extra_D.append(-constr_coeffs[self.var_indices['ll_var_indices']])
                    extra_b.append(-constr.RHS)
                    extra_C_names.append(constr_name)
            elif constr_name in self._instance_data.ul_constraints: # upper level constraint
                idx = self._instance_data.ul_constraints.index(constr_name)
                A[idx, :] = sign * constr_coeffs[self.var_indices['ul_var_indices']]
                B[idx, :] = sign * constr_coeffs[self.var_indices['ll_var_indices']]
                a[idx] = sign * constr.RHS
                if add_mirror:
                    extra_A.append(-constr_coeffs[self.var_indices['ul_var_indices']])
                    extra_B.append(-constr_coeffs[self.var_indices['ll_var_indices']])
                    extra_a.append(-constr.RHS)
                    extra_A_names.append(constr_name)
            else:
                raise ValueError(f"Constraint '{constr_name}' not found in either LL or UL constraints.")

        # Append mirrored '=' rows if any
        if extra_A:
            A = np.concatenate([A, np.vstack(extra_A)], axis=0)
            B = np.concatenate([B, np.vstack(extra_B)], axis=0)
            a = np.concatenate([a, np.array(extra_a)], axis=0)
            self._instance_data.ul_constraints.extend(extra_A_names)
            self._instance_data.nr_ul_constrs = len(self._instance_data.ul_constraints)
        if extra_C:
            C = np.concatenate([C, np.vstack(extra_C)], axis=0)
            D = np.concatenate([D, np.vstack(extra_D)], axis=0)
            b = np.concatenate([b, np.array(extra_b)], axis=0)
            self._instance_data.ll_constraints.extend(extra_C_names)
            self._instance_data.nr_ll_constrs = len(self._instance_data.ll_constraints)

        # upper level objective coefficients
        c_u = np.zeros(self._instance_data.nr_ul_vars) # upper level vars (dim: n_x)
        d_u = np.zeros(self._instance_data.nr_ll_vars) # lower level vars (dim: n_y)
        obj_sense = self._gurobi_model.ModelSense # 1 for minimization, -1 for maximization
        for var_name, coeff in self._instance_data.ul_objective:
            if var_name in self._instance_data.ll_vars: # lower level var
                idx = self._instance_data.ll_vars.index(var_name)
                d_u[idx] = obj_sense * coeff
            elif var_name in self._instance_data.ul_vars: # upper level var
                idx = self._instance_data.ul_vars.index(var_name)
                c_u[idx] = obj_sense * coeff
            else:
                raise ValueError(f"Variable '{var_name}' not found in either LL or UL variable lists.")

        # lower level objective coefficients
        # NOTE: assuming that lower level objective is always minimization
        d_l = np.zeros(self._instance_data.nr_ll_vars) # lower level vars (dim: n_y)
        for var_name, coeff in self._instance_data.ll_objective:
            if var_name in self._instance_data.ll_vars:
                idx = self._instance_data.ll_vars.index(var_name)
                d_l[idx] = coeff
            else:
                raise ValueError(f"Variable '{var_name}' not found in LL variable list.")

        # variable bounds


        # matrices and vectors to store bounds (optional, but helful when formulating KKT conditions of lower-level)
        ul_bd_mat = np.zeros((2*self._instance_data.nr_ul_vars, self._instance_data.nr_ul_vars))
        ul_bd_vec = np.zeros(2*self._instance_data.nr_ul_vars)
        ll_bd_mat = np.zeros((2*self._instance_data.nr_ll_vars, self._instance_data.nr_ll_vars))
        ll_bd_vec = np.zeros(2*self._instance_data.nr_ll_vars)

        # leader lower bounds
        leader_lbs = np.zeros(self._instance_data.nr_ul_vars)
        for var_name, lb in self._instance_data.leader_lower_bounds:
            if var_name in self._instance_data.ul_vars:
                idx = self._instance_data.ul_vars.index(var_name)
                leader_lbs[idx] = lb
                ul_bd_mat[idx, idx] = 1
                ul_bd_vec[idx] = lb
            else:
                raise ValueError(f"Variable '{var_name}' not found in UL variable list.")

        # leader upper bounds
        leader_ubs = np.inf * np.ones(self._instance_data.nr_ul_vars)
        for var_name, ub in self._instance_data.leader_upper_bounds:
            n_u = self._instance_data.nr_ul_vars # number of upper level variables
            if var_name in self._instance_data.ul_vars:
                idx = self._instance_data.ul_vars.index(var_name)
                leader_ubs[idx] = ub
                if ub != np.inf:
                    ul_bd_mat[idx + n_u, idx] = -1
                    ul_bd_vec[idx + n_u] = -ub
            else:
                raise ValueError(f"Variable '{var_name}' not found in UL variable list.")

        # follower lower bounds
        follower_lbs = np.zeros(self._instance_data.nr_ll_vars)
        for var_name, lb in self._instance_data.follower_lower_bounds:
            if var_name in self._instance_data.ll_vars:
                idx = self._instance_data.ll_vars.index(var_name)
                follower_lbs[idx] = lb
                ll_bd_mat[idx, idx] = 1
                ll_bd_vec[idx] = lb
            else:
                raise ValueError(f"Variable '{var_name}' not found in LL variable list.")

        # follower upper bounds
        follower_ubs = np.inf * np.ones(self._instance_data.nr_ll_vars)
        for var_name, ub in self._instance_data.follower_upper_bounds:
            n_l = self._instance_data.nr_ll_vars # number of upper level variables
            if var_name in self._instance_data.ll_vars:
                idx = self._instance_data.ll_vars.index(var_name)
                follower_ubs[idx] = ub
                if ub != np.inf:
                    ll_bd_mat[idx + n_l, idx] = -1
                    ll_bd_vec[idx + n_l] = -ub
            else:
                raise ValueError(f"Variable '{var_name}' not found in LL variable list.")


        self._instance_data.arrays = {
            'leader_ul_constr_mat': A,
            'follower_ul_constr_mat': B,
            'leader_ll_constr_mat': C,
            'follower_ll_constr_mat': D,
            'ul_rhs': a,
            'll_rhs': b,
            'leader_ul_obj': c_u,
            'follower_ul_obj': d_u,
            'follower_ll_obj': d_l,
            'leader_lbs': leader_lbs,
            'leader_ubs': leader_ubs,
            'follower_lbs': follower_lbs,
            'follower_ubs': follower_ubs,
            'leader_bd_mat': ul_bd_mat,
            'leader_bd_vec': ul_bd_vec,
            'follower_bd_mat': ll_bd_mat,
            'follower_bd_vec': ll_bd_vec
        }

    def _get_var_indices(self) -> dict[str, list]:
        ul_var_indices = [self._gurobi_model.getVarByName(var_name).index for var_name in self._instance_data.ul_vars]
        ll_var_indices = [self._gurobi_model.getVarByName(var_name).index for var_name in self._instance_data.ll_vars]
        return {
            'ul_var_indices': ul_var_indices,
            'll_var_indices': ll_var_indices
        }

    def get_np_arrays(self) -> dict[str, np.ndarray]:
        self.read()
        return self._instance_data.arrays

# for testing only
if __name__ == "__main__":
    mps_aux_reader = MPS_AUX_Parser(
        mps_file_name="./test-instance/general20-20-10-20-20-1.mps",
        aux_file_name="./test-instance/general20-20-10-20-20-1.aux",
    )

    mps_aux_reader.read()
    print(mps_aux_reader)
