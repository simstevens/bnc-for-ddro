import os
from instance_data import InstanceData


class MPS_AUX_Writer:
    def __init__(self, path: str, filename: str, gurobi_model, instance_data: InstanceData):
        self._path = path
        self._filename = filename
        self._gurobi_model = gurobi_model
        self._instance_data = instance_data

    def write(self):
        self._gurobi_model.write(os.path.join(self._path, self._filename + ".mps"))
        self._write_aux_file()

    def _write_aux_file(self):
        full_path = os.path.join(self._path, self._filename + ".aux")
        try:
            with open(full_path, "w") as aux_file:

                # Check essential attributes exist
                if self._instance_data.nr_ll_vars == -1:
                    raise ValueError("Missing nr_ll_vars.")
                if not self._instance_data.ll_vars or len(
                        self._instance_data.ll_vars) != self._instance_data.nr_ll_vars:
                    raise ValueError("Mismatch or missing ll_vars.")
                if self._instance_data.nr_ll_constrs == -1:
                    raise ValueError("Missing nr_ll_constrs.")
                if not self._instance_data.ll_constraints:
                    raise ValueError("Missing ll_constraints.")

                aux_file.write(f"@NUMVARS\n{self._instance_data.nr_ll_vars}\n")
                aux_file.write(f"@NUMCONSTRS\n{self._instance_data.nr_ll_constrs}\n")
                aux_file.write(f"@VARSBEGIN\n")
                objective_coeffs = dict(self._instance_data.ll_objective)
                for i in range(self._instance_data.nr_ll_vars):
                    var_name = self._instance_data.ll_vars[i]
                    coef = objective_coeffs.get(var_name, 0.0)
                    aux_file.write(f"{var_name}  {coef:.2f}\n")
                aux_file.write(f"@VARSEND\n")
                aux_file.write(f"@CONSTRSBEGIN\n")
                for constr in self._instance_data.ll_constraints:
                    aux_file.write(f"{constr}\n")
                aux_file.write(f"@CONSTRSEND\n")
                aux_file.write(f"@NAME\n{self._filename}\n")
                aux_file.write(f"@MPS\n{self._filename}.mps\n")
        except Exception as e:
            print(f"Failed to write AUX file to {full_path}: {e}")
            raise
