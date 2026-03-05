import pprint
from collections import OrderedDict
import numpy as np

class InstanceData:
    def __init__(self):
        self.nr_ul_vars: int = -1
        self.nr_ll_vars : int = -1
        self.ul_vars: list = []
        self.ll_vars: list = []

        self.nr_ul_constrs: int = -1
        self.nr_ll_constrs: int = -1
        self.ul_constraints: list = []
        self.ll_constraints: list = []

        self.ul_objective: list = []
        self.ll_objective: list = []

        self.leader_lower_bounds: list = []
        self.leader_upper_bounds: list = []
        self.follower_lower_bounds: list = []
        self.follower_upper_bounds: list = []

        # dictionary to hold numpy arrays defining bilevel instance
        self.arrays: dict[str, np.ndarray] = {}
        self.name = None

    def __str__(self):
        return pprint.pformat(OrderedDict({
            "nr_ul_vars": self.nr_ul_vars,
            "nr_ll_vars": self.nr_ll_vars,
            "ul_vars": self.ul_vars,
            "ll_vars": self.ll_vars,
            "nr_ul_constrs": self.nr_ul_constrs,
            "nr_ll_constrs": self.nr_ll_constrs,
            "ul_constraints": self.ul_constraints,
            "ll_constraints": self.ll_constraints,
            "ul_objective": str(self.ul_objective),
            "ll_objective": str(self.ll_objective),
            "leader_lower_bounds": str(self.leader_lower_bounds),
            "leader_upper_bounds": str(self.leader_upper_bounds),
            "follower_lower_bounds": str(self.follower_lower_bounds),
            "follower_upper_bounds": str(self.follower_upper_bounds),
        }), indent=2)