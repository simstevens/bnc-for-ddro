##################################################################
# This file is part of the code used for the computational study #
# in the paper                                                   #
#                                                                #
#  "Branch-and-Cut for Mixed-Integer Linear                      # 
#   Decision-Dependent Robust Optimization"                      #
#                                                                #
# by Henri Lefebvre, Martin Schmidt, Simon Stevens,              #
# and Johannes Thürauf (2026).                                   #
##################################################################

# Global imports
import argparse
import csv
from pathlib import Path

# Local imports
from bnc_problem_class import BnCProblem
from globals import Config, Tracker


def main():
    """Main function to run the Branch-and-Cut algorithm for the knapsack problem."""
    # Parse command line arguments
    args = parse_command_line_arguments()

    # Validate bobilib instance against the solvable list
    #check_bobilib_instance(args)

    # Determine cut types
    branchandbound, intersection_cuts, interdiction_cuts, no_good_cuts = determine_cut_types(args)

    # Create the config
    config = Config(
        instance_type=args.instance_type,
        instance_file=args.instance_file,
        lower_level=args.lower_level,
        projected=bool(args.projected),
        separation=args.separation,
        time_lim=args.time_lim,
        verbose_level=args.verbose_level,
        max_cuts=args.max_cuts,
        only_root_node=bool(args.only_root_node),
        cplex_cuts=bool(args.cplex_cuts),
        tolerance=args.tolerance,
        write_lps=bool(args.write_lps),
        interdiction_cuts=interdiction_cuts,
        intersection_cuts=intersection_cuts,
        nogood_cuts=no_good_cuts,
        branchandbound=branchandbound,
        minmax=args.minmax
    )
    # Create the tracker
    tracker = Tracker()
    
    # Print parameter information
    print_parameter_information(args, config)
    
    # Create and solve the BnC problem
    bnc_problem = BnCProblem(config, tracker)
    bnc_problem.solve()

def parse_command_line_arguments():
    """Parse command line arguments for the BnC algorithm."""
    parser = argparse.ArgumentParser(description="Run BnC for knapsack problem")
    parser.add_argument(
        "--instance_file",
        type=str,
        required=True,
        help="Path to the knapsack instance file",
    )
    parser.add_argument(
        "--cuts",
        type=str,
        required=True,
        choices=["branchandbound", "intersection", "interdiction", "nogood"],
        help="Type of cuts to use (required: branchandbound, intersection, interdiction, or nogood)",
    )
    parser.add_argument(
        "--verbose_level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Verbosity level (0: silent, 1: warnings, 2: time logging, 3: debug logging)",
    )
    parser.add_argument(
        "--lower_level",
        type=str,
        required=True,
        help="Lower level of the problem (required: general or interdiction)",
        choices=["general", "interdiction"],
    )
    parser.add_argument(
        "--projected",
        type=int,
        default=1,
        help="Whether to use the projected formulation (default: 1) or the unprojected formulation (0)",
        choices=[1, 0],
    )
    parser.add_argument(
        "--separation",
        type=str,
        default="integer",
        choices=["integer", "fractional"],
        help="Separation strategy for intersection cuts (default: integer)",
    )
    parser.add_argument(
        "--time_lim",
        type=float,
        default=60.0,
        help="Time limit for the solver in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--instance_type",
        type=str,
        required=True,
        choices=["knapsack", "bobilib"],
        help="Type of the instance file (required: knapsack or bobilib)",
    )
    parser.add_argument(
        "--max_cuts",
        type=int,
        default=20,
        help="Maximum number of cuts per node (default: 20)",
    )
    parser.add_argument(
        "--cplex_cuts",
        type=int,
        default=-1,
        choices=[-1, 0],
        help="Whether to use CPLEX cuts (0) or not (-1) (default: -1)",
    )
    parser.add_argument(
        "--only_root_node",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to separate cuts only at the root node (1) or at all nodes (0) (default: 0)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for optimality gap (default: 1e-6)",
    )
    parser.add_argument(
        "--write_lps",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to write LP files during solving (1) or not (0) (default: 0)",
    )

    args = parser.parse_args()
    return args

def determine_cut_types(args):
    """Determine which types of cuts to use based on the command line arguments."""
    branchandbound = False
    intersection_cuts = False
    interdiction_cuts = False
    no_good_cuts = False
    if args.cuts is None or args.cuts == "intersection":
        intersection_cuts = True
    elif args.cuts == "interdiction":
        interdiction_cuts = True
    elif args.cuts == "branchandbound":
        branchandbound = True
    elif args.cuts == "nogood":
        no_good_cuts = True
    else:
        raise ValueError(
            "Invalid cuts option. Choose from 'branchandbound', 'intersection', 'interdiction', or 'nogood'."
        )
    return branchandbound, intersection_cuts, interdiction_cuts, no_good_cuts

def check_bobilib_instance(args):
    """Ensure the requested bobilib instance is in solvable-instances.csv."""
    if args.instance_type != "bobilib":
        return

    instance_path = Path(args.instance_file)
    # Remove known extensions; handle optional .gz suffix first.
    name = instance_path.name
    if name.endswith(".mps.gz"):
        name = name[: -len(".mps.gz")]
    elif name.endswith(".mps"):
        name = name[: -len(".mps")]

    csv_path = Path(__file__).resolve().parent.parent / "instances" / "bobilib" / "solvable-instances.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected solvable-instances.csv at {csv_path}")

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        valid_instances = {row["Instance"] for row in reader if row.get("Instance")}

    if name not in valid_instances:
        raise ValueError(
            f"Instance '{name}' is not listed in solvable-instances.csv. "
            "Please choose one of the known solvable instances or update the list."
        )

def print_parameter_information(args, config):
    """Print the parameter information based on the command line arguments and configuration."""
    v_print = config._v_print
    print(f"\n%%%%%%%%%% Parameter information %%%%%%%%%%")
    print(f"Instance file: {args.instance_file}")
    print(f"Cuts used: {args.cuts}")
    if args.cuts == "intersection":
        print(f"Projected formulation: {bool(args.projected)}")
    print(f"Lower level: {args.lower_level}")
    print(f"Separation strategy: {args.separation}")
    print(f"Verbosity level: {args.verbose_level}")
    v_print(1, "INFO-Logging: ON")
    v_print(2, "TIME_Logging: ON")
    v_print(3, "ERROR-Logging: ON")
    print(f"Time limit: {args.time_lim} seconds")
    print(f"Maximum cuts per node: {args.max_cuts}")
    print(f"Only root node cuts: {bool(args.only_root_node)}")
    print(f"CPLEX cuts: {bool(args.cplex_cuts+1)}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Write LP files: {bool(args.write_lps)}")
    print(f"Min-max problem: {bool(args.minmax)}")
    print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")


if __name__ == "__main__":
    main()
