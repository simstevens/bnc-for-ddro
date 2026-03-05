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
import random

def parse_knapsack(file_path):
    ''' Parses a .kp file and returns the knapsack parameters

    Parameters
    -------------
    file_name: string
        path to instance

    Returns
    --------------
    number_of_items: int
        number of items in the knapsack instance
    capacity: int
        capacity of the knapsack
    nom_weights: list
        list of nominal weights of the items
    weight_dev: list
        list of deviations of the weights of the items
    values: list
        list of the values of the items
    b: float
        constant parameter of uncertainty budget
    w: list
        list of weights in uncertainty budget
    f: list
        list of weights in knapsack uncertainty
    '''
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # First three lines
    seed = int(lines[0].strip())
    random.seed(seed)
    number_of_items = int(lines[1].strip())
    capacity = int(lines[2].strip())

    # Second block - extracting nom_weights, weight_dev, and values
    nom_weights = []
    weight_dev = []
    values = []
    hedge_cost = []
    idx = 4
    while lines[idx].strip() and not lines[idx].startswith(' '):
        nom, dev, val, hc = map(float, lines[idx].strip().split())
        nom_weights.append(nom)
        weight_dev.append(dev)
        values.append(val)
        hedge_cost.append(hc)
        idx += 1

    # Third block - extracting b, w, f
    idx += 2
    b_line = lines[idx].strip()
    b = [int(val.strip()) for val in b_line.strip('[]').split(',')]
    idx += 2
    
    # Parse w matrix (first matrix)
    w_line = lines[idx].strip()
    w_line = w_line.strip('[]')
    w_rows = w_line.split('] [')
    
    w = []
    for row_str in w_rows:
        row_str = row_str.strip('[]')
        values_in_row = row_str.split(',')
        row = [int(val.strip()) for val in values_in_row]
        w.append(row)
    
    idx += 2
    
    # Parse f matrix (second matrix)
    f_line = lines[idx].strip()
    f_line = f_line.strip('[]')
    f_rows = f_line.split('] [')
    
    f = []
    for row_str in f_rows:
        row_str = row_str.strip('[]')
        values_in_row = row_str.split(',')
        row = [int(val.strip()) for val in values_in_row]
        f.append(row)
   
    return values, hedge_cost, capacity, nom_weights, weight_dev, f, b, w, "knapsack", file_path