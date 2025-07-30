import numpy as np
import pandas as pd
from itertools import combinations

import math


R_C = 6.0

def cutoff_function(r_ij):
    r_ij = np.asarray(r_ij)
    fc = np.where(
        r_ij <= R_C,
        0.5 * (np.cos(np.pi * r_ij / R_C) + 1),
        0.0
    )
    return fc


df = pd.read_csv('gdb9_100.csv')


positions = df[['x', 'y', 'z']].values
# This organises into these coordinates


r_ij_list = []
# This measures the distances of each atom and make sure they are under the cutoff
for i, j in combinations(range(len(positions)), 2):  
    dist = np.linalg.norm(positions[i] - positions[j])
    r_ij_list.append(dist)

# Convert to numpy array
r_ij_array = np.array(r_ij_list)

# Apply cutoff function
fc_values = cutoff_function(r_ij_array)

print("Cutoff values:", fc_values)



def radial_symmetry_function(positions, i, eta, Rs):
    r_ij = []
    for j in range(len(positions)):
        if j == i:
            continue
        dist = np.linalg.norm(positions[i] - positions[j])
        r_ij.append(dist)

    r_ij = np.array(r_ij)
    fc_values = cutoff_function(r_ij)
    equation = np.exp(-eta * (r_ij - Rs)**2) * fc_values
    return np.sum(equation)


features = [radial_symmetry_function(positions, i, eta=1.0, Rs=0.0)
            for i in range(len(positions))]
print("Radial_G1 per atom:", features)
