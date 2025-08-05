import numpy as np
np.set_printoptions(precision=16, suppress=False)

import pandas as pd

R_C = 6.0


# a very simple cutoff function coded out
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



def radial_symmetry_function(positions, i):
    eta=1.0
    Rs=0.0
    r_ij = [] 

    #finds the distance between two atoms
   
    for j in range(len(positions)):
        if j == i:
            continue
        dist = np.linalg.norm(positions[i] - positions[j])
        r_ij.append(dist)

    fc_values = cutoff_function(r_ij)

    print(f"Cutoff values for atom {i}:", fc_values)



    r_ij = np.array(r_ij)

    fc_values = cutoff_function(r_ij)



    # application of the formula
    equation = np.exp(-eta * (r_ij - Rs)**2) * fc_values
    return np.sum(equation)


features = [radial_symmetry_function(positions, i)
            for i in range(len(positions))]



removed_features=[]

for i in features:
    removed_features.append(float(i))

print("Radial Symmetry Outputs:", removed_features)


def angular_symmetry_function(positions, i):
    """
    Computes the angular symmetry function (G4) for atom i using all (j, k ≠ i)
    and applies the 2^(1 - zeta) scaling at the end.
    """
    n = len(positions)
    output = 0.0

    # Symmetry function parameters
    eta = 1.0
    zeta = 1.0
    lambd = 1.0
    R_C = 6.0

    def cutoff(r):
        return 0.5 * (np.cos(np.pi * r / R_C) + 1) if r <= R_C else 0.0

    for j in range(n):
        if j == i:
            continue
        for k in range(n):
            if k == i or k == j:
                continue

            # Distances
            r_ij = np.linalg.norm(positions[i] - positions[j])
            r_ik = np.linalg.norm(positions[i] - positions[k])
            r_jk = np.linalg.norm(positions[j] - positions[k])

            # Skip if beyond cutoff
            if r_ij > R_C or r_ik > R_C or r_jk > R_C:
                continue

            # Angle at atom i
            vec_ij = positions[j] - positions[i]
            vec_ik = positions[k] - positions[i]
            cos_theta = np.dot(vec_ij, vec_ik) / (np.linalg.norm(vec_ij) * np.linalg.norm(vec_ik))

            # G4 terms
            angular_term = (1 + lambd * cos_theta) ** zeta
            radial_decay = np.exp(-eta * (r_ij**2 + r_ik**2 + r_jk**2))
            fc = cutoff(r_ij) * cutoff(r_ik) * cutoff(r_jk)

            output += angular_term * radial_decay * fc

    output *= 2 ** (1 - zeta)  #  Apply scaling at the end
    return output
    
g2_values = [
    angular_symmetry_function(positions, i)
    for i in range(len(positions))
]



removed_g2_values=[]
for i in g2_values:
    removed_g2_values.append(float(i))


print("Angular Symmetry Outputs: ",removed_g2_values)