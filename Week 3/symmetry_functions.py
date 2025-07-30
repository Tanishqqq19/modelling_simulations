import numpy as np
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
    n = len(positions)
    output=0
    eta = 0.5
    zeta = 1.0
    lambd = 1.0


    for j in range(n):

        # We need the bottom 5 lines. This means you're calculating the distance from an atom to itself
        # This can cause errors while calculating the cosine angle
        if j == i:
            continue
        for k in range(j + 1, n):
            if k == i:
                continue

            # This calculates individual distances
            r_ij = np.linalg.norm(positions[i] - positions[j])
            r_ik = np.linalg.norm(positions[i] - positions[k])
            r_jk = np.linalg.norm(positions[j] - positions[k])

            # Skip if beyond cutoff
            if r_ij > R_C or r_ik > R_C or r_jk > R_C:
                continue

            # Angle at atom i
            vec_ij = positions[j] - positions[i]
            vec_ik = positions[k] - positions[i]

            # This is the vector formula to find the angle
            cos_theta = np.dot(vec_ij, vec_ik) / (np.linalg.norm(vec_ij) * np.linalg.norm(vec_ik))


            # cutoff part of the problem
            fc = cutoff_function(np.array([r_ij]))[0] * cutoff_function(np.array([r_ik]))[0] * cutoff_function(np.array([r_jk]))[0]
            
            # angular part of the problem
            angular_term = (1 + lambd * cos_theta)**zeta

            # radial part of the problem
            radial_decay = np.exp(-eta * (r_ij**2 + r_ik**2 + r_jk**2))

            output += 2**(1 - zeta) * angular_term * radial_decay * fc

    return output
    
g2_values = [
    angular_symmetry_function(positions, i)
    for i in range(len(positions))
]



removed_g2_values=[]
for i in g2_values:
    removed_g2_values.append(float(i))


print("Angular Symmetry Outputs: ",removed_g2_values)