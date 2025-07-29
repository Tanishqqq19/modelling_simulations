import math

R_C=6



def cutoff_function(r_ij):
    if r_ij <= R_C:
        fc = 0.5 * ( math.cos( ( math.pi * r_ij) / R_C ) + 1)

    else:
        fc=0

    return fc


r_ij=3
def radial_symmetry_function(n, rs):

    cutoff_value=cutoff_function(r_ij)

    equation=math.e**( -n )( ( r_ij - rs ) ** 2) - cutoff_value

    gaussian_function= sum(equation)