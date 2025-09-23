import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import pandas as pd
# os.makedirs("./Week_8", exist_ok=True)  # NEW (so the save path works)


def eqn5(x,k,d,V_min):
    """
    x is an array of positions of n particles
    k is hooke's constant
    d is the equilibriam separation of particles such that if distance between all
    particles is d, then the two particles are at rest and there will be no motion
    V_min is the minimum energy of the system

    assumes no friction in the system
    returns V - the energy of the system
    """
    V=0
    for i in range(len(x)-1):
        V+= 0.5 * k * (x[i+1] -x[i]-d)**2

    return V+V_min

# k = 1, d = 0
# x = [1, 2, 3, 4]
# f = [1, 0, 0, 0]

# i in [1, 2]

# f[1] = x[0] - 2*x[1] + x[2]
# f[2] = x[1] - 2*x[2] + x[3]
# f[3] = x[2] - x[3] + d

def eqn6(x,k,d):
    num_of_particles = len(x)
    f= [0.0] * num_of_particles
    f[0]= k* (x[1]-x[0]-d)

    for i in range (1, num_of_particles-1):
        f[i]= k* (x[i-1] - 2*x[i] + x[i+1])

    f[num_of_particles-1] = k * (x[num_of_particles-2] - x[num_of_particles-1] + d)

    return f


def eqn11(x,v,h):
    """
    Returns q which is a list of displacement of particle i from equilibrium i
    Conceptually, returns list of displacement of particles at t+h given displacements
    and potentials at t
    """
    mylist = []
    for i in range(len(x)):
        mylist.append(x[i] + h * v[i])
    return mylist

def eqn12(m,v,h,f):
    """
    v is velocity
    """
    new_v = []
    for i in range(len(v)):
        new_v.append(v[i]+h*(f[i] / m))
    return new_v

def simulate(x, v, k, d, V_min, steps):
    """
    x: initial position of particles
    v: initial velocity of particles
    k: hookes constant
    d: equilibrium distance between particles
    V_min: min energy of system

    returns
    xs_hist: position of all particles at all timesteps
        [positions_of_n_particles_at_time_0, positions_of_n_particles_at_time_1, ...]
    V_hist: energy of system at all timesteps
        [energy_at_time_0, energy_at_time_1, ...]
    """
    xs_hist = [x[:]]
    vs_hist = [v[:]]
    V_hist =  [eqn5(x[:], k, d, V_min)]

    for _ in range(steps):
        f = eqn6(x, k, d)
        x = eqn11(x, v, h)
        v = eqn12(m, v, h, f)
        V = eqn5(x, k, d, V_min)
        xs_hist.append(x[:])
        vs_hist.append(v[:])
        V_hist.append(V)
    
    return xs_hist, V_hist

total_time=10
h=0.001 # timestep diff
k=25.0
m=1.0 # mass
d=0.1
V_min = 0.0
steps = int(total_time / h)
filepath = "./Week_8/simulation_data_2_particles.csv"
num_simulations = 100
random.seed(0)

# x will be position of all particles at time t in simulation_id si
# V will be energy of system at time t in simulation_id si
data = {"simulation_id":[], "time":[], "x": [], "V": []}

for n in range(num_simulations):
    print("Simulating ", n)
    d = random.uniform(0.05, 0.20)
    x = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
    v = [random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)]

    # x = [-0.1, 0.1]
    # v = [0.01, -0.01]

    # returns all x positions and all V for timesteps
    xs_hist, V_hist = simulate(x, v, k, d, V_min, steps)


    for step in range(steps):
        data["simulation_id"].append(n)
        data["time"].append(h*step)
        data["x"].append(xs_hist[step])
        data["V"].append(V_hist[step])


with open(filepath, "w", newline="") as f:
    df = pd.DataFrame(data)
    df.to_csv(f, index=False)