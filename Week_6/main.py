# Simulation of single particle (Eq. 4) and three-particle chain (Eq. 5)
# Integrated with Euler (Eq. 11–12) and Verlet (Eq. 14–16).
# Produces trajectory and energy plots to compare stability/accuracy.

import numpy as np
import matplotlib.pyplot as plt

# ---------- Physics helpers ----------
def hooke_force(x, x_eq=0.0, k=1.0):
    # Eq. 4 -> force via Eq. 3
    return -k * (x - x_eq)

def three_particle_forces(q, k=1.0):
    # From Eq. 5 -> Eq. 6/8 (displacements from equilibrium)
    F1 = k * (q[1] - q[0])
    F2 = k * (q[0] - 2*q[1] + q[2])
    F3 = k * (q[1] - q[2])
    return np.array([F1, F2, F3])

def total_energy_single(x, v, k=1.0, m=1.0, x_eq=0.0):
    V = 0.5 * k * (x - x_eq)**2
    K = 0.5 * m * v**2
    return K + V

def total_energy_chain(q, v, k=1.0, m=1.0):
    # Potential from Eq. 5 using displacements (d eliminated): 0.5*k[(q2-q1)^2 + (q3-q2)^2]
    V = 0.5 * k * ((q[1]-q[0])**2 + (q[2]-q[1])**2)
    K = 0.5 * m * np.sum(v**2)
    return K + V

# ---------- Integrators ----------
def euler_step(q, v, m, h, force_func):
    F = force_func(q)
    q_new = q + h*v
    v_new = v + (h/m)*F
    return q_new, v_new

def verlet_init_q_prev(q0, v0, m, h, force_func):
    F0 = force_func(q0)
    return q0 - h*v0 + 0.5*(h**2/m)*F0  # Eq. 16

def verlet_step(q, q_prev, m, h, force_func):
    F = force_func(q)
    q_new = 2*q - q_prev + (h**2/m)*F  # Eq. 14
    return q_new

# ---------- Simulations ----------
def simulate_single(method="euler", T=5.0, h=1e-3, k=1.0, m=1.0, x0=1.0, v0=0.0):
    steps = int(T/h)
    xs = np.zeros(steps+1)
    vs = np.zeros(steps+1)
    Es = np.zeros(steps+1)
    xs[0], vs[0] = x0, v0
    Es[0] = total_energy_single(xs[0], vs[0], k=k, m=m)
    if method == "verlet":
        x_prev = verlet_init_q_prev(xs[0], vs[0], m, h, lambda x: hooke_force(x, 0.0, k))
    for t in range(steps):
        if method == "euler":
            x_new, v_new = euler_step(xs[t], vs[t], m, h, lambda x: hooke_force(x, 0.0, k))
        else:
            x_new = verlet_step(xs[t], x_prev, m, h, lambda x: hooke_force(x, 0.0, k))
            # velocity at integer step (Eq. 15)
            v_new = (x_new - x_prev) / (2*h)
            x_prev = xs[t]
        xs[t+1], vs[t+1] = x_new, v_new
        Es[t+1] = total_energy_single(xs[t+1], vs[t+1], k=k, m=m)
    return xs, vs, Es, np.linspace(0, T, steps+1)

def simulate_chain(method="euler", T=5.0, h=1e-3, k=1.0, m=1.0, q0=None, v0=None):
    steps = int(T/h)
    if q0 is None:
        q0 = np.array([-1.0, 0.0, 1.0])  # symmetric stretch conditions
    if v0 is None:
        v0 = np.zeros(3)
    qs = np.zeros((steps+1, 3))
    vs = np.zeros((steps+1, 3))
    Es = np.zeros(steps+1)
    qs[0], vs[0] = q0.copy(), v0.copy()
    Es[0] = total_energy_chain(qs[0], vs[0], k=k, m=m)
    if method == "verlet":
        q_prev = verlet_init_q_prev(qs[0], vs[0], m, h, lambda q: three_particle_forces(q, k))
    for t in range(steps):
        if method == "euler":
            q_new, v_new = euler_step(qs[t], vs[t], m, h, lambda q: three_particle_forces(q, k))
        else:
            q_new = verlet_step(qs[t], q_prev, m, h, lambda q: three_particle_forces(q, k))
            v_new = (q_new - q_prev) / (2*h)  # Eq. 15
            q_prev = qs[t]
        qs[t+1], vs[t+1] = q_new, v_new
        Es[t+1] = total_energy_chain(qs[t+1], vs[t+1], k=k, m=m)
    return qs, vs, Es, np.linspace(0, T, steps+1)

# ---------- Run simulations ----------
T = 10.0
h = 1e-3
k = 1.0
m = 1.0

# Single particle
xs_E, vs_E, Es_E, ts_E = simulate_single("euler", T=T, h=h, k=k, m=m, x0=1.0, v0=0.0)
xs_V, vs_V, Es_V, ts_V = simulate_single("verlet", T=T, h=h, k=k, m=m, x0=1.0, v0=0.0)

# Three-particle chain
qs_E, vqs_E, EQ_E, tq_E = simulate_chain("euler", T=T, h=h, k=k, m=m, q0=np.array([-1.0, 0.0, 1.0]), v0=np.zeros(3))
qs_V, vqs_V, EQ_V, tq_V = simulate_chain("verlet", T=T, h=h, k=k, m=m, q0=np.array([-1.0, 0.0, 1.0]), v0=np.zeros(3))

# ---------- Plots (one chart per figure, default colors) ----------
# Single particle trajectories
plt.figure(figsize=(8,5))
plt.plot(ts_E, xs_E, label="Euler")
plt.plot(ts_V, xs_V, label="Verlet")
plt.title("Single Particle: Displacement vs Time")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.legend()
plt.show()

# Single particle energy
plt.figure(figsize=(8,5))
plt.plot(ts_E, Es_E, label="Euler")
plt.plot(ts_V, Es_V, label="Verlet")
plt.title("Single Particle: Total Energy vs Time")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.legend()
plt.show()

# Chain: particle 1 trajectory
plt.figure(figsize=(8,5))
plt.plot(tq_E, qs_E[:,0], label="Euler")
plt.plot(tq_V, qs_V[:,0], label="Verlet")
plt.title("3-Particle Chain (q1): Displacement vs Time")
plt.xlabel("Time")
plt.ylabel("q1(t)")
plt.legend()
plt.show()

# Chain: total energy
plt.figure(figsize=(8,5))
plt.plot(tq_E, EQ_E, label="Euler")
plt.plot(tq_V, EQ_V, label="Verlet")
plt.title("3-Particle Chain: Total Energy vs Time")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.legend()
plt.show()

# Save outputs in case you need to attach them
plt.figure(figsize=(8,5))
plt.plot(tq_V, qs_V[:,0], label="Verlet q1")
plt.title("Saved: 3-Particle Chain (Verlet q1)")
plt.xlabel("Time"); plt.ylabel("q1(t)"); plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/chain_verlet_q1.png")

plt.figure(figsize=(8,5))
plt.plot(tq_E, EQ_E, label="Euler"); plt.plot(tq_V, EQ_V, label="Verlet")
plt.title("Saved: Chain Total Energy")
plt.xlabel("Time"); plt.ylabel("Energy"); plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/chain_energy_compare.png")

"/mnt/data/chain_verlet_q1.png", "/mnt/data/chain_energy_compare.png"
