import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Eq. 4 + Eq. 11 (Single particle + Euler)
# ---------------------------
def hooke_force(x, k=1.0):
    # Eq. 3 from Eq. 4: F = -k*x
    return -k * x

def simulate_single_euler(T=500.0, h=0.00005, k=1.0, m=1.0, x0=1.0, v0=0.0):
    steps = int(T/h)
    xs = np.zeros(steps+1)
    vs = np.zeros(steps+1)
    xs[0], vs[0] = x0, v0
    for t in range(steps):
        F = hooke_force(xs[t], k)
        xs[t+1] = xs[t] + h * vs[t]                # Eq. 11
        vs[t+1] = vs[t] + (h/m) * F                # Eq. 12
    return xs, vs, np.linspace(0, T, steps+1)

# Example run
xs, vs, ts = simulate_single_euler()
plt.plot(ts, xs)
plt.title("Eq. 4 + 11: Single particle Euler")
plt.xlabel("time")
plt.ylabel("x(t)")
plt.show()


# ---------------------------
# Eq. 5 + Eq. 11 (Three-particle chain + Euler)
# ---------------------------
def chain_forces(q, k=1.0):
    # Eq. 6/8 from Eq. 5
    F1 = k * (q[1] - q[0])
    F2 = k * (q[0] - 2*q[1] + q[2])
    F3 = k * (q[1] - q[2])
    return np.array([F1, F2, F3])

def simulate_chain_euler(T=300, h=0.001, k=1.0, m=1.0,
                         q0=np.array([-1.0, 0.0, 1.0]), v0=np.zeros(3)):
    steps = int(T/h)
    qs = np.zeros((steps+1, 3))
    vs = np.zeros((steps+1, 3))
    qs[0], vs[0] = q0.copy(), v0.copy()
    for t in range(steps):
        F = chain_forces(qs[t], k)
        qs[t+1] = qs[t] + h * vs[t]                # Eq. 11
        vs[t+1] = vs[t] + (h/m) * F                # Eq. 12
    return qs, vs, np.linspace(0, T, steps+1)

# Example run
qs, vs, ts = simulate_chain_euler()
plt.plot(ts, qs[:,0], label="q1")
plt.plot(ts, qs[:,1], label="q2")
plt.plot(ts, qs[:,2], label="q3")
plt.title("Eq. 5 + 11: Three-particle Euler")
plt.xlabel("time")
plt.ylabel("displacements")
plt.legend()
plt.show()
