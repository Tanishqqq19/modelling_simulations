import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Eq. 4 + Eq. 14 (Single particle + Verlet)
# ---------------------------
def hooke_force(x, k=1.0):
    # Eq. 3 from Eq. 4: F = -k*x
    return -k * x

def simulate_single_verlet(T=500, h=0.001, k=1.0, m=1.0, x0=1.0, v0=0.0):
    steps = int(T/h)
    xs = np.zeros(steps+1)
    vs = np.zeros(steps+1)
    xs[0], vs[0] = x0, v0

    # Eq. 16: bootstrap x(-h) using Taylor expansion
    x_prev = x0 - h*v0 + 0.5*(h**2/m)*hooke_force(x0, k)

    for t in range(steps):
        # Eq. 14: x(t+h) = 2x(t) - x(t-h) + (h^2/m)F(x(t))
        x_new = 2*xs[t] - x_prev + (h**2/m)*hooke_force(xs[t], k)
        # Eq. 15: v(t) ≈ (x(t+h) - x(t-h)) / (2h)  (store at index t)
        vs[t] = (x_new - x_prev) / (2*h)

        x_prev, xs[t+1] = xs[t], x_new

    # last velocity estimate at T
    vs[-1] = (xs[-1] - x_prev) / (2*h)
    return xs, vs, np.linspace(0, T, steps+1)

# Example run
xs, vs, ts = simulate_single_verlet()
plt.plot(ts, xs)
plt.title("Eq. 4 + 14: Single particle Verlet")
plt.xlabel("time")
plt.ylabel("x(t)")
plt.show()


# ---------------------------
# Eq. 5 + Eq. 14 (Three-particle chain + Verlet)
# ---------------------------
def chain_forces(q, k=1.0):
    # Eq. 6/8 from Eq. 5 (displacements from equilibrium)
    F1 = k * (q[1] - q[0])
    F2 = k * (q[0] - 2*q[1] + q[2])
    F3 = k * (q[1] - q[2])
    return np.array([F1, F2, F3])

def simulate_chain_verlet(T=500.0, h=0.001, k=1.0, m=1.0,
                          q0=np.array([-1.0, 0.0, 1.0]), v0=np.zeros(3)):
    steps = int(T/h)
    qs = np.zeros((steps+1, 3))
    vs = np.zeros((steps+1, 3))
    qs[0], vs[0] = q0.copy(), v0.copy()

    # Eq. 16: bootstrap q(-h)
    F0 = chain_forces(qs[0], k)
    q_prev = qs[0] - h*vs[0] + 0.5*(h**2/m)*F0

    for t in range(steps):
        # Eq. 14 for vector q
        q_new = 2*qs[t] - q_prev + (h**2/m)*chain_forces(qs[t], k)
        # Eq. 15 velocity (at index t)
        vs[t] = (q_new - q_prev) / (2*h)

        q_prev, qs[t+1] = qs[t], q_new

    # last velocity estimate at T
    vs[-1] = (qs[-1] - q_prev) / (2*h)
    return qs, vs, np.linspace(0, T, steps+1)

# Example run
qs, vs, ts = simulate_chain_verlet()
plt.plot(ts, qs[:,0], label="q1")
plt.plot(ts, qs[:,1], label="q2")
plt.plot(ts, qs[:,2], label="q3")
plt.title("Eq. 5 + 14: Three-particle Verlet")
plt.xlabel("time")
plt.ylabel("displacements")
plt.legend()
plt.show()
