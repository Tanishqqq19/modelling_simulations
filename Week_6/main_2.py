import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------
# Physics: Hooke's 3-particle chain
# ---------------------------
def forces_chain(q, k=1.0):
    """
    Forces on the 3-particle chain.
    From Eq. (5) -> Eq. (6)/(8).
    Input q = [q1,q2,q3] are displacements from equilibrium.
    """
    F1 = k * (q[1] - q[0])              # Eq. (6): F1
    F2 = k * (q[0] - 2*q[1] + q[2])     # Eq. (6): F2
    F3 = k * (q[1] - q[2])              # Eq. (6): F3
    return np.array([F1, F2, F3])

def verlet_first(q0, v0, m, h, k):
    """
    Bootstrap step for Verlet (Eq. 16).
    q(-h) = q0 - h*v0 + 0.5*(h^2/m)*F(q0)
    Needed because Verlet (Eq. 14) requires q(t-h).
    """
    return q0 - h*v0 + 0.5*(h**2/m)*forces_chain(q0, k)

def verlet_next(q, q_prev, m, h, k):
    """
    Verlet update (Eq. 14).
    q(t+h) = 2q(t) - q(t-h) + (h^2/m)F(q(t))
    """
    return 2*q - q_prev + (h**2/m)*forces_chain(q, k)

def velocity_from_positions(q_next, q_prev, h):
    """
    Approximate velocity (Eq. 15).
    v(t) = (q(t+h) - q(t-h)) / (2h)
    """
    return (q_next - q_prev) / (2*h)

def total_energy(q, q_prev, m, h, k):
    """
    Total energy = K + U
    - Potential U from Eq. (5).
    - Kinetic K from velocities using Eq. (15).
    """
    v_cur = velocity_from_positions(q, q_prev, h)
    V = 0.5 * k * ((q[1]-q[0])**2 + (q[2]-q[1])**2)  # Eq. (5)
    K = 0.5 * m * np.sum(v_cur**2)
    return K + V

# ---------------------------
# Parameters & Initial Conditions
# ---------------------------
m = 1.0
k = 1.0
h = 0.002
T = 20.0
steps = int(T/h)

# Initial displacements (symmetric stretch) and velocities
q = np.array([-1.0, 0.0, 1.0], dtype=float)
v = np.zeros(3)

# Bootstrap q(-h) using Eq. (16)
q_prev = verlet_first(q, v, m, h, k)

# ---------------------------
# Visualization
# ---------------------------
fig, ax = plt.subplots(figsize=(7, 3))
ax.set_title("Live Verlet MD: 3-Particle Hooke's Chain")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1, 1)
ax.set_xlabel("Displacement (relative to equilibrium)")
ax.get_yaxis().set_visible(False)
ax.grid(True, alpha=0.3)

# Particles plotted as points
particles, = ax.plot(q, [0,0,0], 'o', markersize=10)

# Connectors (springs)
line12, = ax.plot([q[0], q[1]], [0, 0], lw=2)
line23, = ax.plot([q[1], q[2]], [0, 0], lw=2)

# Energy text
energy_text = ax.text(0.02, 0.9, f"Energy ≈ {total_energy(q, q_prev, m, h, k):.4f}", transform=ax.transAxes)

# ---------------------------
# Animation
# ---------------------------
def init():
    # Set initial state
    particles.set_data(q, [0,0,0])
    line12.set_data([q[0], q[1]], [0,0])
    line23.set_data([q[1], q[2]], [0,0])
    return particles, line12, line23, energy_text

def update(frame):
    global q, q_prev
    # Integrate with Verlet (Eq. 14)
    q_new = verlet_next(q, q_prev, m, h, k)
    
    # Update visuals
    particles.set_data(q_new, [0,0,0])
    line12.set_data([q_new[0], q_new[1]], [0,0])
    line23.set_data([q_new[1], q_new[2]], [0,0])
    
    # Update energy using Eq. (5)+(15)
    E = total_energy(q_new, q, m, h, k)
    energy_text.set_text(f"Energy ≈ {E:.4f}")
    
    # Shift state forward
    q_prev, q = q, q_new
    return particles, line12, line23, energy_text

anim = FuncAnimation(fig, update, init_func=init, frames=steps, interval=16, blit=True)
plt.show()
