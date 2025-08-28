# chain_sideways.py
# Three-particle mass-spring chain integrated with Euler.
# Side-to-side animation (x = q_i(t)); saves MP4 only.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------- physics ----------
def chain_forces(q, k=1.0):
    # linear springs between neighbors
    F1 = k * (q[1] - q[0])
    F2 = k * (q[0] - 2.0*q[1] + q[2])
    F3 = k * (q[1] - q[2])
    return np.array([F1, F2, F3])

def simulate_chain_euler(T=20.0, h=0.01, k=1.0, m=1.0,
                         q0=np.array([-1.0, 0.0, 1.0]),
                         v0=np.zeros(3)):
    steps = int(T / h)
    qs = np.zeros((steps + 1, 3))
    vs = np.zeros((steps + 1, 3))
    qs[0], vs[0] = q0.copy(), v0.copy()
    for t in range(steps):
        F = chain_forces(qs[t], k)
        qs[t+1] = qs[t] + h * vs[t]
        vs[t+1] = vs[t] + (h / m) * F
    ts = np.linspace(0.0, T, steps + 1)
    return qs, vs, ts

# ---------- params you’ll tweak ----------
T = 20.0         # physics time (seconds)
h = 0.01         # time step
k = 4.0          # stiffer -> faster oscillations (omega ~ sqrt(k/m))
m = 1.0
q0 = np.array([-1.0, 0.0, 1.0])
v0 = np.zeros(3)

fps = 60         # video playback fps
frame_step = 5   # <-- increase this to shorten video / speed up playback (e.g., 5 => ~5x)

# ---------- simulate ----------
qs, vs, ts = simulate_chain_euler(T=T, h=h, k=k, m=m, q0=q0, v0=v0)

# axis limits from data
xmin = qs.min() - 0.2
xmax = qs.max() + 0.2

# y-rows to separate the three masses visually (still sideways motion)
y1, y2, y3 = 0.2, 0.0, -0.2

fig, ax = plt.subplots(figsize=(8, 2.6))
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.6, 0.6)
ax.set_xlabel("x")
ax.set_yticks([])
ax.set_title("Three-Particle Chain (Euler) — Sideways Motion")

# springs between neighbors (lines), and the 3 masses
spring12, = ax.plot([], [], lw=2, alpha=0.6)
spring23, = ax.plot([], [], lw=2, alpha=0.6)
m1, = ax.plot([], [], "o", markersize=9, label="q1")
m2, = ax.plot([], [], "o", markersize=9, label="q2")
m3, = ax.plot([], [], "o", markersize=9, label="q3")
ax.legend(loc="upper right")

def init():
    for ln in (spring12, spring23):
        ln.set_data([], [])
    for mass in (m1, m2, m3):
        mass.set_data([], [])
    return spring12, spring23, m1, m2, m3

def update(i):
    q1, q2, q3 = qs[i]
    # masses (x varies; y fixed rows)
    m1.set_data([q1], [y1])
    m2.set_data([q2], [y2])
    m3.set_data([q3], [y3])
    # springs (lines between neighbors)
    spring12.set_data([q1, q2], [y1, y2])
    spring23.set_data([q2, q3], [y2, y3])
    return spring12, spring23, m1, m2, m3

# skip frames to shorten to ~1 minute etc.
ani = FuncAnimation(
    fig,
    update,
    frames=range(0, len(ts), frame_step),  # <-- exact line to change speed/length
    init_func=init,
    blit=True,
)

writer = FFMpegWriter(fps=fps)  # MP4 only (needs ffmpeg)
ani.save("osciallator_sideways_eqn5.mp4", writer=writer)
plt.close(fig)

print("Saved chain_sideways.mp4")
