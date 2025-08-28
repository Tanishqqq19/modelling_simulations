import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------- physics ----------
def chain_forces(q, k=1.0):
    """
    Linear nearest-neighbor springs:
      F1 = k*(q2 - q1)
      F2 = k*(q1 - 2*q2 + q3)
      F3 = k*(q2 - q3)
    """
    F1 = k * (q[1] - q[0])
    F2 = k * (q[0] - 2.0*q[1] + q[2])
    F3 = k * (q[1] - q[2])
    return np.array([F1, F2, F3])

def simulate_chain_verlet(T=20.0, h=0.005, k=1.0, m=1.0,
                          q0=np.array([-1.0, 0.0, 1.0]),
                          v0=np.zeros(3)):
    """
    Position-Verlet for the 3-particle chain:
      q_{n+1} = 2 q_n - q_{n-1} + a_n h^2,  where a_n = F(q_n)/m.
    Returns: qs (N x 3), vs (N x 3), ts (N)
    """
    steps = int(T / h)
    qs = np.zeros((steps + 1, 3))
    vs = np.zeros((steps + 1, 3))  # optional (central difference)
    ts = np.linspace(0.0, T, steps + 1)

    # initial conditions
    qs[0] = q0
    a0 = chain_forces(qs[0], k=k) / m
    # first step via Taylor expansion
    qs[1] = qs[0] + h * v0 + 0.5 * (h**2) * a0

    # main Verlet loop
    for n in range(1, steps):
        an = chain_forces(qs[n], k=k) / m
        qs[n + 1] = 2.0 * qs[n] - qs[n - 1] + (h**2) * an

    # central-difference velocities (optional)
    vs[1:-1] = (qs[2:] - qs[:-2]) / (2.0 * h)
    vs[0] = v0
    vs[-1] = (qs[-1] - qs[-2]) / h

    return qs, vs, ts

# ---------- parameters to tweak ----------
T = 20.0          # physics time (seconds)
h = 0.005         # timestep (keep small vs oscillation period)
k = 4.0           # stiffer spring => faster oscillations
m = 1.0
q0 = np.array([-1.0, 0.0, 1.0])  # initial positions
v0 = np.zeros(3)                 # initial velocities

fps = 60          # MP4 framerate
frame_step = 3    # <-- skip frames to shorten video / “fast-forward” (e.g., 3–8)

# ---------- simulate ----------
qs, vs, ts = simulate_chain_verlet(T=T, h=h, k=k, m=m, q0=q0, v0=v0)

# axis limits from data (pad a bit)
xmin = float(qs.min()) - 0.3
xmax = float(qs.max()) + 0.3

# y-rows to separate the three masses visually (still 1D motion in x)
y1, y2, y3 = 0.25, 0.0, -0.25

# ---------- animation ----------
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.6, 0.6)
ax.set_xlabel("x")
ax.set_yticks([])
ax.set_title("Three-Particle Chain — Verlet (Sideways)")

# springs (lines) and masses
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

ani = FuncAnimation(
    fig,
    update,
    frames=range(0, len(ts), frame_step),  # <-- exact line to change video length/speed
    init_func=init,
    blit=True,
)

writer = FFMpegWriter(fps=fps)  # MP4 only (requires ffmpeg installed)
ani.save("chain_verlet_sideways.mp4", writer=writer)
plt.close(fig)

print("Saved chain_verlet_sideways.mp4")
