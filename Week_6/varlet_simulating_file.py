# verlet_sideways.py
# Single harmonic oscillator (Hooke) integrated with VERLET (not Euler).
# Sideways animation: the mass moves along x(t) on the x-axis. MP4 only.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ----- physics -----
def accel(x, k=1.0, m=1.0):
    # a = F/m = -(k/m) * x   (Eq. 4 in your notes: Hooke's law)
    return -(k / m) * x

def simulate_single_verlet(T=20.0, h=0.005, k=1.0, m=1.0, x0=1.0, v0=0.0):
    """
    Position-Verlet for m x'' + k x = 0
    Returns: xs (N), vs (N), ts (N)
    """
    steps = int(T / h)
    xs = np.zeros(steps + 1)
    vs = np.zeros(steps + 1)  # optional (computed by central difference)
    ts = np.linspace(0.0, T, steps + 1)

    # initial conditions
    xs[0] = x0
    a0 = accel(xs[0], k=k, m=m)
    # first step (kick-start Verlet with Taylor expansion)
    xs[1] = xs[0] + h * v0 + 0.5 * (h ** 2) * a0

    # main loop
    for n in range(1, steps):
        an = accel(xs[n], k=k, m=m)
        xs[n + 1] = 2.0 * xs[n] - xs[n - 1] + (h ** 2) * an

    # central-difference velocity (optional)
    vs[1:-1] = (xs[2:] - xs[:-2]) / (2.0 * h)
    vs[0] = v0
    vs[-1] = (xs[-1] - xs[-2]) / h

    return xs, vs, ts

# ----- params you’ll tweak -----
T = 20.0       # physics time (seconds)
h = 0.005      # timestep (keep small enough for stability; ω = sqrt(k/m))
k = 4.0        # stiffer spring => faster oscillations
m = 1.0
x0 = 1.0
v0 = 0.0

fps = 60       # MP4 playback framerate
frame_step = 3 # <-- skip frames to shorten video / speed up playback (e.g., 3~5)

# ----- simulate -----
xs, vs, ts = simulate_single_verlet(T=T, h=h, k=k, m=m, x0=x0, v0=v0)

# axis limits based on amplitude
amp = max(1e-6, np.max(np.abs(xs)))
pad = 0.2 * amp
xmin, xmax = -amp - pad, amp + pad

# ----- animation -----
fig, ax = plt.subplots(figsize=(7, 2.2))
ax.set_xlim(xmin, xmax)
ax.set_ylim(-1, 1)
ax.set_xlabel("x")
ax.set_yticks([])
ax.set_title("Harmonic Oscillator — Verlet (Sideways)")

# optional wall at x=0 and a spring line
wall = ax.axvline(0, linestyle="--", linewidth=1)
spring_line, = ax.plot([], [], lw=2, alpha=0.6)
mass_point, = ax.plot([], [], "o", markersize=10)

def init():
    mass_point.set_data([], [])
    spring_line.set_data([], [])
    return mass_point, spring_line, wall

def update(i):
    x = xs[i]
    mass_point.set_data([x], [0.0])
    spring_line.set_data([0.0, x], [0.0, 0.0])
    return mass_point, spring_line, wall

ani = FuncAnimation(
    fig,
    update,
    frames=range(0, len(ts), frame_step),  # <-- exact line to make the video shorter/faster
    init_func=init,
    blit=True,
)

writer = FFMpegWriter(fps=fps)  # MP4 only (needs ffmpeg installed)
ani.save("verlet_sideways.mp4", writer=writer)
plt.close(fig)

print("Saved verlet_sideways.mp4")
