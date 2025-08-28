# oscillator_sideways.mp4
# Single harmonic oscillator integrated with Euler.
# Animation shows a mass moving horizontally with x(t) along the x-axis.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ----- physics -----
def hooke_force(x, k=1.0):
    return -k * x

def simulate_single_euler(T=30.0, h=0.005, k=1.0, m=1.0, x0=1.0, v0=0.0):
    steps = int(T / h)
    xs = np.zeros(steps + 1)
    vs = np.zeros(steps + 1)
    xs[0], vs[0] = x0, v0
    for t in range(steps):
        F = hooke_force(xs[t], k)
        xs[t+1] = xs[t] + h * vs[t]
        vs[t+1] = vs[t] + (h / m) * F
    ts = np.linspace(0, T, steps + 1)
    return xs, vs, ts

# ----- simulate -----
T = 30.0     # seconds of video
h = 0.005    # time step
k = 25.0
m = 1.0
x0 = 1.0     # start at +1
v0 = 0.0

xs, vs, ts = simulate_single_euler(T=T, h=h, k=k, m=m, x0=x0, v0=v0)

# ----- animation: mass sliding sideways -----
# Axis limits based on amplitude (slightly padded)
amp = max(1e-6, np.max(np.abs(xs)))
pad = 0.2 * amp
xmin, xmax = -amp - pad, amp + pad

fig, ax = plt.subplots(figsize=(7, 2.2))
ax.set_xlim(xmin, xmax)
ax.set_ylim(-1, 1)   # thin horizontal strip
ax.set_xlabel("x")
ax.set_yticks([])    # hide y ticks since motion is 1D along x
ax.set_title("Harmonic Oscillator (Euler) — Sideways Motion")

# A fixed wall at x = 0 and a line to the mass to look like a spring (optional)
wall = ax.axvline(0, linestyle="--", linewidth=1)
spring_line, = ax.plot([], [], lw=2, alpha=0.6)

# The mass (marker) moving at y=0
(mass_point,) = ax.plot([], [], marker="o", markersize=10)

def init():
    mass_point.set_data([], [])
    spring_line.set_data([], [])
    return mass_point, spring_line, wall

def update(i):
    x = xs[i]
    # mass at (x, 0)
    mass_point.set_data([x], [0.0])
    # draw spring from wall (0,0) to mass (x,0)
    spring_line.set_data([0.0, x], [0.0, 0.0])
    return mass_point, spring_line, wall

ani = FuncAnimation(fig, update, frames=range(0, len(ts), 5), init_func=init, blit=True)

# save MP4 (requires ffmpeg installed on your system)
writer = FFMpegWriter(fps=60)
ani.save("oscillator_sideways.mp4", writer=writer)
plt.close(fig)

print("Saved oscillator_sideways.mp4")
