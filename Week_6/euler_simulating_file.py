import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# -----------------------------
# PHYSICS: force law (Eq. 4)

def hooke_force(x, k=1.0):
    return -k * x   # <-- Eq. 4

def simulate_single_euler(T=30.0, h=0.005, k=1.0, m=1.0, x0=1.0, v0=0.0):
    steps = int(T / h)                 # total time steps in the physics simulation
    xs = np.zeros(steps + 1)           # positions over time (simulation result)
    vs = np.zeros(steps + 1)           # velocities over time (simulation result)
    xs[0], vs[0] = x0, v0              # initial conditions

    for t in range(steps):
        F = hooke_force(xs[t], k)      # uses Eq. 4 to compute force
        xs[t+1] = xs[t] + h * vs[t]    # Eq. 11 (Euler position update)
        vs[t+1] = vs[t] + (h / m) * F  # Eq. 12 (Euler velocity update)

    ts = np.linspace(0, T, steps + 1)  # time grid corresponding to the arrays
    return xs, vs, ts

# T,h,k,m,x0,v0 set the physical system

T = 30.0    # total simulated physical time (seconds). How long the motion runs.
h = 0.0025   # time step (seconds). The increment between simulation updates
k = 25.0    # sping constant
m = 1.0     # mass
x0 = 1.0    # initial position
v0 = 0.0    # initial velocity

# xs,vs,ts are the physics outputs.

xs, vs, ts = simulate_single_euler(T=T, h=h, k=k, m=m, x0=x0, v0=v0)


# choose axis limits based on amplitude.

amp = max(1e-6, np.max(np.abs(xs)))
pad = 0.2 * amp
xmin, xmax = -amp - pad, amp + pad

fig, ax = plt.subplots(figsize=(7, 2.2))
ax.set_xlim(xmin, xmax)
ax.set_ylim(-1, 1)
ax.set_xlabel("x")
ax.set_yticks([])
ax.set_title("Harmonic Oscillator (Euler) — Sideways Motion")

# DRAWABLE OBJECTS (purely visualization)
# "wall" and "spring_line" are graphics,
# mass_point shows x(t) along x-axis.

wall = ax.axvline(0, linestyle="--", linewidth=1)
spring_line, = ax.plot([], [], lw=2, alpha=0.6)
(mass_point,) = ax.plot([], [], marker="o", markersize=10)

# ANIMATION BOILERPLATE (graphics only)
# init(): empty scene
# update(i): uses simulated xs[i]
# to place the mass and spring.

def init():
    mass_point.set_data([], [])
    spring_line.set_data([], [])
    return mass_point, spring_line, wall

def update(i):
    x = xs[i]                          
    mass_point.set_data([x], [0.0])     
    spring_line.set_data([0.0, x], [0.0, 0.0])
    return mass_point, spring_line, wall


# frames=range(0, len(ts), 5): skip 4 of every 5 physics
# steps when *saving* → shorter/faster video.

ani = FuncAnimation(fig, update, frames=range(0, len(ts), 5), init_func=init, blit=True)

writer = FFMpegWriter(fps=60)    
ani.save("oscillator_sideways.mp4", writer=writer)
plt.close(fig)

print("Saved oscillator_sideways.mp4")
