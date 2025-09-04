import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# def eqn5(x,k,d,v_min):
#     v=0
#     for i in range(len(x)-1):
#         v+= 0.5 * k * (x[i+1] -x[i]-d)**2

#     return v+v_min

def eqn6(x,k,d):
    num_of_particles = len(x)
    f= [0.0] * num_of_particles
    f[0]= k* (x[1]-x[0]-d)

    for i in range (1, num_of_particles-1):
        f[i]= k* (x[i-1] - 2*x[i] + x[i+1])

    f[num_of_particles-1] = k * (x[num_of_particles-2] - x[num_of_particles-1] + d)

    return f


def eqn11(x,v,h):
    mylist = []
    for i in range(len(x)):
        mylist.append(x[i] + h * v[i])
    return mylist

def eqn12(m,v,h,f):
    new_v = []
    for i in range(len(v)):
        new_v.append(v[i]+h*(f[i] / m))
    return new_v




total_time=0.5
h=0.001
k=25.0
m=1.0
d=1.0

x = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
v = [-0.2, -0.1, -0.1, 0.1, 0.2, 0.2]

steps = int(total_time / h)
time = [0.0]
xs_hist = [x[:]]

for i in range(steps):
    f = eqn6(x, k, d)
    x = eqn11(x, v, h)
    v = eqn12(m, v, h, f)

    xs_hist.append(x[:])
    time.append(time[-1] + h)

# print(time)
# print(xs_hist)



num_particles = 6
trajectories = []
for i in range(num_particles):
    series = []
    for snapshot in xs_hist:
        series.append(snapshot[i])
    trajectories.append(series)


plt.figure(figsize=(9,4.5))
for i in range(num_particles):
    plt.plot(time, trajectories[i], label=f"x{i+1}(t)")
plt.xlabel("time (s)")
plt.ylabel("position x (t)")
plt.xlim(0, 1)   # tight around the cluster

plt.title("6-Particle Chain")
plt.legend(loc="upper right", ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("./Week_7/multi_particle_euler.png")
plt.show()



fig, ax = plt.subplots()
scat = ax.scatter([], [], s=80)

ax.set_xlim(-1.5, 2.0)   # zoomed but fixed
ax.set_ylim(-0.2, 0.2)   # thin band
ax.set_yticks([])

def init():
    scat.set_offsets(np.empty((0, 2)))
    return scat,

def update(frame):
    xs = xs_hist[frame]
    ys = [0] * len(xs)
    scat.set_offsets(np.c_[xs, ys])
    return scat,

ani = animation.FuncAnimation(fig, update, frames=len(xs_hist),
                              init_func=init, blit=True, interval=20)

ani.save("./Week_7/harmonic_chain.mp4", writer="ffmpeg", fps=30)
