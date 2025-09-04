import matplotlib.pyplot as plt

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


total_time=5.0
h=0.001
k=25.0
m=1.0
d=1.0

x = [i*d for i in range(6)]
v = [0.0, 0.0, 0.5, -0.5, 0, 0]

steps = int(total_time / h)
time = [0.0]
xs_hist = [x[:]]

for _ in range(steps):
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
plt.ylabel("position x_i(t)")
plt.title("6-Particle Chain (Euler) — Positions vs Time")
plt.legend(loc="upper right", ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("./Week_7/multi_particle_euler.png")
plt.show()

