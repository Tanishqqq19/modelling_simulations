import matplotlib.pyplot as plt
import matplotlib.animation as animation

# def eqn4(position, k):
#     r_eq=0.0
#     v_min=0.0

#     return 0.5 * k * (position - r_eq)**2 + v_min


# Euler integration does not work with energy potentials
# It needs force. This is seen in eq 3

# eqn3 is a derivative of eqn4
def eqn3(position, k):
    r_eq=0.0

    return -k*(position-r_eq)

# increment is the really small stepping value
def eqn11(position, vel, increment):
    return position + vel * increment

def eqn12(vel, force, mass, increment):
    acceleration= force/mass
    return vel+ increment * acceleration


# running the simulation
total_time_s=10.0
increment=0.001
k=25.0
m=1.0

x=1.0
v=0.0

steps= total_time_s/increment
positions= [1.0]
vel = [0.0]
time= [0.0]

for step in range(int(steps)):
    f = eqn3(x,k)

    updated_x= eqn11(x, v, increment)
    updated_v= eqn12(v, f, m, increment)

    x= updated_x
    v= updated_v

    positions.append(x)
    vel.append(v)
    time.append(time[-1]+ increment)


print(len(positions))
print(len(time))


plt.figure(figsize=(8,4))

plt.plot(time, positions, label="x(t)")
plt.xlabel("time")
plt.ylabel("x(t)")
plt.title("Single particle Euler")
plt.legend()
plt.grid(True)
plt.savefig("./Week_7/single_particle_euler.png")







# fig, ax = plt.subplots()
# (line,) = ax.plot([], [], "o")
# ax.set_xlim(min(positions) - 0.5, max(positions) + 0.5)
# ax.set_ylim(-0.2, 0.2)

# def init():
#     line.set_data([], [])
#     return (line,)

# def update(i):
#     line.set_data([positions[i]], [0.0])  # wrap in [] so it's a sequence
#     return (line,)

# ani = animation.FuncAnimation(
#     fig, update, init_func=init,
#     frames=len(positions), blit=True
# )

# # Save as mp4
# ani.save("single_particle_euler.mp4", writer=animation.FFMpegWriter(fps=60))
