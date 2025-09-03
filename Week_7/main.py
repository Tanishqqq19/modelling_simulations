import matplotlib.pyplot as plt
import pandas as pd

def eqn4(position, k):
    r_eq=0.0
    v_min=0.0

    return 0.5 * k * (position - r_eq)**2 + v_min

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



plt.figure(figsize=(8,4))
plt.plot(time, positions, label="x(t)")
plt.xlabel("time")
plt.ylabel("x(t)")
plt.title("Eq. 4 + 11: Single particle Euler")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
