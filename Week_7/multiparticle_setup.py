def eqn5(x,k,d,v_min):
    v=0
    for i in range(len(x)-1):
        v+= 0.5 * k * (x[i+1] -x[i]-d)**2

    return v+v_min

def eqn6(x,k,d):
    num_of_particles = len(x)
    f= [0.0] * num_of_particles

    f[0]= k* (x[1]-x[0]-d)

    for i in range (1, num_of_particles-1):
        f[i]= k* (x[i-1] - 2*x[i] + x[i+1])

    f[num_of_particles-1] = k * (x[num_of_particles-2] - x[num_of_particles-1] + d)