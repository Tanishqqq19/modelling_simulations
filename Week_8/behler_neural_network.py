import random
import math

Rc=6.0        # cutoff 
eta_G1=0.2
Rs_G1=2.0
eta_G2=0.01
zeta_G2=1.0
lambda_G2=1.0


def cutoff_cos(r, Rc):
    if r > Rc:
        return 0.0
    return 0.5*(math.cos(math.pi * r / Rc) + 1.0)





def distance(a, b):
    return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))

def dot(u, v):
    return sum(ui*vi for ui,vi in zip(u,v))

def norm(u):
    return math.sqrt(dot(u,u))




def G1_single(R, Rc, eta, Rs):
    n = len(R)
    out = []
    for i in range(n):
        total = 0.0
        for j in range(n):
            if j == i: 
                continue

            rij= math.sqrt(sum((ai - bi)**2 for ai, bi in zip(R[i], R[j])))
            fc = cutoff_cos(rij, Rc)
            total += math.exp(-eta * (rij - Rs)**2) * fc

        out.append(total)
    return out  # list length N


def G2_single(R, Rc, eta, zeta, lam):
    N = len(R)
    out = []
    for i in range(N):
        total = 0.0
        for j in range(N):
            if j == i: 
                continue
            for k in range(N):
                if k == i or k == j: 
                    continue
                rij = distance(R[i], R[j])
                rik = distance(R[i], R[k])
                rjk = distance(R[j], R[k])
                if rij > Rc or rik > Rc or rjk > Rc:
                    continue

                vij= [ai-bi for ai, bi in zip(R[i], R[j])]
                vik= [ai-bi for ai, bi in zip(R[i], R[k])]

                cos_theta = dot(vij, vik)/(norm(vij)*norm(vik)+1e-12)

                ang = (1.0 + lam * cos_theta)**zeta
                rad = math.exp(-eta * (rij**2 + rik**2 + rjk**2))
                fc  = cutoff_cos(rij,Rc)*cutoff_cos(rik, Rc)*cutoff_cos(rjk,Rc)

                total += (2.0**(1.0 - zeta)) * ang*rad*fc
        out.append(total)
    return out

def symmetry_functions(R):
    g1 = G1_single(R, Rc, eta_G1, Rs_G1)
    g2 = G2_single(R, Rc, eta_G2, zeta_G2, lambda_G2)
    return [[g1[i], g2[i]] for i in range(len(R))]




class AtomicNetwork:
    def __init__(self, num_inputs, num_hidden):
        self.weights_input_hidden = [
            [random.uniform(-0.1, 0.1) for _ in range(num_inputs)]
            for _ in range(num_hidden)]
        
        self.bias_hidden=[0.0 for _ in range(num_hidden)]

        self.weights_hidden_output=[
            random.uniform(-0.1, 0.1) for _ in range(num_hidden)
        ]
        self.bias_output=0.0

    def forward(self, symmetry_vector):
        hidden_activations = []
        for j in range(len(self.weights_input_hidden)):
            z=sum(w * x for w, x in zip(self.weights_input_hidden[j], symmetry_vector)) + self.bias_hidden[j]
            hidden_activations.append(math.tanh(z))  

        energy_i=sum(w * h for w, h in zip(self.weights_hidden_output, hidden_activations)) + self.bias_output
        return energy_i


class BehlerParrinelloModel:
    def __init__(self, element_types, num_inputs=2, num_hidden=5):
        self.atomic_nets = {element: AtomicNetwork(num_inputs, num_hidden) for element in element_types}

    def forward(self, coordinates, atomic_numbers):
        """
        coordinates: list of [x,y,z] positions for atoms
        atomic_numbers: list of atomic numbers (e.g. [8,1,1] for water)
        """
        # step 1: build symmetry functions (G1, G2 per atom)
        feature_vectors = symmetry_functions(coordinates)  # [[G1,G2], ...]
        
        # step 2: predict atomic energies with element-specific nets
        atomic_energies = []
        for idx, Z in enumerate(atomic_numbers):
            net = self.atomic_nets[Z]
            E_i = net.forward(feature_vectors[idx])
            atomic_energies.append(E_i)

        # step 3: total energy = sum of atomic energies
        total_energy = sum(atomic_energies)
        return total_energy, atomic_energies
