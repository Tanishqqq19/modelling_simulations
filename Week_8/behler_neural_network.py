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
    


    def backward_and_update(self, cache, dL_dy, lr):
        # output layer grads
        h = cache["h"]
        dW2 = [dL_dy * hj for hj in h]
        db2 = dL_dy

        # backprop to hidden
        dL_dh  = [dL_dy * w2 for w2 in self.weights_hidden_output]
        dL_dz1 = [dh * (1.0 - (hj * hj)) for dh, hj in zip(dL_dh, h)]  # tanh' = 1 - tanh^2

        # input -> hidden grads
        x = cache["x"]
        dW1 = [[dz * xi for xi in x] for dz in dL_dz1]
        db1 = dL_dz1

        # SGD update
        for j in range(len(self.weights_hidden_output)):
            self.weights_hidden_output[j] -= lr * dW2[j]
        self.bias_output -= lr * db2

        for j in range(len(self.weights_input_hidden)):
            row = self.weights_input_hidden[j]
            for i in range(len(row)):
                row[i] -= lr * dW1[j][i]
            self.bias_hidden[j] -= lr * db1[j]




class BehlerParrinelloModel:
    def __init__(self, element_types, num_inputs=2, num_hidden=5):
        self.atomic_nets = {element: AtomicNetwork(num_inputs, num_hidden) for element in element_types}

    def forward(self, coordinates, atomic_numbers):
        """
        coordinates: list[[x,y,z], ...]  (Å)
        atomic_numbers: list[int]
        returns: total_energy (float), per_atom (list[float]), caches (list[(Z, cache)])
        """
        features = symmetry_functions(coordinates)  # uses your paper-accurate G1/G2 + cutoff
        per_atom = []
        caches = []
        for idx, Z in enumerate(atomic_numbers):
            net = self.atomic_nets[int(Z)]
            e_i, cache = net.forward(features[idx])
            per_atom.append(e_i)
            caches.append((int(Z), cache))
        return sum(per_atom), per_atom, caches

    def train_one(self, coordinates, atomic_numbers, energy_ref, lr):
        # forward
        E_pred, per_atom, caches = self.forward(coordinates, atomic_numbers)
        # paper: minimize energy error (squared)
        diff = E_pred - energy_ref
        loss = 0.5 * diff * diff
        # dL/dE_total = (E_pred - E_ref); and since E_total = sum_i E_i, dE_total/dE_i = 1
        dL_dEtotal = diff
        for (Z, cache) in caches:
            self.atomic_nets[Z].backward_and_update(cache, dL_dEtotal, lr)
        return E_pred, loss


# -------- minimal trainer (expects pure-python data) --------
def train_bp(model, dataset, epochs=10, lr=1e-3, verbose_every=1):
    """
    dataset: list of dicts with *pure python* values:
      {"R": [[x,y,z], ...], "Z": [int,...], "E": float}
    """
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for s in dataset:
            R = s["R"]                      # list[[x,y,z], ...]
            Z = [int(z) for z in s["Z"]]    # list[int]
            E_ref = float(s["E"])           # float
            _, loss = model.train_one(R, Z, E_ref, lr)
            total_loss += loss
        if ep % verbose_every == 0:
            print(f"epoch {ep:4d} | mean loss {total_loss / max(1, len(dataset)):.6f}")
