
class BehlerParrinelloModel:

    def __init__(self, element_types, num_inputs=2, num_hidden=5): 
        # This basically creates a neural network for each one of its elements.
        self.atomic_nets = {element: AtomicNetwork(num_inputs, num_hidden) for element in element_types}

    def forward(self, coordinates, atomic_numbers): # This maps to Eqn 12
        features = symmetry_functions(coordinates)
        per_atom = []
        caches = []
        for idx, Z in enumerate(atomic_numbers):
            net = self.atomic_nets[int(Z)]
            e_i, cache = net.forward(features[idx])
            per_atom.append(e_i)
            caches.append((int(Z), cache))
        return sum(per_atom), per_atom, caches

    def train_one(self, coordinates, atomic_numbers, energy_ref, lr):
        # This basically follows the error function equation. Eqn 14.

        E_pred, per_atom, caches = self.forward(coordinates, atomic_numbers)
        diff = E_pred - energy_ref
        loss = 0.5 * diff * diff
        dL_dEtotal = diff
        for (Z, cache) in caches:
            self.atomic_nets[Z].backward_and_update(cache, dL_dEtotal, lr)
        return E_pred, loss

def train_bp(model, dataset, epochs=10, lr=1e-3, verbose_every=1):
    #This basically loops over training samples and finds the averaged square error
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for s in dataset: 
            R = s["R"]
            Z = [int(z) for z in s["Z"]]
            E_ref = float(s["E"])
            _, loss = model.train_one(R, Z, E_ref, lr)
            total_loss += loss
        if ep % verbose_every == 0:
            print(f"epoch {ep:4d} | mean loss {total_loss / max(1, len(dataset)):.6f}")