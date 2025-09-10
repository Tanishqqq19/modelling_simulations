import torch, numpy as np, matplotlib.pyplot as plt
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"
device

"""Data is stored as Tensors.

The structures is a Toy Dataset.

The train and test data also gets split over here.
"""

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_one():
    Z = np.array([8, 1, 1])  # O, H, H
    R = np.random.rand(3, 3) * 2.0  # random positions in a 2 Å cube
    E = np.random.uniform(-80, -75) # fake energies near -76
    return {
        "R": torch.tensor(R, dtype=torch.float32, device=device),
        "Z": torch.tensor(Z, dtype=torch.long, device=device),
        "E": torch.tensor(E, dtype=torch.float32, device=device)
    }

structures = [make_one() for _ in range(1000)]
elements = sorted({int(z.item()) for s in structures for z in s["Z"]})


"""
1: Z=[8, 1, 1] E=-75.9735 R=[[0.414000004529953, 0.0820000022649765, 0.9200000166893005], [1.2680000066757202, 0.703000009059906, 0.7760000228881836], [1.0700000524520874, 1.3799999952316284, 1.5429999828338623]]
Here:
1 is the index
[8,1,1] ---> Here there are 8 oxygen atoms, 1 hydrogen atom, 1 hydrogen atom
It then gives cartesian coordinates of each atom
"""

class AtomicNet(nn.Module):
    def __init__(self, dim_in, hidden=(16,16)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1)
        )
    def forward(self, G):  # (n_i, F)
        return self.net(G).squeeze(-1)  # (n_i,)

class BPModel(nn.Module):
    """
    Behler–Parrinello model using YOUR NumPy symmetry functions inside.
    - Features per atom = [G2_radial, G4_angular]  --> dim_in = 2
    - One AtomicNet per element; total energy = sum of atomic energies
    NOTE: Features use NumPy -> no autograd path to positions (no forces).
    """
    def __init__(self, elements, Rc=6.0, hidden=(16,16)):
        super().__init__()
        self.Rc = float(Rc)
        self.dim_in = 2  # [radial, angular]
        self.elements = [int(z) for z in elements]
        self.elem_nets = nn.ModuleDict({
            str(z): AtomicNet(self.dim_in, hidden) for z in self.elements
        })


    def cutoff_function_np(self, r_ij):
        r_ij = np.asarray(r_ij)
        Rc = self.Rc
        return np.where(
            r_ij <= Rc,
            0.5 * (np.cos(np.pi * r_ij / Rc) + 1.0),
            0.0
        )

    def radial_symmetry_function_np(self, positions, i):
        eta = 1.0
        Rs  = 0.0

        r_ij = []
        for j in range(len(positions)):
            if j == i:
                continue
            dist = np.linalg.norm(positions[i] - positions[j])
            r_ij.append(dist)

        r_ij = np.array(r_ij, dtype=np.float64)
        fc = self.cutoff_function_np(r_ij)
        # G2 radial
        term = np.exp(-eta * (r_ij - Rs) ** 2) * fc
        return float(np.sum(term))

    def angular_symmetry_function_np(self, positions, i):
        n = len(positions)
        out = 0.0

        # Params (as in your code)
        eta   = 1.0
        zeta  = 1.0
        lambd = 1.0
        Rc    = self.Rc

        def cutoff(r):
            return 0.5 * (np.cos(np.pi * r / Rc) + 1.0) if r <= Rc else 0.0

        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue

                r_ij = np.linalg.norm(positions[i] - positions[j])
                r_ik = np.linalg.norm(positions[i] - positions[k])
                r_jk = np.linalg.norm(positions[j] - positions[k])

                if r_ij > Rc or r_ik > Rc or r_jk > Rc:
                    continue

                v_ij = positions[j] - positions[i]
                v_ik = positions[k] - positions[i]
                cos_theta = np.dot(v_ij, v_ik) / (np.linalg.norm(v_ij) * np.linalg.norm(v_ik))

                ang = (1.0 + lambd * cos_theta) ** zeta
                rad = np.exp(-eta * (r_ij**2 + r_ik**2 + r_jk**2))
                fc  = cutoff(r_ij) * cutoff(r_ik) * cutoff(r_jk)

                out += ang * rad * fc

        out *= 2.0 ** (1.0 - zeta)
        return float(out)

    # ---------- forward: build features directly, then sum atomic energies ----------

    def forward(self, R, Z, per_atom=False):
        """
        R: (N,3) float32 torch tensor
        Z: (N,)  int64  torch tensor (used to route to element nets)
        """
        # Build per-atom features using YOUR NumPy functions
        R_np = R.detach().cpu().numpy()
        N = R_np.shape[0]
        feats = np.zeros((N, self.dim_in), dtype=np.float32)
        for i in range(N):
            feats[i, 0] = self.radial_symmetry_function_np(R_np, i)
            feats[i, 1] = self.angular_symmetry_function_np(R_np, i)
        G = torch.from_numpy(feats).to(R.device)  # (N, 2)

        # Element-specific AtomicNet predictions
        E_atoms = torch.zeros(N, device=R.device)
        for z in torch.unique(Z):
            mask = (Z == z)
            if mask.any():
                E_atoms[mask] = self.elem_nets[str(int(z.item()))](G[mask])

        return E_atoms if per_atom else E_atoms.sum()


model = BPModel(elements=elements, hidden=(16, 16)).to(device)


epochs  = 200
loss_fn = nn.MSELoss()
opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 80/20 split
idx = torch.randperm(len(structures))
split = max(1, int(0.8 * len(idx)))
train = [structures[i] for i in idx[:split].tolist()]
test  = [structures[i] for i in idx[split:].tolist()]
print(f"n_train: {len(train)} | n_test: {len(test)} | elements: {elements}")


train_curve, test_curve = [], []

for ep in range(epochs):
    model.train()
    order = torch.randperm(len(train))
    losses = []

    for j in order:
        s = train[int(j)]
        E_pred = model(s["R"], s["Z"])          # forward (total energy)
        loss   = loss_fn(E_pred, s["E"])        # MSE

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # epoch means
    train_loss = float(np.mean(losses)) if losses else float("nan")
    train_curve.append(train_loss)

    model.eval()
    with torch.inference_mode():
        test_losses = [loss_fn(model(s["R"], s["Z"]), s["E"]).item() for s in test]
    test_loss = float(np.mean(test_losses)) if test_losses else float("nan")
    test_curve.append(test_loss)

    if ep % 10 == 0 or ep == epochs - 1:
        print(f"epoch {ep:4d} | train {train_loss:.6f} | test {test_loss:.6f}")

# -------------------------------------------------------
# 4) PLOTS
# -------------------------------------------------------
xs = range(epochs)

# Loss curves (log scale) - helpful when early loss is huge
plt.figure(figsize=(8,5))
plt.semilogy(xs, np.maximum(train_curve, 1e-12), label="Train (log)")
plt.semilogy(xs, np.maximum(test_curve,  1e-12), label="Test (log)")
plt.xlabel("Epoch"); plt.ylabel("MSE loss (log)")
plt.title("Training & Test Loss (log-scale)")
plt.legend(); plt.grid(True, which="both"); plt.tight_layout()
plt.show()
