import random
import math
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import ast 

Rc=6.0        
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

def G1_single(R, Rc, eta, Rs): # Radial Function
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
    return out

def G2_single(R, Rc, eta, zeta, lam): # Angular Function
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




class SymmetryFunctionLayer(nn.Module):
    def __init__(self, Rc=6.0, eta_G1=0.2, Rs_G1=2.0,
                 eta_G2=0.01, zeta_G2=1.0, lambda_G2=1.0):
        super().__init__()
        self.Rc = Rc
        self.eta_G1 = eta_G1
        self.Rs_G1 = Rs_G1
        self.eta_G2 = eta_G2
        self.zeta_G2 = zeta_G2
        self.lambda_G2 = lambda_G2

    def forward(self, R):
        # R: [batch, N_atoms, 3]
        batch_size, N, _ = R.shape
        out = []
        for b in range(batch_size):
            coords = R[b].cpu().tolist()
            g1 = G1_single(coords, self.Rc, self.eta_G1, self.Rs_G1)
            g2 = G2_single(coords, self.Rc, self.eta_G2, self.zeta_G2, self.lambda_G2)
            out.append([[g1[i], g2[i]] for i in range(N)])
        return torch.tensor(out, dtype=torch.float32, device=R.device)  # [batch, N, 2]


def symmetry_functions(R): # Each particle is represented by symmetry functions. This simplifies the process.
    g1 = G1_single(R, Rc, eta_G1, Rs_G1)
    g2 = G2_single(R, Rc, eta_G2, zeta_G2, lambda_G2)
    return [[g1[i], g2[i]] for i in range(len(R))]






class AtomicNetwork(nn.Module):
    def __init__(self, num_inputs=2, hidden_size=15):
        super().__init__()
        self.sym_layer = SymmetryFunctionLayer()
        self.W1 = nn.Linear(num_inputs, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, 1)

    def forward(self, R):
        x = self.sym_layer(R)   # compute [G1, G2]
        x = torch.tanh(self.W1(x))
        x = torch.tanh(self.W2(x))
        x = self.W3(x)
        return x

class Dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = torch.tensor(ast.literal_eval(self.df['x'].iloc[idx]), dtype=torch.float32)  
        V = torch.tensor([self.df['V'].iloc[idx]], dtype=torch.float32)
        return x, V


def train_epoch(net, optimizer, dataloader):
    total_loss = 0.0
    loss_fn = nn.MSELoss()
    
    for batch in dataloader:
        x_label, V_label = batch
        V_pred = net(x_label)
        loss = loss_fn(V_pred, V_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(net, dataloader):
    net.eval()
    total = 0.0

    loss_fn = nn.MSELoss()
    
    with torch.no_grad():
        for batch in dataloader:
            x_label, V_label = batch
            V_pred = net(x_label)
            loss = loss_fn(V_pred, V_label)
            total += loss.item()

    net.train()
    return total / len(dataloader)

# random.seed(0)
# torch.manual_seed(0)

fpath = "./simulation_data_2_particles.csv"
# fpath = "./Week_8/x1_times_x2.csv"
df = pd.read_csv(fpath)
df = df.iloc[:10000]

split = int(len(df) * 0.8)

train_df = df.iloc[:split]
test_df = df.iloc[split:]
print("Length of train", len(train_df))
print("Length of test", len(test_df))

net = AtomicNetwork(2, 15)

train_dataset = Dataset(train_df)
test_dataset = Dataset(test_df)

train_dataloader = DataLoader(train_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)


lr = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


epochs = 100

train_curve = []
test_curve  = []

for ep in range(1, epochs+1):
    train_mse = train_epoch(net, optimizer, train_dataloader)
    test_mse  = evaluate(net, test_dataloader)
    train_curve.append(train_mse)
    test_curve.append(test_mse)
    if ep % 2 == 0:
        print(f"epoch {ep:3d} | train MSE {train_mse:.16f} | test MSE {test_mse:.16f}")

plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_curve, label="Train MSE")
plt.plot(range(1, epochs+1), test_curve, label="Test MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
# plt.yscale('log')
plt.title("Training vs Test Curves")
plt.legend()
plt.grid(True)
plt.savefig("./loss_learn_curves_1.png")
plt.show()
