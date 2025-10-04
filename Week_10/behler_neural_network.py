import random
import math
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import ast 


# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if torch.cuda.is_available():
  device = torch.device('cuda')
elif torch.backends.mps.is_available():
  device = torch.device('mps')
else:
  device = torch.device('cpu')

print(f"Using device: {device}")


Rc=6.0        
eta_G1=0.2
Rs_G1=2.0
eta_G2=0.01
zeta_G2=1.0
lambda_G2=1.0

radial_params = [
    {"eta": 0.02, "Rs": 0.0} #, {"eta": 0.02, "Rs": 1.0}, {"eta": 0.02, "Rs": 2.0}, {"eta": 0.02, "Rs": 3.0}, {"eta": 0.02, "Rs": 4.0},
    # {"eta": 0.05, "Rs": 0.0}, {"eta": 0.05, "Rs": 1.0}, {"eta": 0.05, "Rs": 2.0}, {"eta": 0.05, "Rs": 3.0}, {"eta": 0.05, "Rs": 4.0},
    # {"eta": 0.10, "Rs": 0.0}, {"eta": 0.10, "Rs": 1.0}, {"eta": 0.10, "Rs": 2.0}, {"eta": 0.10, "Rs": 3.0}, {"eta": 0.10, "Rs": 4.0},
    # {"eta": 0.20, "Rs": 0.0}, {"eta": 0.20, "Rs": 1.0}, {"eta": 0.20, "Rs": 2.0}, {"eta": 0.20, "Rs": 3.0}, {"eta": 0.20, "Rs": 4.0},
]

# === Angular symmetry functions (24 total) ===
# Grid: 3 etas × 4 zetas × 2 lambdas
angular_params = [
    # eta = 0.001
    {"eta": 0.001, "zeta": 1, "lam":  1} #, {"eta": 0.001, "zeta": 1, "lam": -1},
    # {"eta": 0.001, "zeta": 2, "lam":  1}, {"eta": 0.001, "zeta": 2, "lam": -1},
    # {"eta": 0.001, "zeta": 4, "lam":  1}, {"eta": 0.001, "zeta": 4, "lam": -1},
    # {"eta": 0.001, "zeta": 8, "lam":  1}, {"eta": 0.001, "zeta": 8, "lam": -1},
    # # eta = 0.005
    # {"eta": 0.005, "zeta": 1, "lam":  1}, {"eta": 0.005, "zeta": 1, "lam": -1},
    # {"eta": 0.005, "zeta": 2, "lam":  1}, {"eta": 0.005, "zeta": 2, "lam": -1},
    # {"eta": 0.005, "zeta": 4, "lam":  1}, {"eta": 0.005, "zeta": 4, "lam": -1},
    # {"eta": 0.005, "zeta": 8, "lam":  1}, {"eta": 0.005, "zeta": 8, "lam": -1},
    # # eta = 0.010
    # {"eta": 0.010, "zeta": 1, "lam":  1}, {"eta": 0.010, "zeta": 1, "lam": -1},
    # {"eta": 0.010, "zeta": 2, "lam":  1}, {"eta": 0.010, "zeta": 2, "lam": -1},
    # {"eta": 0.010, "zeta": 4, "lam":  1}, {"eta": 0.010, "zeta": 4, "lam": -1},
    # {"eta": 0.010, "zeta": 8, "lam":  1}, {"eta": 0.010, "zeta": 8, "lam": -1},
]



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
            # rij= math.sqrt(sum((ai - bi)**2 for ai, bi in zip(R[i], R[j])))
            rij = abs(R[i] - R[j])
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




class SymmetryFunctionLayer:
    def __init__(self, Rc=6.0, eta_G1=0.2, Rs_G1=2.0, eta_G2=0.01, zeta_G2=1.0, lambda_G2=1.0):
        super().__init__()
        self.Rc = Rc
        self.eta_G1 = eta_G1
        self.Rs_G1 = Rs_G1
        self.eta_G2 = eta_G2
        self.zeta_G2 = zeta_G2
        self.lambda_G2 = lambda_G2

    def forward(self, R):
        # R_batch shape: (batch, 2) → [pos1, pos2]
        # feats = []
        # for R in R_batch:
            # Convert to "atom list" form for your G1/G2 functions
        # R_np = [[R[0].item()], [R[1].item()]]  # 2 atoms in 1D

        # feats.append([g1[0], g2[0]])  # features for atom 1
        return 

    # def forward(self, R_batch):
    #     feats = []
    #     for R in R_batch:
    #         R_np = [[R[0].item()], [R[1].item()]]  # 2 atoms in 1D
    #         atom_feats = []
    #         for p in radial_params:
    #             g1 = G1_single(R_np, Rc, p["eta"], p["Rs"])
    #             atom_feats.append(g1[0])
    #         for p in angular_params:
    #             g2 = G2_single(R_np, Rc, p["eta"], p["zeta"], p["lam"])
    #             atom_feats.append(g2[0])
    #         feats.append(atom_feats)

    #     return torch.tensor(feats, dtype=torch.float32, device=R_batch.device)

# class AtomicNetwork(nn.Module):
#     def __init__(self, hidden_size=64):
#         super().__init__()

#         self.sym_layer = SymmetryFunctionLayer()
#         num_inputs = len(radial_params) + len(angular_params)  
#         self.W1 = nn.Linear(num_inputs, hidden_size)
#         self.W2 = nn.Linear(hidden_size, hidden_size)
#         self.W3 = nn.Linear(hidden_size, 1)

#     def forward(self, R):
#         x = self.sym_layer(R)   # compute [G1, G2]
#         x = torch.tanh(self.W1(x))
#         x = torch.tanh(self.W2(x))
#         x = self.W3(x)
#         return x


class AtomicNetwork(nn.Module):
    def __init__(self, hidden_size=20, Rc=6.0, eta_G1=0.2, Rs_G1=2.0, eta_G2=0.01, zeta_G2=1.0, lambda_G2=1.0):
        super().__init__()
        self.Rc = Rc
        self.eta_G1 = eta_G1
        self.Rs_G1 = Rs_G1
        self.eta_G2 = eta_G2
        self.zeta_G2 = zeta_G2
        self.lambda_G2 = lambda_G2
        
        num_inputs = len(radial_params) + len(angular_params)  # 2 here
        print('num_inputs to system', num_inputs)
        self.W1 = nn.Linear(num_inputs, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, 1)
        # self.W4 = nn.Linear(hidden_size, hidden_size)
        # self.W5 = nn.Linear(hidden_size, 1)
    def forward(self, R):

        # R is a batch of (batch_size, num_atoms)

        result = torch.zeros(len(R), dtype=R.dtype, device=R.device)

        # loop over batch
        for b in range(len(R)):
            E = 0 # E
            # calculate energy for each atom in train point
            # pass in all atoms in one train point
            # and get g1 and g2 values for all atoms
            R_list = R[b].detach().cpu().tolist()

            g1_all_atoms = G1_single(R_list, 2.0, 0.5, 0.0)
            # g2_all_atoms = G1_single(R_list, 2.0, 0.5, 1.0)
            # print('gs', g1_all_atoms, g2_all_atoms)
            g2_all_atoms = G2_single(R[b], self.Rc, self.eta_G2, self.zeta_G2, self.lambda_G2)
            
            for i in range(len(R[0])):
                # forward pass
                # x = self.sym_layer.forward(R[i])
                
                # import pdb; pdb.set_trace()

                # {"eta": 0.10, "Rs": 1.0}
                
                x = torch.tensor([g1_all_atoms[i], g2_all_atoms[i]], dtype=torch.float32, device=R.device)
                # x = R[b]
                
                # print('R[b]', R[b], 'g values', x)
                # import pdb; pdb.set_trace()

                # print('symmetry values', x)

                # pass through neural network
                x = torch.tanh(self.W1(x))
                
                # x is between -1 and 1
                x = torch.tanh(self.W2(x))
                x = self.W3(x) # E_i
                
                # print('output 1', x)
                # constrains x between 0 and infinity

                E += x
            # print('E', E)
            
            result[b] = E
        # print('result out', result)
            
        # x = torch.tanh(self.W4(x))
        return result


class Dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = torch.tensor([df['x1'].iloc[idx], df['x2'].iloc[idx]], dtype=torch.float32)  
        V = torch.tensor([self.df['V_normalized'].iloc[idx]], dtype=torch.float32)
        return x, V


def train_epoch(net, optimizer, dataloader):
    total_loss = 0.0
    loss_fn = nn.MSELoss()
    
    # for batch in dataloader:
    #     x_label, V_label = batch
    #     V_pred = net(x_label)
    #     loss = loss_fn(V_pred, V_label)

    for batch in dataloader:
        x_input, V_target = batch
        x_input = x_input.to(device)      
        V_target = V_target.to(device)      
        V_pred = net(x_input).unsqueeze(-1)
        # print('x_input', x_input, 'V_pred', V_pred)
        # print('V_target', V_target, 'V_pred', V_pred)
        # print('shapes', V_pred.shape, V_target.shape)

        # print('x_input', x_input)
        # print('V_target', V_target)
        # print('V_pred', V_pred)
        loss = loss_fn(V_pred, V_target)

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

            x_label = x_label.to(device)   
            V_label = V_label.to(device) 

            V_pred = net(x_label).unsqueeze(-1)
            loss = loss_fn(V_pred, V_label)
            total += loss.item()

    net.train()
    return total / len(dataloader)

# random.seed(0)
# torch.manual_seed(0)

fpath = "./simulation_data_2_particles.csv"
df = pd.read_csv(fpath)

# df['V_normalized'] = (df['V'] - df['V'].mean()) / df['V'].std()

df = df.iloc[:10_000]
# df = df.iloc[:5]
batch_size=10

split = int(len(df) * 0.8)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
df[['x1', 'x2']] = pd.DataFrame(df['x'].apply(ast.literal_eval).tolist(), index=df.index)


# df['x1_normalized'] = (df['x1'] - df['x1'].mean()) / df['x1'].std()
# df['x2_normalized'] = (df['x2'] - df['x2'].mean()) / df['x2'].std()
# df['V_normalized'] = (df['V'] - df['V'].mean()) / df['V'].std()
df['V_normalized'] = df['V']

# x1, x2 and V are mean 0 std 1

train_df = df.iloc[:split]
test_df = df.iloc[split:]
print("Length of train", len(train_df))
print("Length of test", len(test_df))

net = AtomicNetwork(10).to(device)

train_dataset = Dataset(train_df)
test_dataset = Dataset(test_df)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


lr = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


epochs = 10

train_curve = []
test_curve  = []

for ep in range(1, epochs+1):
    train_mse = train_epoch(net, optimizer, train_dataloader)
    test_mse  = evaluate(net, test_dataloader)
    train_curve.append(train_mse)
    test_curve.append(test_mse)
    if ep % 1 == 0:
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
