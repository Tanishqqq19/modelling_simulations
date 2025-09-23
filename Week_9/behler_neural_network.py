import random
import math

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

def symmetry_functions(R): # Each particle is represented by symmetry functions. This simplifies the process.
    g1 = G1_single(R, Rc, eta_G1, Rs_G1)
    g2 = G2_single(R, Rc, eta_G2, zeta_G2, lambda_G2)
    return [[g1[i], g2[i]] for i in range(len(R))]

class AtomicNetwork:
    def __init__(self, num_inputs, num_hidden):  # Eqn 1 is being coded out here
        
        self.input_to_hidden_weights=[
            [random.uniform(-0.1, 0.1) for i in range(num_inputs)]
            for i in range(num_hidden)
        ]
        
        self.hidden_biases = [0.0 for i in range(num_hidden)]
        self.hidden_to_output_weights = [
            random.uniform(-0.1, 0.1) for i in range(num_hidden)
        ]
        self.output_bias = 0.0

    def forward(self, symmetry_vector):
        hidden_linear = []
        hidden_activation = []
        for j in range(len(self.input_to_hidden_weights)):
            
            linear_sum = sum(w*x for w,x in zip(self.input_to_hidden_weights[j], symmetry_vector))+self.hidden_biases[j]
            
            hidden_linear.append(linear_sum)
            
            hidden_activation.append(math.tanh(linear_sum))
        
        
        atomic_energy = sum(w*h for w,h in zip(self.hidden_to_output_weights, hidden_activation))+self.output_bias
        
        cache = {
            "inputs":symmetry_vector,
            "hidden_linear":hidden_linear,
            "hidden_activation":hidden_activation,
            "output":atomic_energy
        }
        
        return atomic_energy,cache

    def backward_and_update(self, cache, dL_d_output, learning_rate): # Eqn 14
        # This is basically coding out all the four derivative equations.
        hidden_activation = cache["hidden_activation"]
        dW_output = [dL_d_output * h for h in hidden_activation]
        db_output = dL_d_output
        dL_d_hidden_activation = [dL_d_output * w for w in self.hidden_to_output_weights]
        dL_d_hidden_linear = [dh * (1.0 - (h * h)) for dh, h in zip(dL_d_hidden_activation, hidden_activation)]
        inputs = cache["inputs"]
        dW_input = [[dz * x for x in inputs] for dz in dL_d_hidden_linear]
        db_hidden = dL_d_hidden_linear
        
        for j in range(len(self.hidden_to_output_weights)):
            self.hidden_to_output_weights[j] -= learning_rate * dW_output[j]
        self.output_bias -= learning_rate * db_output
        for j in range(len(self.input_to_hidden_weights)):
            for i in range(len(self.input_to_hidden_weights[j])):
                self.input_to_hidden_weights[j][i] -= learning_rate * dW_input[j][i]
            self.hidden_biases[j] -= learning_rate * db_hidden[j]


import pandas as pd
import matplotlib.pyplot as plt

train_files = [
    # "./3particles.csv",
    # "./harmonic_chain_5particles.csv",
    # "./harmonic_chain_2particles.csv",
    # "./harmonic_chain_6particles.csv",
    # "./harmonic_chain_4particles.csv",
    "./Week_8/simulation_data_2_particles.csv"
]
test_file = ""
import ast

all_samples = []
for f in train_files:
    df = pd.read_csv(f)

    # t  = df[0].tolist()
    # n  = df[1].tolist()
    # E  = df[2].tolist()

    x = df['x'].tolist()
    # TODO: replace with V
    E = df['V'].tolist()

    for xi, Ei in zip(x, E):
        # not using the right positions of the particles
        all_samples.append({"x": ast.literal_eval(xi), "E": float(Ei)})
        # all_samples.append()

# TODO: remove this and simplify 
all_samples = all_samples[:2000]

split_idx = int(0.8 * len(all_samples))

print("Length of all  :", len(all_samples)) #


train_samples = all_samples[:split_idx]
test_samples  = all_samples[split_idx:]

print("Length of train :", len(train_samples)) #
print("Length of test :", len(test_samples)) #




def init_direct_model(num_hidden=50):
    return AtomicNetwork(num_inputs=2, num_hidden=num_hidden)

def forward_energy(net, x):
    return net.forward(x)

def train_epoch(net, train_samples, lr):
    total_loss = 0.0
    for s in train_samples:
        y_pred, cache = forward_energy(net, s["x"])
        diff = y_pred - s["E"]
        loss = 0.5 * diff * diff
        net.backward_and_update(cache, dL_d_output=diff, learning_rate=lr)
        total_loss += loss
    return total_loss / len(train_samples)

def evaluate(net, test_samples):
    total = 0.0
    for s in test_samples:
        y_pred, _ = forward_energy(net, s["x"])
        diff = y_pred - s["E"]
        total += diff * diff
    return total / len(test_samples)

net = init_direct_model(num_hidden=100)
epochs = 400
lr = 1e-4

train_curve = []
test_curve  = []

for ep in range(1, epochs+1):
    train_mse = train_epoch(net, train_samples, lr)
    test_mse  = evaluate(net, test_samples)
    train_curve.append(train_mse)
    test_curve.append(test_mse)
    if ep % 20 == 0:
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
