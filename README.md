# **Behler–Parrinello Neural Network (BPNN) — PyTorch Implementation**

I built a **Behler–Parrinello Neural Network (BPNN)** from scratch in **PyTorch** to model **atomic potential energy surfaces** using **symmetry functions**.  
The network learns how local atomic environments contribute to the total energy of a molecule — mimicking how quantum mechanical systems behave — but at a fraction of the computational cost.

This project reproduces the original concept from *Behler & Parrinello (2007)* and includes:
- Computation of **radial (G1)** and **angular (G2)** symmetry functions  
- A fully connected **atomic neural network** for per-atom energy prediction  
- End-to-end **training pipeline** on simulated 2-particle data  
- **Visualization of training and test losses** to track energy convergence  

---

## **Sample Results**

| Input Atomic Configuration | Predicted Energy Distribution |
|-----------------------------|-------------------------------|
| <img src="https://via.placeholder.com/150" width="150"> | <img src="https://via.placeholder.com/150" width="150"> |

*(Replace the placeholders with your real plots or structure images!)*

---

## **Project Overview**

The Behler–Parrinello approach breaks down the total molecular energy into **atomic contributions**, each determined by a local neural network.  
Instead of directly learning atomic coordinates, the model uses **invariant descriptors (G1 and G2)** that encode the geometry of neighboring atoms while preserving physical symmetries (rotation, translation, permutation).

This makes the model a powerful alternative to traditional force fields and a bridge between **machine learning** and **quantum chemistry**.

---

## **Implementation Highlights**

- **Language:** Python  
- **Framework:** PyTorch  
- **Input Features:** Radial & Angular Symmetry Functions  
- **Cutoff Function:** Smooth cosine cutoff for locality  
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam (`lr = 1e-4`)  
- **Epochs:** 10  
- **Visualization:** Matplotlib for learning curves  

---