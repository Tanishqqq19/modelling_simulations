# **Behler–Parrinello Neural Network (BPNN) — PyTorch Implementation**

I built a **Behler–Parrinello Neural Network (BPNN)** from scratch in **PyTorch** to model **atomic potential energy surfaces** using **symmetry functions**.  
The network learns how local atomic environments contribute to the total energy of a molecule but at a fraction of the computational cost.

This project reproduces the architecture from *Behler & Parrinello (2007)* and includes:
- Computation of symmetry functions  
- A fully connected **atomic neural network** for per-atom energy prediction  
- **data generation** for a 2-particle system  
- End-to-end **training pipeline** and **energy surface visualization**  

---

## **Sample Results**


---

## **Project Overview**

The Behler–Parrinello approach decomposes the total molecular energy into **atomic contributions**, each computed by a small neural network that takes symmetry-based descriptors of the atom’s neighborhood as input.  
By using **radial** and **angular** symmetry functions, the model maintains **rotational, translational, and permutational invariance**, allowing it to generalize across different atomic configurations.

---

## **Data Generation**

Instead of relying on existing molecular datasets, I **generated all the data from scratch** to maintain complete control over the physics behind the system.  
The dataset represents a **two-particle molecular interaction**, where each data point corresponds to:
- Randomly sampled **interatomic distances**
- Computed **potential energies** using a physically realistic analytical potential (e.g., Lennard-Jones–like form)
- Normalized energy values for numerical stability during training  

All configurations and energies were stored in a CSV file (`simulation_data_2_particles.csv`), which the PyTorch `Dataset` class reads during training.

---

## **Implementation Highlights**

- **Framework:** PyTorch  
- **Visualization:** Matplotlib for learning curves and convergence plots  