import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolfiles
import numpy as np




class AtomicVideoGenerator:
    def __init__(self):
        self.coordinates = []
        self.atom_types = []
        self.timesteps = []
        self.colors = {
            'H': 'blue',
            'C': 'black',
            'N': 'blue',
            'O': 'red',
            'P': 'orange',
            'S': 'yellow',
            'default': 'gray'
        }
        self.sizes = {
            'H': 30,
            'C': 50,
            'N': 45,
            'O': 40,
            'P': 55,
            'S': 50,
            'default': 40
        }
    
    def read_xyz_file(self, filename):
        """
        Read XYZ format file (common format for atomic coordinates)
        Format: atom_type x y z (one line per atom)
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines if present
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#'):
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        float(parts[1])  # Try to convert x coordinate
                        start_idx = i
                        break
                except ValueError:
                    continue
        
        atoms = []
        types = []
        
        for line in lines[start_idx:]:
            if line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    atom_type = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    atoms.append([x, y, z])
                    types.append(atom_type)
        
        return np.array(atoms), types
    

    def set_bonds(self, bond_list):
        self.bonds = bond_list

    
    def read_trajectory_files(self, file_pattern, num_frames=None):
        """
        Read multiple coordinate files representing trajectory
        file_pattern: string with {} placeholder for frame number
        Example: "frame_{}.xyz" for files frame_0.xyz, frame_1.xyz, etc.
        """
        frame = 0
        while True:
            filename = file_pattern.format(frame)
            if not os.path.exists(filename):
                break
            
            coords, types = self.read_xyz_file(filename)
            self.coordinates.append(coords)
            
            # Store atom types from first frame
            if frame == 0:
                self.atom_types = types
            
            self.timesteps.append(frame)
            frame += 1
            
            if num_frames and frame >= num_frames:
                break
        
        if not self.coordinates:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        
        print(f"Loaded {len(self.coordinates)} frames")
    
    def read_from_arrays(self, coordinate_arrays, atom_types):
        """
        Read coordinates from numpy arrays
        coordinate_arrays: list of numpy arrays, each shape (n_atoms, 3)
        atom_types: list of atom type strings
        """
        self.coordinates = coordinate_arrays
        self.atom_types = atom_types
        self.timesteps = list(range(len(coordinate_arrays)))
    
    def create_2d_animation(self, output_filename='atomic_animation_2d.mp4', 
                          fps=10, plane='xy', figsize=(10, 8)):
        """
        Create 2D animation projecting onto specified plane
        plane: 'xy', 'xz', or 'yz'
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine coordinate indices based on plane
        if plane == 'xy':
            idx1, idx2 = 0, 1
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        elif plane == 'xz':
            idx1, idx2 = 0, 2
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
        elif plane == 'yz':
            idx1, idx2 = 1, 2
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
        
        # Set axis limits based on all frames
        all_coords = np.concatenate(self.coordinates)
        margin = 2.0
        ax.set_xlim(all_coords[:, idx1].min() - margin, 
                   all_coords[:, idx1].max() + margin)
        ax.set_ylim(all_coords[:, idx2].min() - margin, 
                   all_coords[:, idx2].max() + margin)
        
        # Initialize empty scatter plot
        scat = ax.scatter([], [], s=50, c='yellow', alpha=0.8)



        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           verticalalignment='top', fontsize=12)
        
        bond_list = []
        for i in range(0, len(self.atom_types), 3):
            bond_list.append((i, i + 1))
            bond_list.append((i, i + 2))

        bond_lines = [ax.plot([], [], color='gray', linewidth=1)[0] for _ in bond_list]


        def animate(frame):
            coords = self.coordinates[frame]

            colors = [self.colors.get(atom, self.colors['default']) for atom in self.atom_types]
            sizes = [self.sizes.get(atom, self.sizes['default']) for atom in self.atom_types]

            scat.set_offsets(coords[:, [idx1, idx2]])
            scat.set_sizes(sizes)
            scat.set_color(colors)

            time_text.set_text(f'Frame: {frame}')

            # Draw bonds
            for line, (i, j) in zip(bond_lines, bond_list):
                x = [coords[i, idx1], coords[j, idx1]]
                y = [coords[i, idx2], coords[j, idx2]]
                line.set_data(x, y)

            return [scat, time_text] + bond_lines

        
        anim = FuncAnimation(fig, animate, frames=len(self.coordinates),
                           interval=1000//fps, blit=True, repeat=True)
        
        # Save animation
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='AtomicVideoGenerator'),
                            bitrate=1800)
        anim.save(output_filename, writer=writer)
        plt.close()
        
        print(f"2D animation saved as: {output_filename}")
    
    def create_3d_animation(self, output_filename='atomic_animation_3d.mp4', 
                          fps=10, figsize=(12, 9)):
        """
        Create 3D animation
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set axis limits
        all_coords = np.concatenate(self.coordinates)
        margin = 2.0
        ax.set_xlim(all_coords[:, 0].min() - margin, 
                   all_coords[:, 0].max() + margin)
        ax.set_ylim(all_coords[:, 1].min() - margin, 
                   all_coords[:, 1].max() + margin)
        ax.set_zlim(all_coords[:, 2].min() - margin, 
                   all_coords[:, 2].max() + margin)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        def animate(frame):
            ax.cla()  # Clear plot but keep axes
            ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
            ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
            ax.set_zlim(all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin)

            coords = self.coordinates[frame]
            colors = [self.colors.get(atom, self.colors['default']) for atom in self.atom_types]
            sizes = [self.sizes.get(atom, self.sizes['default']) for atom in self.atom_types]

            for i in range(len(coords)):
                ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                        c=colors[i], s=sizes[i]*5, alpha=0.9)

            # ✅ Draw bonds
            if hasattr(self, 'bonds'):
                for i, j in self.bonds:
                    bond = np.array([coords[i], coords[j]])
                    ax.plot(bond[:, 0], bond[:, 1], bond[:, 2], c='gray', linewidth=2)


            ax.set_title(f'Frame: {frame}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        
        anim = FuncAnimation(fig, animate, frames=len(self.coordinates),
                           interval=1000//fps, repeat=True)
        
        # Save animation
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='AtomicVideoGenerator'),
                            bitrate=1800)
        anim.save(output_filename, writer=writer)
        plt.close()
        
        print(f"3D animation saved as: {output_filename}")
    
    def add_bonds(self, bond_list, bond_colors='black', bond_width=1):
        """
        Add bonds between atoms (for more complex visualizations)
        bond_list: list of tuples (atom1_idx, atom2_idx)
        """
        # This would be implemented for more advanced visualizations
        # with bond connectivity
        pass

# Example usage and utility functions
def generate_sample_data_h2o_moving(n_frames=60, n_molecules=10):
    """
    Simulates multiple H₂O molecules with center-of-mass motion, vibration, and simple bouncing.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np

    # Build single water molecule
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    coords_base = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    atom_types_single = [atom.GetSymbol() for atom in mol.GetAtoms()]  # ['O', 'H', 'H']

    n_atoms = len(atom_types_single)
    atom_types = atom_types_single * n_molecules

    # Initialize molecule centers and velocities
    centers = np.random.uniform(1.5, 8.5, size=(n_molecules, 3))
    velocities = np.random.uniform(-0.05, 0.05, size=(n_molecules, 3))

    coordinates = []

    for frame in range(n_frames):
        frame_atoms = []

        # Add vibration and place atoms
        for i in range(n_molecules):
            noise = np.random.normal(0, 0.01, coords_base.shape)
            mol_coords = coords_base + noise + centers[i]
            frame_atoms.append(mol_coords)

        coords = np.vstack(frame_atoms)
        coordinates.append(coords)

        # Elastic bounce between O atoms (1st atom of each molecule)
        for i in range(n_molecules):
            for j in range(i + 1, n_molecules):
                O_i = coords_base[0] + centers[i]
                O_j = coords_base[0] + centers[j]
                distance = np.linalg.norm(O_i - O_j)
                if distance < 1.8:
                    # Swap velocities for simple elastic bounce
                    velocities[i], velocities[j] = -velocities[i], -velocities[j]

        # Update center positions
        centers += velocities

    return coordinates, atom_types




def create_sample_xyz_files():
    """Create sample XYZ files for testing"""
    coords, atom_types = generate_sample_data_h2o_moving()
    
    for i, frame_coords in enumerate(coords):
        filename = f"frame_{i:03d}.xyz"
        with open(filename, 'w') as f:
            f.write(f"{len(atom_types)}\n")
            f.write(f"Frame {i}\n")
            for atom_type, coord in zip(atom_types, frame_coords):
                f.write(f"{atom_type} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
    
    print(f"Created {len(coords)} sample XYZ files")

# Main execution example
if __name__ == "__main__":
    # Example 1: Generate sample data and create videos
    print("Generating sample data...")
    coordinates, atom_types = generate_sample_data_h2o_moving()

    bond_list = []
    for i in range(0, len(atom_types), 3):  # Every H₂O molecule has 3 atoms: O, H, H
        bond_list.append((i, i + 1))  # O-H
        bond_list.append((i, i + 2))  # O-H

    
    # Create video generator
    generator = AtomicVideoGenerator()
    generator.read_from_arrays(coordinates, atom_types)
    
    # Create 2D animation
    print("Creating 2D animation...")
    generator.create_2d_animation('new_2d_h2o.mp4', fps=5, plane='xy')
    
    # Create 3D animation
    print("Creating 3D animation...")
    generator.create_3d_animation('new_3d_h2o.mp4', fps=5)
    
    # Example 2: Create sample XYZ files and read them
    print("\nCreating sample XYZ files...")
    create_sample_xyz_files()
    
    # Read from files
    generator2 = AtomicVideoGenerator()
    generator2.read_trajectory_files("frame_{:03d}.xyz", num_frames=20)
    generator2.create_2d_animation('from_files_2d.mp4', fps=10, plane='xz')
    
    print("\nAll animations created successfully!")
    print("You'll need ffmpeg installed to generate MP4 files.")
    print("Install with: pip install ffmpeg-python")


# ===== Classical MD Simulation Engine =====


import numpy as np

# Lennard-Jones parameters (ε in kcal/mol, σ in Å)
LJ_PARAMS = {
    'H': {'epsilon': 0.046, 'sigma': 1.358},
    'O': {'epsilon': 0.1521, 'sigma': 3.1507}
}

# Atomic masses (approximate, in atomic mass units)
ATOMIC_MASS = {
    'H': 1.008,
    'O': 15.999
}

class Atom:
    def __init__(self, position, velocity, atom_type):
        self.x = np.array(position, dtype=np.float64)  # position
        self.v = np.array(velocity, dtype=np.float64)  # velocity
        self.f = np.zeros(3, dtype=np.float64)         # force
        self.type = atom_type
        self.mass = ATOMIC_MASS[atom_type]

def lennard_jones_force(r_vec, epsilon, sigma):
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros(3)
    sr6 = (sigma / r)**6
    force_mag = 24 * epsilon / r * (2 * sr6**2 - sr6)
    return force_mag * (r_vec / r)

def compute_forces(atoms,bond_list):
    n = len(atoms)
    for atom in atoms:
        atom.f[:] = 0.0

    total_energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r_vec = atoms[j].x - atoms[i].x
            r = np.linalg.norm(r_vec)
            if r == 0:
                continue

            eps_i = LJ_PARAMS[atoms[i].type]['epsilon']
            sig_i = LJ_PARAMS[atoms[i].type]['sigma']
            eps_j = LJ_PARAMS[atoms[j].type]['epsilon']
            sig_j = LJ_PARAMS[atoms[j].type]['sigma']

            epsilon = np.sqrt(eps_i * eps_j)
            sigma = 0.5 * (sig_i + sig_j)

            sr6 = (sigma / r)**6
            lj_energy = 4 * epsilon * (sr6**2 - sr6)
            force = lennard_jones_force(r_vec, epsilon, sigma)

            atoms[i].f += force
            atoms[j].f -= force
            total_energy += lj_energy
        
        for i, j in bond_list:
            r_vec = atoms[j].x - atoms[i].x
            f = bond_force(r_vec)
            atoms[i].f += f
            atoms[j].f -= f

    return total_energy

    return total_energy


    for i, j in bond_list:
        r_vec = atoms[j].x - atoms[i].x
        f = bond_force(r_vec)
        atoms[i].f += f
        atoms[j].f -= f


def run_md_simulation(atom_positions, atom_types, n_frames=100, dt=0.5):
    atoms = []
    for pos, typ in zip(atom_positions, atom_types):
        velocity = np.random.normal(0, 0.05, 3)
        atoms.append(Atom(position=pos, velocity=velocity, atom_type=typ))


    bond_list = []
    n_molecules = len(atom_positions) // 3
    for i in range(n_molecules):
        offset = i * 3
        bond_list += [(offset, offset + 1), (offset, offset + 2)]




    coordinates = []
    compute_forces(atoms, bond_list)

    for _ in range(n_frames):
        # First half step: position update
        for atom in atoms:
            atom.x += atom.v * dt + 0.5 * atom.f / atom.mass * dt**2

        old_forces = [atom.f.copy() for atom in atoms]
        compute_forces(atoms, bond_list)
        
        # Second half step: velocity update
        for atom, f_old in zip(atoms, old_forces):
            atom.v += 0.5 * (atom.f + f_old) / atom.mass * dt

        # Record positions
        frame_coords = np.array([atom.x.copy() for atom in atoms])
        coordinates.append(frame_coords)

    atom_type_list = [atom.type for atom in atoms]
    return coordinates, atom_type_list



# ----------------- Classical MD Utilities -------------------

def lennard_jones_force(r_vec, epsilon=0.65, sigma=3.15):
    r = np.linalg.norm(r_vec)
    if r < 1e-10:
        return np.zeros(3)
    force = 48 * epsilon * ((sigma**12 / r**13) - 0.5 * (sigma**6 / r**7)) * r_vec / r
    return force

def bond_force(r_vec, r0=0.96, k=450):
    r = np.linalg.norm(r_vec)
    return -k * (r - r0) * (r_vec / r)

def velocity_verlet_update(positions, velocities, forces, masses, dt):
    new_positions = positions + velocities * dt + 0.5 * (forces / masses[:, np.newaxis]) * dt**2
    return new_positions



def generate_sample_data_h2o_moving(n_frames=60, n_molecules=10):
    """
    Simulates multiple H₂O molecules with center-of-mass motion, vibration, and simple bouncing.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np

    # Build single water molecule
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    coords_base = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    atom_types_single = [atom.GetSymbol() for atom in mol.GetAtoms()]  # ['O', 'H', 'H']

    n_atoms = len(atom_types_single)
    atom_types = atom_types_single * n_molecules

    # Initialize molecule centers and velocities
    centers = np.random.uniform(1.5, 8.5, size=(n_molecules, 3))
    velocities = np.random.uniform(-0.05, 0.05, size=(n_molecules, 3))

    coordinates = []

    for frame in range(n_frames):
        frame_atoms = []

        # Add vibration and place atoms
        for i in range(n_molecules):
            noise = np.random.normal(0, 0.01, coords_base.shape)
            mol_coords = coords_base + noise + centers[i]
            frame_atoms.append(mol_coords)

        coords = np.vstack(frame_atoms)
        coordinates.append(coords)

        # Elastic bounce between O atoms (1st atom of each molecule)
        for i in range(n_molecules):
            for j in range(i + 1, n_molecules):
                O_i = coords_base[0] + centers[i]
                O_j = coords_base[0] + centers[j]
                distance = np.linalg.norm(O_i - O_j)
                if distance < 1.8:
                    # Swap velocities for simple elastic bounce
                    velocities[i], velocities[j] = -velocities[i], -velocities[j]

        # Update center positions
        centers += velocities

    return coordinates, atom_types
