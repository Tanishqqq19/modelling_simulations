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
        
        bond_list = [(0, 1), (0, 2), (3, 4), (3, 5)]
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
                        c=colors[i], s=sizes[i]*5, alpha=0.9)  # multiply size for better visibility

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
def generate_sample_data_h2o_moving(n_frames=60):
    """
    Two water molecules with center-of-mass motion + vibration.
    They bounce off each other on close contact.
    """
    # Create real H2O molecule
    smiles = "O"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    atom_types_single = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords_base = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    
    atom_types = atom_types_single * 3

    # Initial positions
    mol1_center = np.array([1.0, 2.0, 2.0])
    mol2_center = np.array([5.0, 2.0, 2.0])
    
    # Initial velocities
    v1 = np.array([0.05, 0.0, 0.0])  # molecule 1 to the right
    v2 = np.array([-0.05, 0.0, 0.0]) # molecule 2 to the left

    coordinates = []

    for frame in range(n_frames):
        # Simple vibration
        noise1 = np.random.normal(0, 0.01, coords_base.shape)
        noise2 = np.random.normal(0, 0.01, coords_base.shape)

        # Calculate new positions
        mol1 = coords_base + noise1 + mol1_center
        mol2 = coords_base + noise2 + mol2_center

        combined = np.vstack([mol1, mol2])
        coordinates.append(combined)

        # Distance between O atoms (index 0 and 3)
        O1 = mol1[0]
        O2 = mol2[0]
        dist = np.linalg.norm(O1 - O2)

        # Simple collision check
        if dist < 1.8:  # approximate contact threshold
            v1, v2 = -v1, -v2  # reverse direction (elastic bounce)

        # Update molecule centers
        mol1_center += v1
        mol2_center += v2

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
    
    # Create video generator
    generator = AtomicVideoGenerator()
    generator.read_from_arrays(coordinates, atom_types)
    
    # Create 2D animation
    print("Creating 2D animation...")
    generator.create_2d_animation('official_2d_h2o.mp4', fps=5, plane='xy')
    
    # Create 3D animation
    print("Creating 3D animation...")
    generator.create_3d_animation('official_3d_h2o.mp4', fps=5)
    
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