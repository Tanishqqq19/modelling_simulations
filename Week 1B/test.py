import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT  # simple potential
from ase.md.langevin import Langevin
from ase import units
from ase.io.trajectory import Trajectory

# TIP3P bond geometry
angle_HOH = 104.52  # degrees
r_OH = 0.9572       # angstrom

# Convert angle to radians and compute positions
x = angle_HOH * np.pi / 180 / 2
positions = [
    [0, 0, 0],  # O
    [r_OH * np.cos(x), r_OH * np.sin(x), 0],  # H
    [r_OH * np.cos(x), -r_OH * np.sin(x), 0], # H
]

# Create molecule
atoms = Atoms('OH2', positions=positions)
atoms.center(vacuum=5.0)  # add empty space around it

# Add calculator (use TIP3P if required, else EMT is fine)
atoms.calc = EMT()

# Set up MD with Langevin thermostat
dyn = Langevin(
    atoms,
    timestep=1 * units.fs,
    temperature_K=300,
    friction=0.01,
    logfile='single_water.log'
)

# Output trajectory to visualize in VMD
traj = Trajectory('single_water.traj', 'w', atoms)
dyn.attach(traj.write, interval=1)

# Run simulation
dyn.run(1000)
