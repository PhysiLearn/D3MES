import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, QED, rdMolDescriptors
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# ----- Step 1: Convert genlogpqedclass1.npz to SDF files -----
# Load NPZ data
data = np.load('clagenclass0.npz')

# Create output folder for SDF files
output_folder = 'class1_sdf'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get atomic data
samples = data['data']

# Define a function for rounding bond values
def round_bond_value(value):
    bond_values = [1, 1.5, 2, 3, 4]
    return min(bond_values, key=lambda x: abs(x - value))

# Map for elements (example for 'C', 'N', 'O', etc.)
element_map = {
    'Ag': [1, 0, 0], 'Al': [0, 1, 0], 'As': [0, 0, 1], 'B': [1, 1, 0], 'Ba': [1, 0, 1],
    'Br': [0, 1, 1], 'C': [1, 1, 1], 'Ca': [2, 0, 0], 'Cl': [0, 2, 0], 'F': [0, 0, 2],
    'I': [2, 2, 0], 'K': [2, 0, 2], 'Li': [0, 2, 2], 'Mg': [2, 2, 2], 'N': [3, 0, 0],
    'Na': [0, 3, 0], 'O': [0, 0, 3], 'P': [3, 3, 0], 'Ra': [3, 0, 3], 'S': [0, 3, 3],
    'Se': [3, 3, 3], 'Si': [4, 0, 0], 'Te': [0, 4, 0], 'Zn': [0, 0, 4]
}

# Process each molecule and create SDF files
for i, sample in enumerate(samples):
    positions = sample[0]  # First channel: atomic positions
    elements = sample[1]  # Second channel: element data
    bonds = sample[2]  # Third channel: bond connections

    # Remove all-zero elements
    non_zero_elements_idx = np.any(np.round(elements) != 0, axis=1)
    elements = elements[non_zero_elements_idx]
    positions = positions[non_zero_elements_idx]

    # Replace element information
    element_symbols = []
    for e in elements:
        element_key = next((key for key, value in element_map.items() if np.array_equal(np.round(e), value)), 'C')
        element_symbols.append(element_key)

    # Clean bond data and remove all-zero rows
    bond_data = []
    for bond in bonds:
        bond = np.round(bond)
        if np.all(bond[:2] != 0):
            bond[2] = round_bond_value(bond[2])
            bond_data.append(bond)

    # Create molecule using RDKit
    mol = Chem.RWMol()
    atom_idx_map = {}

    existing_bonds = set()

    # Add atoms to the molecule
    for j, element in enumerate(element_symbols):
        atom = Chem.Atom(element)
        atom_idx = mol.AddAtom(atom)
        atom_idx_map[j] = atom_idx

    # Add bonds to the molecule
    for bond in bond_data:
        atom1, atom2 = bond[:2]
        if atom1 not in atom_idx_map or atom2 not in atom_idx_map:
            continue
        if atom1 == atom2:
            continue
        bond_type = bond[2]
        if bond_type == 1.0:
            bond_type = Chem.BondType.SINGLE
        elif bond_type == 2.0:
            bond_type = Chem.BondType.DOUBLE
        elif bond_type == 3.0:
            bond_type = Chem.BondType.TRIPLE
        elif bond_type == 4.0:
            bond_type = Chem.BondType.QUADRUPLE
        elif bond_type == 1.5:
            bond_type = Chem.BondType.AROMATIC

        bond_pair = tuple(sorted([atom_idx_map[atom1], atom_idx_map[atom2]]))
        if bond_pair not in existing_bonds:
            mol.AddBond(atom_idx_map[atom1], atom_idx_map[atom2], bond_type)
            existing_bonds.add(bond_pair)

    # Generate SDF file
    sdf_file_name = os.path.join(output_folder, f'mol{i + 1}.sdf')
    writer = Chem.SDWriter(sdf_file_name)
    writer.write(mol)
    writer.close()

print("SDF files have been generated.")

# ----- Step 2: Calculate LogP and QED for each SDF -----
def calculate_logp_qed(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file)
    molecule_data = []
    for mol in suppl:
        if mol is not None:
            logp = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
            qed_value = QED.qed(mol)
            mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else 'Unknown'
            molecule_data.append([mol_name, logp, qed_value])
    return molecule_data

# Get all SDF files
sdf_folder = 'class1_sdf'
sdf_files = [os.path.join(sdf_folder, f) for f in os.listdir(sdf_folder) if f.endswith('.sdf')]

# Store results in a list
all_molecule_data = []

for sdf_file in sdf_files:
    all_molecule_data.extend(calculate_logp_qed(sdf_file))

# Save results to CSV
df = pd.DataFrame(all_molecule_data, columns=["Name", "LogP", "QED"])
df.to_csv('genlogpqed_class1.csv', index=False)

print("LogP and QED values have been written to genlogpqed_class1.csv.")

# ----- Step 3: Plot LogP and QED Distribution -----
# Load the generated CSV file
df_class1 = pd.read_csv('genlogpqed_class1.csv')

# LogP and QED data
logp_data = df_class1['LogP'].dropna().astype(float)
qed_data = df_class1['QED'].dropna().astype(float)

# Plot LogP distribution
plt.figure(figsize=(10, 6))
sns.histplot(logp_data, kde=True, color='orange', bins=30, label='LogP Distribution')
plt.title('LogP Distribution')
plt.xlabel('LogP')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Plot QED distribution
plt.figure(figsize=(10, 6))
sns.histplot(qed_data, kde=True, color='deepskyblue', bins=30, label='QED Distribution')
plt.title('QED Distribution')
plt.xlabel('QED')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
