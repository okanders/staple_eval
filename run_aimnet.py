#
# COPY AS TO PLAY AROUND WITH ATOM MAPPINGS
#

import random
import os

import tqdm

import numpy as np
import h5py

import seaborn as sns
import matplotlib.pyplot as plt


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry.rdGeometry import Point3D

from ase import Atoms, units
from ase.optimize import BFGS
from ase.constraints import FixBondLengths
from ase.neighborlist import NeighborList, natural_cutoffs, get_connectivity_matrix

import torch
from aimnet import load_AIMNetSMD, load_AIMNetSMD_ens, AIMNetCalculator
from ase.calculators.mixing import SumCalculator
from dftd4.ase import DFTD4

#from dftd4.calculators import D4_model

if torch.cuda.is_available():
    model = load_AIMNetSMD_ens().cuda()
    print('gpu')
else:
    model = load_AIMNetSMD_ens()
    print('cpu')



#NEW
# Assuming 'model' is your AIMNet model already defined
# and AIMNetCalculator is an ASE-compatible calculator
aimnet_calculator = AIMNetCalculator(model)

# Create the DFTD4 calculator with the specified method
dftd4_calculator = DFTD4(method='wB97x')
#calculator= dftd4_calculator.set(method=aimnet_calculator)

# Combine the DFTD4 and AIMNet calculators
calculator = SumCalculator([dftd4_calculator, aimnet_calculator])
#calculator = dftd4_calculator.add_calculator(aimnet_calculator)

#OLD
#calculator = D4_model(xc='wB97x', calc=AIMNetCalculator(model))


#change
run_size = 10




# Define the date for the runs
date_for_runs = "04-03-2024"

# Simplified access to the scratch directory
scratch_directory = os.path.expanduser('~/scratch/okanders/b-hairpin-runs')

# Construct the parent directory path for this specific date
parent_directory = os.path.join(scratch_directory, date_for_runs)

# Example run name - replace or modify as needed
run_name = "run_size1_mixed_aromatics_staples_gpu_alphahelix_aimnet"
# Full path to the run-specific directory
run_directory = os.path.join(parent_directory, run_name)

# Subdirectories for different types of files within the run directory
hdf5_directory = os.path.join(run_directory, '_hdf5s')
pdb_directory = os.path.join(run_directory, '_pdbs')
figures_directory = os.path.join(run_directory, '_figures')

# Create these directories if they do not exist
for directory in [parent_directory, run_directory, hdf5_directory, pdb_directory, figures_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")



smiles_dict = {

    'WHL' : 'CC(=O)Nc1ccc(cc1)NC(=O)C',


    #Hydrogen Substitution on the Aromatic Ring:
    'WHL-H1': 'CC(=O)Nc1ccc(cc1)NC(=O)C', #(Original, for reference)
    #Slight Alteration: Introduce a subtle electronic effect by substituting one hydrogen with a halogen or a small group directly on the aromatic ring, not altering the beginning and end.

    #Isotopic Labeling of Aromatic Hydrogens:
    #no'WHL-D2': 'CC(=O)Nc1ccc(cc1D)NC(=O)C', #Replace one or more hydrogen atoms on the aromatic ring with deuterium. This would have minimal impact on chemical properties but can be useful for metabolic studies.

    #Introduction of a Small Alkyl Group on the Aromatic Ring:
    'WHL-Me1': 'CC(=O)Nc1ccc(cc1C)NC(=O)C', #(Adding a methyl group on the benzene ring for subtle lipophilicity changes).

    #Substituting an Aromatic Hydrogen with a Halogen:
    'WHL-Cl1': 'CC(=O)Nc1ccc(cc1Cl)NC(=O)C', #(Chlorine substitution for subtle electronic and steric effects).

    #Oxidation of an Aromatic Hydrogen to a Hydroxyl Group:
    'WHL-OH1': 'CC(=O)Nc1ccc(cc1O)NC(=O)C', #(Introducing a hydroxyl group for minimal polarity increase).

    #Incorporation of a Minimal Heteroatom:
    'WHL-N1': 'CC(=O)Nc1ccc(cc1N)NC(=O)C', #(Substituting an aromatic hydrogen with a nitrogen atom, subtly affecting the aromatic system's electronic nature).

    #Fluorination of the Aromatic Ring:
    'WHL-F2': 'CC(=O)Nc1ccc(cc1F)NC(=O)C', #(Introducing a fluorine atom, affecting electronic distribution and metabolic stability minimally).

    #Minor Aromatic Ring Expansion or Contraction:
    'WHL-benzo': 'CC(=O)Nc1ccccc1NC(=O)C', #(Expanding the aromatic system to a benzene ring, subtly impacting the molecule’s shape and electronic properties).

    #Bromination: 
    #no'WHL-Br1': 'CC(=O)Nc1ccc(cc1Br)NC(=O)C', # (Bromination can offer different electronic effects compared to chlorination).

    #Introduction of Substituents with Minimal Electronic Effects:
    'WHL-OCH3': 'CC(=O)Nc1ccc(cc1OC)NC(=O)C', # Methoxyl Group on Aromatic Ring: (Introduces a methoxyl group for slight polarity and electronic effects).

    'WHL1' : 'CC(=O)Nc1ccc(cc1C)NC(=O)C', #(Methyl substitution on the benzene ring)
    'WHL1a' : 'CC(=O)Nc1ccc(cc1Cl)NC(=O)C', #(Chlorine substitution on the benzene ring)
    'WHL1b' : 'CC(=O)Nc1ccncc1NC(=O)C', #(Substitution with a pyridine ring)









    # Failed staple alternatives: could not pass optimization or conformation embedding

    # Adjustments in Linker Composition**:
    # 'WHL2' : 'CC(=O)Nc1ccc(OCC)cc1NC(=O)C', #(Introduction of an ether linker)
    # 'WHL2a' : 'CC(=O)Nc1ccc(CCC)cc1NC(=O)C', #(Extension with a propyl linker)
    # 'WHL2b' : 'CC(=O)Nc1ccc(C#N)cc1NC(=O)C', #(Incorporation of a cyano linker)

    # Modifications of Ring Size or Introduction of Different Ring Systems**:
    # 'WHL3' : 'CC(=O)Nc1ccccc1NC(=O)C', #(Replacement with a benzene ring, increasing the size)
    # 'WHL3a' : 'CC(=O)Nc1c(C)cccc1NC(=O)C', #(Benzene ring with a methyl group)
    # 'WHL3b' : 'CC(=O)Nc1ccccc1C2CCCC2NC(=O)C', #(Introduction of a bicyclic system with cyclohexane)

    # Introducing Heteroatoms or Functional Groups in the Middle Section**:
    # 'WHL4' : 'CC(=O)Nc1ccc(cc1O)NC(=O)C', #(Introduction of a hydroxy group on the benzene ring)
    # 'WHL4a' : 'CC(=O)Nc1ccc(S)cc1NC(=O)C', #(Introduction of a thiol group on the benzene ring)
    # 'WHL4b' : 'CC(=O)Nc1cc(Cl)ccc1NC(=O)C', #(Introduction of a chlorine atom on the benzene ring)

    # Using Different Aromatic Systems**:
    # 'WHL5a' : 'CC(=O)Nc2cc1ccccc1cc2NC(=O)C', #(Naphthalene as the central aromatic system)


    #Modification of Aromatic Ring Saturation:

    #no'WHL-Hyd': 'CC(=O)Nc1c(c(cc1)H)NC(=O)C', # Partial Hydrogenation: (Adding hydrogen to create a dihydrobenzene structure for subtle changes in aromaticity).

    #Incorporation of Silicon in Place of Carbon:

    #no'WHL-SiMe3': 'CC(=O)Nc1ccc(cc1Si(CH3)3)NC(=O)C', # Silylation: (Replacing a hydrogen with a trimethylsilyl group introduces steric bulk and affects the molecule’s lipophilicity subtly).

    #Nitration of the Aromatic Ring:
    #no'WHL-NO2': 'CC(=O)Nc1ccc(cc1NO2)NC(=O)C', # Nitration: (Adding a nitro group to the ring introduces electronic effects and potential for further chemical modification).

    #Addition of Sulfur-Containing Groups:
    #no'WHL-SMe': 'CC(=O)Nc1ccc(cc1SCH3)NC(=O)C', # Thiomethyl Substitution: (Incorporating a thiomethyl group for minor changes in lipophilicity and electronic properties).

    #Isosteric Replacements:
    #no'WHL-Oxa': 'CC(=O)Nc1cc(no[n]1)NC(=O)C', # Oxadiazole for Benzene: (Replacing the benzene ring with an oxadiazole ring can subtly alter electronic distribution and conformation).

    #Alkyne Insertion:
    #no'WHL-Alk': 'CC(=O)Nc1ccc(cc#C)NC(=O)C', # (Introducing an alkyne bond offers a rigid, linear substituent with minimal steric impact but notable electronic effects).



    #similar ligands, but fail for atom mapping
    # 'ZEK' : 'CC(=O)Nc1cccc(c1)C(=O)NC',
    # 'UQM' : 'CC(=O)Nc1ccc(cc1)CC(=O)N',


    # #SIMILAR LIGANDS FROM PDB QUERY BASED ON SMILE
    # #no'IJX' : 'CC(=O)Nc1ccc(cc1)Br',
    # 'TYL' : 'CC(=O)Nc1ccc(cc1)O',
    # 'B1A' : 'CC(=O)Nc1ccc(cc1)Nc2ccccc2',
    # 'T9V' : 'CC(=O)Nc1ccc(cc1)OC',
    # 'L3V' : 'CC(=O)Nc1ccc(cc1)CO',
    # 'S0P' : 'CC(=O)Nc1ccc(cc1)C#N',
    # 'VYZ' : 'CC(=O)Nc1ccc(cc1)N(=O)=O',
    # 'N4E' : 'CCOc1ccc(cc1)NC(=O)C',
    # 'TYZ' : 'CC(=O)Nc1ccc(cc1)C(=O)O',
    # #(Similar Stereospecific and Stereoisomer)
    # '20N' : 'CC(=O)Nc1ccc(cc1)NC(=O)C2CC2',
}


def create_atom_map(template_mol, target_mol):

    # Identify relevant atoms in the template. This part needs manual identification based on your criteria
    # Example: Find indices of sulfur atoms in CYS residues
    template_atoms = [atom.GetIdx() for atom in template_mol.GetAtoms() if atom.GetSymbol() == 'S']
    
    # # Identify relevant atoms in the target molecule. This also needs manual setup based on the molecule structure
    # # Example: Assuming the relevant atoms are the first and last nitrogen in the WHL molecule
    # target_atoms = [0, target_mol.GetNumAtoms() - 1]  # Placeholder indices
    
    # For the first carbon (CH), which is the second atom in the SMILES string
    # For the last carbon (CK), which is the last atom in the SMILES string before the oxygen
    target_atoms = [1, target_mol.GetNumAtoms() - 2]  # Adjusted indices for CH and CK


    # Create atom map for alignment
    atomMap = list(zip(target_atoms, template_atoms))
    atomMap = dict(atomMap)

    return atomMap



def perturb(mol, conformer_id):
    for i, j, k, l in AllChem.FindAllPathsOfLengthN(mol, 4, useBonds=False, useHs=False):
        if not mol.GetBondBetweenAtoms(j, k).IsInRing():
            value = AllChem.GetDihedralDeg(mol.GetConformer(conformer_id), i, j, k, l)
            AllChem.SetDihedralDeg(mol.GetConformer(conformer_id), i, j, k, l, float(value + random.gauss(0.0, 10.0)))

def align(mol, template_mol, atomMap):
    for conformer in mol.GetConformers():
        rmsd = AllChem.AlignMol(mol, template_mol, conformer.GetId(), 0, list(atomMap.items()))
        conformer.SetDoubleProp('rmsd', rmsd)

def optimize(mol):
    for conformer in tqdm.tqdm(mol.GetConformers()):
        atoms = Atoms(
            numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
            positions=conformer.GetPositions(),
            calculator=calculator,
        )

        coord = atoms.positions
        number_idx = torch.tensor([atom.number for atom in atoms], dtype=torch.long)

        #ADDED
        # Debugging: Print the types and shapes of the data
        print("Atoms object:", type(atoms))
        print("Number of atoms:", len(atoms))
        print("Positions shape:", atoms.positions.shape)
        # If additional tensors are involved in AIMNet calculations, add their print statements here
        print("coord data type:", coord.dtype)
        print("coord shape:", coord.shape)
        print("number_idx data type:", number_idx.dtype)
        print("number_idx shape:", number_idx.shape)


        #ADDED
        try:
            opt = BFGS(atoms, logfile='/dev/null')
            # opt.attach(save, interval=1)
            opt.run(steps=99, fmax=0.05)
        except Exception as e:
            # Print the exception for debugging
            tqdm.tqdm.write(f'Error during optimization: {e}')
            # Fix for the NameError
            conformer_id = conformer.GetId()
            tqdm.tqdm.write(f'FAILED MINIMIZE: {conformer_id}')

        # try:
        #     opt = BFGS(atoms, logfile='/dev/null')
        #     # opt.attach(save, interval=1)
        #     opt.run(steps=99, fmax=0.05)
        # except:
        #     tqdm.tqdm.write(f'FAILED MINIMIZE: {conformer_id}')


        for i, position in enumerate(atoms.get_positions()):
            conformer.SetAtomPosition(i, position)

        connectivity_matrix = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.int8)
        for bond in mol.GetBonds():
            connectivity_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = 1
            connectivity_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = 1

        neighbor_list = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
        neighbor_list.update(atoms)

        if np.array_equal(neighbor_list.get_connectivity_matrix(sparse=False), connectivity_matrix):

            #ADDED
            # Print out the Atoms object for inspection
            print("Atoms object:", atoms)

            # Print out the number of atoms
            print("Number of atoms:", len(atoms))

            # Print out the potential and kinetic energy separately
            #potential_energy = atoms.get_potential_energy()
            #kinetic_energy = atoms.get_kinetic_energy()
            #print("Potential Energy:", potential_energy)
            #print("Kinetic Energy:", kinetic_energy)

            # Get and print the total energy
            total_energy = atoms.get_total_energy()
            print("Total Energy:", total_energy)

            energy = atoms.get_total_energy()[0] * (1/(units.kcal/units.mol))
            
        else:
            energy = 0.0

        conformer.SetDoubleProp('energy', energy)
        tqdm.tqdm.write(f'energy: {energy}')


#ADDED to assist in conformer generation
# Enhanced Conformer Generation Function
# def generate_conformers(mol, num_confs=25, max_attempts=100, coordMap=None):
#     params = AllChem.EmbedParameters()  # Use generic EmbedParameters for broader compatibility
#     params.useExpTorsionAnglePrefs = True
#     params.useBasicKnowledge = True
#     params.maxAttempts = max_attempts
#     params.numThreads = 1  # Adjust based on compatibility; some versions of RDKit might not support numThreads in EmbedParameters directly

#     if coordMap is not None:
#         # Ensure coordMap is a dict with integer keys and Point3D values
#         formattedCoordMap = {int(k): Point3D(*v) for k, v in coordMap.items()}
#         params.coordMap = formattedCoordMap

#     try:
#         ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
#         print(f"Generated {len(ids)} conformers.")
#         return ids
#     except Exception as e:
#         print(f"Conformer generation failed: {e}")
#         return []


name_smiles = list(smiles_dict.items())
random.shuffle(name_smiles)

for name, smiles in name_smiles:
    print(name)

    print('Generating Conformers...')
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    template_mol = Chem.AddHs(Chem.MolFromPDBFile('selected_residues.pdb'))


    #template_mol = Chem.AddHs(Chem.MolFromPDBFile('1h2c_motif_78_93.pdb'))
    #other: template_mol = Chem.MolFromPDBFile("1h2c_motif_78_93.pdb", removeHs=False)

    # atom_map = {
    #     cysteine_sulfur_idx_mol1: linker_atom_idx_template1,
    #     cysteine_sulfur_idx_mol2: linker_atom_idx_template2
    # }

    # Example usage
    atomMap = create_atom_map(template_mol, mol)
    print(f"Atom Map: {atomMap}")

    coordMap = {a1 : template_mol.GetConformer().GetAtomPosition(a2) for a1, a2 in atomMap.items()}
    #coordMap = {a1: template_mol.GetConformer().GetAtomPosition(a2) for a1, a2 in atomMap.items() if a1 < mol.GetNumAtoms()}
    #coordMap = {int(k): Point3D(v.x, v.y, v.z) for k, v in coordMap.items()}


    # Ensure template molecule has a conformer
    if not template_mol.GetNumConformers():
        print("Template molecule has no conformers. Make sure PDB is loaded correctly.")

    # Print detailed atom positions for both template and target molecules
    print("Template Molecule Atom Information:")
    for i, atom in enumerate(template_mol.GetAtoms()):
        pos = template_mol.GetConformer().GetAtomPosition(i)
        print(f"Template Atom Index: {i}, Element: {atom.GetSymbol()}, Position: (X: {pos.x:.3f}, Y: {pos.y:.3f}, Z: {pos.z:.3f})")

    print("\nTarget Molecule Atom Information:")
    for i, atom in enumerate(mol.GetAtoms()):
        print(f"Target Atom Index: {i}, Element: {atom.GetSymbol()}")

    # Validate and print actual coordinates from coordMap
    print("\ncoordMap Content (Actual Coordinates):")
    for target_idx, template_idx in atomMap.items():
        if template_idx < template_mol.GetNumAtoms():
            template_pos = template_mol.GetConformer().GetAtomPosition(template_idx)
            print(f"Target Atom Index: {target_idx}, Mapped to Template Atom {template_idx}, Position: (X: {template_pos.x:.3f}, Y: {template_pos.y:.3f}, Z: {template_pos.z:.3f})")
        else:
            print(f"Error: Template Atom Index {template_idx} is out of range for the template molecule.")

    # Additional check: Ensure coordMap is being used correctly in conformer generation
    # This step is conceptual; adapt it according to how you're using coordMap in your conformer generation logic

    #CHANGED
    #AllChem.EmbedMultipleConfs(mol, numConfs=run_size, coordMap=coordMap, useSmallRingTorsions=True, numThreads=4)

    #AllChem.EmbedMultipleConfs(mol, numConfs=run_size, clearConfs=False, useSmallRingTorsions=True, numThreads=4)


    # if not generate_conformers(mol, num_confs=25, max_attempts=100, coordMap=coordMap):
    #     print(f"Failed to generate conformers for {name}, skipping...")
    #     continue  # Skip to the next molecule if no conformers could be generated




    # #ADDED
    try:

        # Initialize EmbedParameters and configure them as needed
        params = AllChem.EmbedParameters()
        #params.useExpTorsionAnglePrefs = True
        #params.useBasicKnowledge = True
        params.useSmallRingTorsions = True
        #params.maxAttempts = 100
        params.numThreads = 4  # Adjust this based on your system capabilities

        # Apply the coordinate map if provided
        if coordMap:
            # Ensure the coordMap uses integer keys with Point3D values
            formattedCoordMap = {int(k): Point3D(v.x, v.y, v.z) for k, v in coordMap.items()}
            params.coordMap = formattedCoordMap

            # Now, use these parameters in the AllChem.EmbedMultipleConfs call
            #numConfs = 25  # Desired number of conformers
            ids = AllChem.EmbedMultipleConfs(mol, numConfs=run_size, params=params)

            print(f"Generated {len(ids)} conformers.")
        #use params
        params.coordMap = None
        params.clearConfs=False
        AllChem.EmbedMultipleConfs(mol, numConfs=run_size, params=params)


        # Try to generate multiple conformations: One with Coordmap and one without
        #AllChem.EmbedMultipleConfs(mol, numConfs=run_size, coordMap=coordMap, useSmallRingTorsions=True, numThreads=4)
        #AllChem.EmbedMultipleConfs(mol, numConfs=run_size, clearConfs=False, useSmallRingTorsions=True, numThreads=4)



    except Exception as e:
        # Handle the exception
        print(f'Error during conformer generation: {e}')
        continue  # Skip to the next molecule


    #ADDED
    # Collect all conformers in a list
    conformer_list = list(mol.GetConformers())

    print('conformer list', conformer_list)

    # Iterate over the list of conformers
    for conformer in conformer_list:
        new_conformer_id = mol.AddConformer(conformer, assignId=True)
        perturb(mol, new_conformer_id)


    # for conformer in mol.GetConformers():
    #     new_conformer_id = mol.AddConformer(conformer, assignId=True)
    #     perturb(mol, new_conformer_id)

    optimize(mol)
    align(mol, template_mol, atomMap)

    energies = np.array([conformer.GetDoubleProp('energy') for conformer in mol.GetConformers()])
    rmsds = np.array([conformer.GetDoubleProp('rmsd') for conformer in mol.GetConformers()])
    xyzs = np.array([conformer.GetPositions() for conformer in mol.GetConformers()])


    #ADDED AT END
    if os.path.exists(f'{hdf5_directory}/{name}.hdf5'):
        with h5py.File(f'{hdf5_directory}/{name}.hdf5', 'r') as hdf5_file:
            energies = np.concatenate([hdf5_file['energies'], energies])
            rmsds = np.concatenate([hdf5_file['rmsds'], rmsds])
            xyzs = np.concatenate([hdf5_file['xyzs'], xyzs])

    with h5py.File(f'{hdf5_directory}/{name}.hdf5', 'w') as hdf5_file:
        hdf5_file.create_dataset('energies', data=energies)
        hdf5_file.create_dataset('rmsds', data=rmsds)
        hdf5_file.create_dataset('xyzs', data=xyzs)

    Atoms(
        numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
        positions=xyzs[np.argmin(energies)],
    ).write(f'{pdb_directory}/{name}.pdb')

    sns.set_context('talk', font_scale=1.0)
    sns.set_style('ticks')
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(x=rmsds, y=energies - np.min(energies), color='steelblue', ax=ax)
    ax.set_title(name, fontweight='bold')
    ax.set_ylabel('Delta Energy (kcal/mol)', fontweight='bold')
    ax.set_ylim(-1.0, 26.0)
    ax.set_xlabel('RMSD to Template', fontweight='bold')
    ax.set_xlim(-0.1, 6.1)
    plt.tight_layout()
    f.savefig(f'{figures_directory}/{name}.png', dpi=300)
    plt.close(f)
