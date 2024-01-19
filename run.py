import random
import os

import tqdm

import numpy as np
import h5py

import seaborn as sns
import matplotlib.pyplot as plt

import sys


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


#OLD
# from dftd4.calculators import D4_model

if torch.cuda.is_available():
    model = load_AIMNetSMD_ens().cuda()
else:
    model = load_AIMNetSMD_ens()


#NEW
# Assuming 'model' is your AIMNet model already defined
# and AIMNetCalculator is an ASE-compatible calculator
aimnet_calculator = AIMNetCalculator(model)

# Create the DFTD4 calculator with the specified method
dftd4_calculator = DFTD4(method='wB97x')

# Combine the DFTD4 and AIMNet calculators
calculator = SumCalculator([dftd4_calculator, aimnet_calculator])



#OLD
#calculator = D4_model(xc='wB97x', calc=AIMNetCalculator(model))

run_size = 25




#suggested
# from ase.build import molecule
# from ase.calculators.mixing import SumCalculator
# from ase.calculators.nwchem import NWChem
# from dftd4.ase import DFTD4
# atoms = molecule('H2O')
# calculator = SumCalculator([DFTD4(method="wB97x"), NWChem(xc="wB97x")])


smiles_dict = {
    'COS' : 'CC(=O)N[C@H](C(=O)NC)CSCc1c(cccc1)CS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'CMS' : 'CC(=O)N[C@H](C(=O)NC)CSCc1cc(ccc1)CS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'CPS' : 'CC(=O)N[C@H](C(=O)NC)CSCc1ccc(cc1)CS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'CYS' : 'CC(=O)N[C@H](C(=O)NC)CSCc1nc(ccc1)CS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'CSS' : 'CC(=O)N[C@H](C(=O)NC)CSS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'C1S' : 'CC(=O)N[C@H](C(=O)NC)CSCS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'C2S' : 'CC(=O)N[C@H](C(=O)NC)CSCCS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'C3S' : 'CC(=O)N[C@H](C(=O)NC)CSCCCS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'C4S' : 'CC(=O)N[C@H](C(=O)NC)CSCCCCS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'CAS' : 'CC(=O)N[C@H](C(=O)NC)CSCC=CCS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',

    'COR' : 'CC(=O)N[C@H](C(=O)NC)CSCc1c(cccc1)CS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'CMR' : 'CC(=O)N[C@H](C(=O)NC)CSCc1cc(ccc1)CS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'CPR' : 'CC(=O)N[C@H](C(=O)NC)CSCc1ccc(cc1)CS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'CYR' : 'CC(=O)N[C@H](C(=O)NC)CSCc1nc(ccc1)CS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'CSR' : 'CC(=O)N[C@H](C(=O)NC)CSS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'C1R' : 'CC(=O)N[C@H](C(=O)NC)CSCS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'C2R' : 'CC(=O)N[C@H](C(=O)NC)CSCCS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'C3R' : 'CC(=O)N[C@H](C(=O)NC)CSCCCS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'C4R' : 'CC(=O)N[C@H](C(=O)NC)CSCCCCS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'CAR' : 'CC(=O)N[C@H](C(=O)NC)CSCC=CCS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',

    'HOS' : 'CC(=O)N[C@H](C(=O)NC)CCSCc1c(cccc1)CS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'HMS' : 'CC(=O)N[C@H](C(=O)NC)CCSCc1cc(ccc1)CS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'HPS' : 'CC(=O)N[C@H](C(=O)NC)CCSCc1ccc(cc1)CS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'HYS' : 'CC(=O)N[C@H](C(=O)NC)CCSCc1nc(ccc1)CS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'HSS' : 'CC(=O)N[C@H](C(=O)NC)CCSS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'H1S' : 'CC(=O)N[C@H](C(=O)NC)CCSCS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'H2S' : 'CC(=O)N[C@H](C(=O)NC)CCSCCS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'H3S' : 'CC(=O)N[C@H](C(=O)NC)CCSCCCS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'H4S' : 'CC(=O)N[C@H](C(=O)NC)CCSCCCCS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'HAS' : 'CC(=O)N[C@H](C(=O)NC)CCSCC=CCS[C@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',

    'HOR' : 'CC(=O)N[C@H](C(=O)NC)CCSCc1c(cccc1)CS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'HMR' : 'CC(=O)N[C@H](C(=O)NC)CCSCc1cc(ccc1)CS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'HPR' : 'CC(=O)N[C@H](C(=O)NC)CCSCc1ccc(cc1)CS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'HYR' : 'CC(=O)N[C@H](C(=O)NC)CCSCc1nc(ccc1)CS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'HSR' : 'CC(=O)N[C@H](C(=O)NC)CCSS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'H1R' : 'CC(=O)N[C@H](C(=O)NC)CCSCS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'H2R' : 'CC(=O)N[C@H](C(=O)NC)CCSCCS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'H3R' : 'CC(=O)N[C@H](C(=O)NC)CCSCCCS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'H4R' : 'CC(=O)N[C@H](C(=O)NC)CCSCCCCS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
    'HAR' : 'CC(=O)N[C@H](C(=O)NC)CCSCC=CCS[C@@H](C1)C[C@@H](C(=O)NC)N1C(=O)C',
}

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

        try:
            opt = BFGS(atoms, logfile='/dev/null')
            # opt.attach(save, interval=1)
            opt.run(steps=99, fmax=0.05)
        except:
            tqdm.tqdm.write(f'FAILED MINIMIZE: {conformer_id}')

        for i, position in enumerate(atoms.get_positions()):
            conformer.SetAtomPosition(i, position)

        connectivity_matrix = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.int8)
        for bond in mol.GetBonds():
            connectivity_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = 1
            connectivity_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = 1

        neighbor_list = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
        neighbor_list.update(atoms)

        if np.array_equal(neighbor_list.get_connectivity_matrix(sparse=False), connectivity_matrix):
            energy = atoms.get_total_energy()[0] * (1/(units.kcal/units.mol))
        else:
            energy = 0.0

        conformer.SetDoubleProp('energy', energy)
        tqdm.tqdm.write(f'energy: {energy}')

name_smiles = list(smiles_dict.items())
random.shuffle(name_smiles)

for name, smiles in name_smiles:
    print(name)

    print('Generating Conformers...')
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    template_mol = Chem.AddHs(Chem.MolFromPDBFile('1h2c_motif_78_93.pdb'))

    atomMap = {
        0 : 0,
        1 : 1,
        2 : 2,
        3 : 3,
        4 : 4,
        5 : 5,
        6 : 6,
        7 : 8,
        8 : 9,

        mol.GetNumHeavyAtoms() - 1 : 10,
        mol.GetNumHeavyAtoms() - 3 : 11,
        mol.GetNumHeavyAtoms() - 2 : 12,
        mol.GetNumHeavyAtoms() - 4 : 13,
        mol.GetNumHeavyAtoms() - 9 : 14,
        mol.GetNumHeavyAtoms() - 8 : 15,
        mol.GetNumHeavyAtoms() - 7 : 16,
        mol.GetNumHeavyAtoms() - 6 : 20,
        mol.GetNumHeavyAtoms() - 5 : 21,
    }

    coordMap = {a1 : template_mol.GetConformer().GetAtomPosition(a2) for a1, a2 in atomMap.items()}

    AllChem.EmbedMultipleConfs(mol, numConfs=run_size, coordMap=coordMap, useSmallRingTorsions=True, numThreads=4)

    AllChem.EmbedMultipleConfs(mol, numConfs=run_size, clearConfs=False, useSmallRingTorsions=True, numThreads=4)

    for conformer in mol.GetConformers():
        new_conformer_id = mol.AddConformer(conformer, assignId=True)
        perturb(mol, new_conformer_id)

    optimize(mol)
    align(mol, template_mol, atomMap)

    energies = np.array([conformer.GetDoubleProp('energy') for conformer in mol.GetConformers()])
    rmsds = np.array([conformer.GetDoubleProp('rmsd') for conformer in mol.GetConformers()])
    xyzs = np.array([conformer.GetPositions() for conformer in mol.GetConformers()])

    if os.path.exists(f'_hdf5s/{name}.hdf5'):
        with h5py.File(f'_hdf5s/{name}.hdf5', 'r') as hdf5_file:
            energies = np.concatenate([hdf5_file['energies'], energies])
            rmsds = np.concatenate([hdf5_file['rmsds'], rmsds])
            xyzs = np.concatenate([hdf5_file['xyzs'], xyzs])

    with h5py.File(f'_hdf5s/{name}.hdf5', 'w') as hdf5_file:
        hdf5_file.create_dataset('energies', data=energies)
        hdf5_file.create_dataset('rmsds', data=rmsds)
        hdf5_file.create_dataset('xyzs', data=xyzs)

    Atoms(
        numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
        positions=xyzs[np.argmin(energies)],
    ).write(f'_pdbs/{name}.pdb')

    sns.set_context('talk', font_scale=1.0)
    sns.set_style('ticks')
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(rmsds, energies - np.min(energies), color='steelblue', ax=ax)
    ax.set_title(name, fontweight='bold')
    ax.set_ylabel('Delta Energy (kcal/mol)', fontweight='bold')
    ax.set_ylim(-1.0, 26.0)
    ax.set_xlabel('RMSD to Template', fontweight='bold')
    ax.set_xlim(-0.1, 6.1)
    plt.tight_layout()
    f.savefig(f'_figures/{name}.png', dpi=300)
    plt.close(f)
