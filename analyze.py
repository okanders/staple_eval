import numpy as np
import h5py

import seaborn as sns
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw

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

names = list(smiles_dict)

sns.set_context('talk', font_scale=1.0)
sns.set_style('ticks')
f, ax = plt.subplots(1, 1, figsize=(10, 4))

energies = [h5py.File(f'_hdf5s/{name}.hdf5', 'r')['energies'][:] for name in names]
delta_energies = [energy - np.min(energy) for energy in energies]

rmsds = [h5py.File(f'_hdf5s/{name}.hdf5', 'r')['rmsds'][:] for name in names]

pnears = [
    np.sum(np.exp(-(rmsd/0.5)**2) * np.exp(-delta_energy/0.59)) /
    np.sum(np.exp(-delta_energy/0.59))
    for rmsd, delta_energy in zip(rmsds, delta_energies)
]

sns.heatmap(np.array(pnears).reshape(4,10), vmin=0.0, vmax=1.0, annot=np.array([f'{name}\n{"%.2f" % pnear}' for name, pnear in zip(names, pnears)]).reshape(4,10), fmt='', xticklabels=False, yticklabels=False, cmap='Blues', ax=ax)
ax.collections[0].colorbar.set_label(r'$P_{Near}$', labelpad=32, rotation=0)

plt.tight_layout()
f.savefig(f'analysis.png', dpi=300)
plt.close(f)

options = Draw.MolDrawOptions()
options.legendFontSize = 128

Draw.MolsToGridImage([Chem.MolFromSmiles(value) for value in smiles_dict.values()], molsPerRow=5, subImgSize=(500,500), legends=names, drawOptions=options).save('crosslinkers.png')
