import numpy as np
import h5py
import os
import seaborn as sns
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw



# Simplified access to the scratch directory
scratch_directory = os.path.expanduser('~/scratch/okanders/b-hairpin-runs')
date_for_runs = "04-05-2024"

# Construct the parent directory path for this specific date
parent_directory = os.path.join(scratch_directory, date_for_runs)

# Example run name - replace or modify as needed
run_name = "run_size200_mixed_aromatics_staples_gpu_alphahelix_aimnet2"
# Full path to the run-specific directory
run_directory = os.path.join(parent_directory, run_name)


smiles_dict = {
    'WHL' : 'CC(=O)Nc1ccc(cc1)NC(=O)C',

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
    'WHL-Br1': 'CC(=O)Nc1ccc(cc1Br)NC(=O)C', # (Bromination can offer different electronic effects compared to chlorination).

    #Introduction of Substituents with Minimal Electronic Effects:
    'WHL-OCH3': 'CC(=O)Nc1ccc(cc1OC)NC(=O)C', # Methoxyl Group on Aromatic Ring: (Introduces a methoxyl group for slight polarity and electronic effects).

    #Modification of Aromatic Ring Saturation:

    #no 'WHL-Hyd': 'CC(=O)Nc1c(c(cc1)H)NC(=O)C', # Partial Hydrogenation: (Adding hydrogen to create a dihydrobenzene structure for subtle changes in aromaticity).

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

    'WHL-Nitration' : 'CC(=O)Nc1ccc(cc1[N+](=O)[O-])NC(=O)C',
    'WHL-Ethylation' : 'CC(=O)Nc1ccc(cc1CC)NC(=O)C',

    #1. **Variation in Aromatic Substitution**:
    'WHL_PyridineRing' : 'CC(=O)Nc1ccncc1NC(=O)C', #(Substitution with a pyridine ring)


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


# Extract the names (keys) from the SMILES dictionary
names = list(smiles_dict)

# Set the visualization context and style
sns.set_context('talk', font_scale=1.0)
sns.set_style('ticks')

# Create a figure for plotting
f, ax = plt.subplots(1, 1, figsize=(10, 4))

# Load energy and RMSD data for each compound
energies = [h5py.File(f'{run_directory}/_hdf5s/{name}.hdf5', 'r')['energies'][:] for name in names]
delta_energies = [energy - np.min(energy) for energy in energies]
rmsds = [h5py.File(f'{run_directory}/_hdf5s/{name}.hdf5', 'r')['rmsds'][:] for name in names]

# Print energy and RMSD data for verification
print("Energies:", energies)
print("Delta Energies:", delta_energies)
print("RMSDs:", rmsds)

# Calculate P_near values
pnears = []
for rmsd, delta_energy in zip(rmsds, delta_energies):
    numerator = np.sum(np.exp(-(rmsd/2.5)**2) * np.exp(-delta_energy/0.59))
    denominator = np.sum(np.exp(-delta_energy/0.59))
    
    # Add print statements to debug individual components of the P_near calculation
    print("Numerator:", numerator)
    print("Denominator:", denominator)
    
    if denominator == 0:  # Check to prevent division by zero
        print("Warning: Denominator is zero. Setting P_near to 0.")
        pnears.append(0)
    else:
        pnear = numerator / denominator
        pnears.append(pnear)
        print("P_near:", pnear)

# Determine the shape of the heatmap based on the number of items
num_items = len(names)
num_rows = int(np.sqrt(num_items))
num_columns = np.ceil(num_items / num_rows).astype(int)

# Plot the heatmap
sns.heatmap(np.array(pnears).reshape(num_rows, num_columns), vmin=0.0, vmax=1.0, 
            annot=np.array([f'{name}\n{"%.2f" % pnear}' for name, pnear in zip(names, pnears)]).reshape(num_rows, num_columns),
            fmt='', xticklabels=False, yticklabels=False, cmap='Blues', ax=ax,
            annot_kws={"size":10})
ax.collections[0].colorbar.set_label(r'$P_{Near}$', labelpad=32, rotation=270)

plt.tight_layout()
f.savefig(f'{run_directory}/analysis.png', dpi=300)
plt.close(f)


options = Draw.MolDrawOptions()
options.legendFontSize = 128

Draw.MolsToGridImage([Chem.MolFromSmiles(value) for value in smiles_dict.values()], molsPerRow=5, subImgSize=(500,500), legends=names, drawOptions=options).save(f'{run_directory}/crosslinkers.png')
