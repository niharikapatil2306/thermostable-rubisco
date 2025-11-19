from Bio.PDB import PDBParser, DSSP, NeighborSearch
import pandas as pd
import os

def extract_pdb_data_complete(pdb_file_path):
    """Extract all structural data from a PDB file"""
    try:
        # Parse PDB file
        parser = PDBParser(QUIET=True)
        protein_id = os.path.basename(pdb_file_path).replace('.pdb', '')
        structure = parser.get_structure(protein_id, pdb_file_path)
        model = structure[0]
        chain = model['A']  # Assuming chain A
        
        # Data lists to store comprehensive information
        sequence = ""
        all_atom_data = []
        confidence_scores = []
        
        # Define a dictionary for amino acid name conversion
        aa_dict = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        # Iterate through residues to get the full sequence and atom data
        for residue in chain:
            res_name = residue.get_resname()
            if res_name in aa_dict:
                sequence += aa_dict[res_name]
            else:
                sequence += 'X'
            for atom in residue:
                all_atom_data.append({
                    'residue_name': res_name,
                    'residue_number': residue.get_id()[1],
                    'atom_name': atom.get_name(),
                    'coordinates': tuple(atom.get_coord()),
                    'confidence_score': atom.get_bfactor()
                })
                if atom.get_name() == 'CA':
                    confidence_scores.append(atom.get_bfactor())
        
        # Create a confidence mask based on C-alpha scores
        confidence_mask = [1 if score > 90 else 0 for score in confidence_scores]
        
        # Run DSSP for secondary structure and angles
        dssp = DSSP(model, pdb_file_path, dssp='mkdssp')
        
        secondary_structure = ""
        phi_angles = []
        psi_angles = []
        
        for key in dssp.keys():
            secondary_structure += dssp[key][2]
            phi_angles.append(dssp[key][4])
            psi_angles.append(dssp[key][5])
        
        # Recalculate neighbor count using NeighborSearch on all atoms
        atom_list = [atom for atom in chain.get_atoms()]
        neighbor_search = NeighborSearch(atom_list)
        
        all_neighbor_counts = []
        for atom in atom_list:
            neighbors = neighbor_search.search(atom.get_coord(), 8.0)
            all_neighbor_counts.append(len(neighbors) - 1)
        
        return {
            'protein_id': protein_id,
            'sequence': sequence,
            'all_atom_data': all_atom_data,
            'confidence_mask': confidence_mask,
            'secondary_structure': secondary_structure,
            'phi_angles': phi_angles,
            'psi_angles': psi_angles,
            'all_neighbor_counts': all_neighbor_counts
        }
    
    except Exception as e:
        print(f"Error processing {pdb_file_path}: {e}")
        return None

# Process all PDB files in the pdb_data folder
pdb_folder = "/app/new_pdb"
output_folder = "/app/new_pdb"
all_data = []

print("Processing PDB files...")
for filename in os.listdir(pdb_folder):
    if filename.endswith(".pdb"):
        pdb_path = os.path.join(pdb_folder, filename)
        print(f"Processing: {filename}")
        
        data = extract_pdb_data_complete(pdb_path)
        if data:
            all_data.append(data)
            print(f"  ✓ Success: {len(data['sequence'])} residues")
        else:
            print(f"  ✗ Failed")

print(f"\nTotal proteins processed: {len(all_data)}")

# Convert to DataFrame and save to CSV
if all_data:
    # Convert lists to strings for CSV storage
    for protein_data in all_data:
        protein_data['all_atom_data'] = str(protein_data['all_atom_data'])
        protein_data['confidence_mask'] = str(protein_data['confidence_mask'])
        protein_data['phi_angles'] = str(protein_data['phi_angles'])
        protein_data['psi_angles'] = str(protein_data['psi_angles'])
        protein_data['all_neighbor_counts'] = str(protein_data['all_neighbor_counts'])
    
    df = pd.DataFrame(all_data)
    csv_filename = os.path.join(output_folder, "final_data.csv")
    df.to_csv(csv_filename, index=False)
    print(f"\nDataset saved to: {csv_filename}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
else:
    print("No data to save!")