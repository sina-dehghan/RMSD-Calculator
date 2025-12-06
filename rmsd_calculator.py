from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
import pandas as pd
import sys

def explain_rmsd(): 
    # Formula: RMSD = sqrt( sum((atom_i_ref - atom_i_target)^2) / N )
    print(explain_rmsd.__doc__)

def load_molecule(file_path, mol_index=0):
    # Load a molecule from SDF file.
    
    # SDMolSupplier reads SDF files (can contain multiple molecules)
    try:
        supplier = Chem.SDMolSupplier(file_path, removeHs=False)

        # Getting the specific molecule
        mol = supplier[mol_index]

        if mol is None:
            raise ValueError(f"Could not read the molecule from {file_path}")
        
        print(f"✓ Loaded molecule from {file_path}")
        print(f" - Number of atoms: {mol.GetNumAtoms()}")
        print(f" -Molecular Formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")

        return mol
    except Exception as e:
        print(f"  ❌ Error loading {file_path}: {str(e)}")
        raise



def load_all_molecules(file_path):
    # Load ALL the molecules from SDF files.
    try:
        supplier = Chem.SDMolSupplier(file_path, removeHs=False)

        molecules = []

        for i, mol in enumerate(supplier):
            if mol is not None:
                molecules.append(mol)
        
        print(f"✓ Loaded {len(molecules)} molecules from {file_path}")
        return molecules
        
    except Exception as e:
        print(f"  ❌ Error loading {file_path}: {str(e)}")
        raise


def calculate_rmsd(reference_mol, target_mol, align=True):
    # Calculate RMSD between reference and target molecule.

    # Check if molecules have the same number of atoms
    if reference_mol.GetNumAtoms() != target_mol.GetNumAtoms():
        print(f"⚠ Warning: molecules have different atom counts.")
        print(f"  Reference: {reference_mol.GetNumAtoms()} atoms.")
        print(f"  Target: {target_mol.GetNumAtoms()} atoms.")

    if align:
        # AlignMol: Rotates and translates target molecule to best match reference
        rmsd = rdMolAlign.AlignMol(target_mol, reference_mol)
    else:
        # CalcRMS: Just calculates RMSD without alignment (not recommended)
        rmsd = rdMolAlign.CalcRMS(reference_mol, target_mol)

    return rmsd


# Calculate RMSD for multiple structures against one reference.
def calculate_rmsd_batch(reference_mol, target_molecules, labels=None):
    results = []

    for i, target_mol in enumerate(target_molecules):
        # Create a lable if not provided.
        if labels and i < len(labels):
            label = labels[i]
        else:
            label = f"structure_{i+1}"
        
        try:
            # Calculate RMSD
            rmsd = calculate_rmsd(reference_mol, target_mol, align=True)

            results.append({
                'Structure_ID': i+1,
                'Label': label,
                'RMSD (Å)': round(rmsd, 3)
            })

            print(f"{label}: RMSD = {rmsd:.3f} Å")

        except Exception as e:
            print(f" ⚠ Calculating RMSD for {label}: {str(e)}")
            results.append({
                'Structure_ID': i+1,
                'Label': label,
                'RMSD (Å)': None
            })
    
    # Creating excel (Data Frame)
    results_df = pd.DataFrame(results)

    return results_df

def main():
    # Main function of running the RMSD calculations
    print("="*70)
    print("RMSD CALCULATOR FOR MOLECULAR STRUCTURES")
    print("="*70)
    print()

    # Step 1: Getting the files from user
    print("Please provide the following files:")
    print("-"*70)

    # X-ray reference structure
    xray_file = input("Enter path to X-ray ligand SDF file: ").strip()

    # Confomers file
    conformer_file = input("Enter path to conformers SDF file: ").strip()

    # Docking Pose
    docking_file = input(" Enter path to docking poses SDF file: ").strip()

    # Output file
    output_file = input("Enter output CSV filename (e.g., rmsd_results.csv): ").strip()

    if not output_file:
        output_file = "rsmd_results.csv"

    print()
    print("="*70)
    print("LOADING STRUCTURES...")
    print("="*70)
    print()


    try:
        # Step 2: Load the X-ray reference structure
        print("1. Loading X-ray reference structure...")
        reference_mol = load_molecule(xray_file, mol_index=0)
        print()

        # Step 3: Load conformers
        print("2. Loading conformers...")
        conformers = load_all_molecules(conformer_file)
        conformer_labels = [f"Conformer_{i+1}" for i in range(len(conformers))]
        print()

        # Step 4: Load docking poses
        print("3. Loading docking poses...")
        docking_poses = load_all_molecules(docking_file)
        docking_labels = [f"DockingPose_{i+1}" for i in range(len(docking_poses))]
        print()

        print("="*70)
        print("CALCULATING RMSD VALUES...")
        print("="*70)
        print()

        #  Step 5: Calculate RMSD for conformers
        print("Calculating RMSD for CONFORMERS:")
        print("-" * 70)
        conformers_results = calculate_rmsd_batch(
            reference_mol,
            conformers,
            labels=conformer_labels
        )
        conformers_results['Type']= 'Conformer'
        print()

        # Step 6: Calculate RMSD for docking poses
        print("Calculating RMSD for DOCKING POSES:")
        print("-" * 70)
        docking_results = calculate_rmsd_batch(
            reference_mol,
            docking_poses,
            labels=docking_labels
        )
        docking_results['Type']= 'Docking_Pose'

        # Step 7: Combine results
        all_results = pd.concat([conformers_results, docking_results], ignore_index=True)

        # Reorder Columns
        all_results = all_results[['Type', 'Structure_ID', 'Label', 'RMSD (Å)']]

        # Step 8: Save results
        all_results.to_csv(output_file, index=False)


        print("="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print()
        print(all_results.to_string(index=False))
        print()

        # Statistics

        # Remove None values for statistics
        conformers_rmsd = conformers_results['RMSD (Å)'].dropna()
        docking_rmsd = docking_results['RMSD (Å)'].dropna()

        print("="*70)
        print("STATISTICS")
        print("="*70)
        print()

        if len(conformers_rmsd) > 0:
            print("CONFORMERS:")
            print(f"  - Total: {len(conformers_results)}")
            print(f"  - Mean RMSD: {conformers_rmsd.mean():.3f} Å")
            print(f"  - Min RMSD: {conformers_rmsd.min():.3f} Å")
            print(f"  - Max RMSD: {conformers_rmsd.max():.3f} Å")
            print()

        if len(docking_rmsd) > 0:
            print("DOCKING POSES:")
            print(f"  - Total: {len(docking_results)}")
            print(f"  - Mean RMSD: {docking_rmsd.mean():.3f} Å")
            print(f"  - Min RMSD: {docking_rmsd.min():.3f} Å")
            print(f"  - Max RMSD: {docking_rmsd.max():.3f} Å")
            print()
        
        print(f"✓ Results saved to: {output_file}")
        print()
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()