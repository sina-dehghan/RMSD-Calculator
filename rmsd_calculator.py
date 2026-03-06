from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolDescriptors
import pandas as pd
import sys
import copy
""" rdMolAlign.AlignMol physically moves the target 
 molecule's 3D coordinates in memory to optimally superimpose
 it onto the reference. This mutation happens in-place on 
 whatever object you pass in."""


def explain_rmsd(): 
    """
    Formula: RMSD = sqrt( sum((atom_i_ref - atom_i_target)^2) / N )

    Interpretation:
      - RMSD < 1.0 Å  → Excellent agreement
      - RMSD < 2.0 Å  → Good agreement (acceptable docking pose)
      - RMSD > 2.0 Å  → Poor agreement (different binding mode)
    """
    print(explain_rmsd.__doc__)

def load_molecule(file_path, mol_index=0, removeHs=True):
    # Load a molecule from SDF file.
    # (file_path: str, mol_index: int, removeHs: bool)
    # SDMolSupplier reads SDF files (can contain multiple molecules)
    try:
        # removeHs=False at load time to preseve all atoms initially
        supplier = Chem.SDMolSupplier(file_path, removeHs=False)

        if supplier is None:
            raise FileNotFoundError(f"Could not open file: {file_path}")
        
        # Getting the specific molecule
        mol = supplier[mol_index]

        if mol is None:
            raise ValueError(f"Could not parse molecule at index {mol_index} from {file_path}")

        if removeHs:
            mol = Chem.RemoveHs(mol)

        print(f"  ✓ Loaded molecule from {file_path}")
        print(f"    - Number of atoms: {mol.GetNumAtoms()} ({'heavy atoms only' if removeHs else 'including H'})")
        print(f"    - Molecular Formula: {rdMolDescriptors.CalcMolFormula(mol)}")

        return mol
    

    except Exception as e:
        print(f"  ❌ Error loading {file_path}: {str(e)}")
        raise



def load_all_molecules(file_path, removeHs=True):
    # Load ALL the molecules from SDF files.
    try:
        supplier = Chem.SDMolSupplier(file_path, removeHs=False)

        if supplier is None:
            raise FileNotFoundError(f"Could not open file: {file_path}")
        
        molecules = []
        skipped = 0

        for i, mol in enumerate(supplier):
            if mol is not None:
                if removeHs:
                    mol = Chem.RemoveHs(mol)
                molecules.append(mol)
            else:
                skipped += 1
                print(f"  ⚠ Skipped unreadable molecule at index {i}")

        print(f"  ✓ Loaded {len(molecules)} molecules from {file_path}")
        if skipped > 0:
            print(f"  ⚠ Skipped {skipped} unreadable entries")

        return molecules
        
    except Exception as e:
        print(f"  ❌ Error loading {file_path}: {str(e)}")
        raise

def get_smarts_atoms_map(reference_mol, target_mol, smarts_pattern):
    # Building an atom map from a SMARTS pattern for selective RMSD.
    # (reference_mol: rdkit.Chem.Mol,
    # target_mol: rdkit.Chem.Mol,
    # smarts_pattern: str)

    # Returns: list of (int, int):(target_atom_idx, ref_atom_idx) and list of int: reference atom

    query = Chem.MolFromSmarts(smarts_pattern)
    if query is None:
        raise ValueError(f"Invalid SMARTS pattern: '{smarts_pattern}'")
    
    # Get ALL matches in each molecules
    ref_matches = reference_mol.GetSubstructMatches(query)
    tgt_matches = target_mol.GetSubstructMatches(query)

    if not ref_matches:
        raise ValueError(
            f"SMARTS pattern '{smarts_pattern}' matched 0 atoms in the reference molecule."
        )
    if not tgt_matches:
        raise ValueError(
            f"SMARTS pattern '{smarts_pattern}' matched 0 atoms in the target molecule."
        )

    # Collect unique atom indices from all matches
    ref_atom_set = sorted(set(idx for match in ref_matches for idx in match))
    tgt_atom_set = sorted(set(idx for match in tgt_matches for idx in match))

    if len(ref_atom_set) != len(tgt_atom_set):
        raise ValueError(
            f"SMARTS pattern '{smarts_pattern}' matched {len(ref_atom_set)} atoms "
            f"in reference but {len(tgt_atom_set)} in target. "
            f"Both molecules must have the same number of matching atoms."
        )
    
    # Build atom map: list of (target_idx, ref_idx) tuples
    atom_map = list(zip(tgt_atom_set, ref_atom_set))

    return atom_map, ref_atom_set
 

def calculate_rmsd(reference_mol, target_mol, align=True, atom_map=None):
    # Calculate RMSD between reference and target molecule.
    # (reference_mol:rdkit.Chem.Mol, target_mol:rdkit.Chem.Mol, and align: bool, atom_map: list of (int, int))

    if atom_map is None:
        # Full-molecule mode: validate atom counts
        n_ref = reference_mol.GetNumAtoms()
        n_tgt = target_mol.GetNumAtoms()

        if n_ref != n_tgt:
            raise ValueError(
                f"Atom count mismatch: reference has {n_ref} atoms, "
                f"target has {n_tgt} atoms. Cannot compute RMSD without "
                f"atom mapping. Ensure both molecules represent the same "
                f"compound, or use a SMARTS pattern for selective RMSD."
            )

    if align:
        # Work on a copy so we never mutate the caller's molecules
        target_copy = copy.deepcopy(target_mol)
        # AlignMol returns RMSD after optimal rigid-body alignment
        # atomMap restricts which atoms are used for alignment + RMSD
        rmsd = rdMolAlign.AlignMol(
            target_copy, reference_mol,
            atomMap=atom_map if atom_map else []
        )
    else:
        # CalcRMS computes the RMSD after optimal alignment but does not
        # modify the molecule coordinates - good for read-only comparison
        rmsd = rdMolAlign.CalcRMS(
            reference_mol, target_mol,
            map=[atom_map] if atom_map else []
        )

    return rmsd


# Calculate RMSD for multiple structures against one reference.
def calculate_rmsd_batch(reference_mol, target_molecules, labels=None,
                         smarts_pattern=None):
    #pd.DataFrame columns: Structure_ID, Label, RMSD (Å), [Atoms_Used].

    results = []

    for i, target_mol in enumerate(target_molecules):
        # Create a lable if not provided.
        if labels and i < len(labels):
            label = labels[i]
        else:
            label = f"structure_{i+1}"
        
        try:
            atoms_map = None
            n_atoms_used = reference_mol.GetNumAtoms()

            if smarts_pattern:
                atoms_map, ref_indices = get_smarts_atoms_map(
                    reference_mol, target_mol, smarts_pattern
                )
                n_atoms_used = len(atoms_map)

            rmsd = calculate_rmsd(reference_mol, target_mol, align=True,
                                   atom_map=atoms_map)

            results.append({
                'Structure_ID': i+1,
                'Label': label,
                'RMSD (Å)': round(rmsd, 3),
                'Atoms_Used': n_atoms_used
            })
            print(f"    {label}: RMSD = {rmsd:.3f} Å  ({n_atoms_used} atoms)")

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

def print_statistics(name, rmsd_series, results_df):
    # Print summary statistics for a set of RRMSD values.

    if len(rmsd_series) == 0:  
        return
    print(f"  {name}:")
    print(f"    - Total structures: {len(results_df)}")
    print(f"    - Successfully computed: {len(rmsd_series)}")
    print(f"    - Mean RMSD: {rmsd_series.mean():.3f} Å")
    print(f"    - Min  RMSD: {rmsd_series.min():.3f} Å  (best match)")
    print(f"    - Max  RMSD: {rmsd_series.max():.3f} Å")
    if len(rmsd_series) > 1:
        print(f"    - Std  RMSD: {rmsd_series.std():.3f} Å")
    good_poses = (rmsd_series < 2.0).sum()
    print(f"    - Poses < 2.0 Å: {good_poses}/{len(rmsd_series)}")
    print()

def main():
    # Main function of running the RMSD calculations
    print("="*70)
    print("RMSD CALCULATOR FOR MOLECULAR STRUCTURES")
    print("="*70)
    print()

    print("Available comparison modes:")
    print("  1. X-ray  vs  Docking poses only")
    print("  2. X-ray  vs  Conformers only")
    print("  3. X-ray  vs  Conformers + Docking poses")
    print("  4. Custom (you choose reference and target files)")
    print()
    mode = input("Select mode [1/2/3/4]: ").strip()

    if mode not in ('1', '2', '3', '4'):
        print("❌ Invalid mode. Exiting.")
        sys.exit(1)


    # Step 1: Getting the files from user
    print()
    print("-"*70)

    xray_file = None
    conformer_file = None
    docking_file = None
    custom_ref_file = None   
    custom_tgt_file = None

    if mode in ('1', '2', '3'):
        xray_file = input("Enter path to X-ray ligand SDF file (reference): ").strip()

    if mode == '1':
        docking_file = input("Enter path to docking poses SDF file: ").strip()
    elif mode == '2':
        conformer_file = input("Enter path to conformers SDF file: ").strip()
    elif mode == '3':
        conformer_file = input("Enter path to conformers SDF file: ").strip()
        docking_file = input("Enter path to docking poses SDF file: ").strip()
    elif mode == '4':
        print()
        print("  The REFERENCE is the structure you trust most.")
        print("  Typically: X-ray > docking pose > conformer")
        print("  The TARGET file can contain one or many structures to compare.")
        print()
        custom_ref_file = input("Enter path to REFERENCE SDF file: ").strip()
        custom_tgt_file = input("Enter path to TARGET SDF file: ").strip()

    output_file = input("Enter output CSV filename (e.g., rmsd_results.csv): ").strip()
    if not output_file:
        output_file = "rmsd_results.csv"

    # Option: heavy-atom RMSD (standard) vs all-atom
    h_choice = input("Use heavy-atom RMSD? (recommended) [Y/n]: ").strip().lower()
    remove_hydrogens = h_choice != 'n'

    # Option: SMARTS-based selective RMSD
    print()
    print("  Optional: SMARTS-based selective RMSD")
    print("  Common patterns:")
    print("    [R]           → all ring atoms")
    print("    [#6,#7,#8]    → C, N, O atoms only")
    print("    [R1]          → atoms in exactly one ring")
    print("    [#6;R]        → ring carbons only")
    print("  Leave blank for all-atom RMSD.")
    smarts_input = input(" Enter SMARTS pattern (or press Enter to skip): ").strip()
    smarts_pattern = smarts_input if smarts_input else None

    if smarts_pattern:
        # Validate the pattern early
        test_query = Chem.MolFromSmarts(smarts_pattern)
        if test_query is None:
            print (f"  ❌ Invalid SMARTS pattern: '{smarts_pattern}'")
            sys.exit(1)
        print(f"  ✓ SMARTS pattern valid: '{smarts_pattern}'")

    print()
    print("="*70)
    print("LOADING STRUCTURES...")
    print(f"  H mode: {'Heavy-atom RMSD (excluding H)' if remove_hydrogens else 'All-atom RMSD (including H)'}")
    if smarts_pattern:
        print(f"  SMARTS filter: '{smarts_pattern}' (selective atom RMSD)")
    else:
        print(f"  Atom scope: All atoms")
    print("=" * 70)
    print()

    try:
        all_result_frames = []

        if mode in ('1','2','3'):
            print("1. Loading X-ray reference structure...")
            reference_mol = load_molecule(xray_file, mol_index=0, removeHs=remove_hydrogens)
            print()

            step_num = 2

            # Conformers (modes 2 and 3)
            if conformer_file:
                print(f"{step_num}. Loading conformers...")
                conformers = load_all_molecules(conformer_file,
                                                removeHs=remove_hydrogens)
                conformer_labels = [f"Conformer_{i+1}" for i in range(len(conformers))]
                print()

                print("  X-RAY vs CONFORMERS:")
                print("  " + "-" * 40)
                conf_results = calculate_rmsd_batch(
                    reference_mol, conformers,
                    labels=conformer_labels,
                    smarts_pattern=smarts_pattern
                )
                conf_results['Type'] = 'Conformer'
                conf_results['Comparison'] = 'Xray_vs_Conformer'
                all_result_frames.append(conf_results)
                print()
                step_num += 1

            # Docking poses (modes 1 and 3)
            if docking_file:
                print(f"{step_num}. Loading docking poses...")
                docking_poses = load_all_molecules(docking_file,
                                                   removeHs=remove_hydrogens)
                docking_labels = [f"DockingPose_{i+1}" for i in range(len(docking_poses))]
                print()

                print("  X-RAY vs DOCKING POSES:")
                print("  " + "-" * 40)
                dock_results = calculate_rmsd_batch(
                    reference_mol, docking_poses,
                    labels=docking_labels,
                    smarts_pattern=smarts_pattern
                )
                dock_results['Type'] = 'Docking_Pose'
                dock_results['Comparison'] = 'Xray_vs_Docking'
                all_result_frames.append(dock_results)
                print()

        # Mode 4: Custom Reference vs Target
        elif mode == '4':
            print("1. Loading REFERENCE structure...")
            reference_mol = load_molecule(custom_ref_file, mol_index=0,
                                          removeHs=remove_hydrogens)
            print()

            print("2. Loading TARGET structures...")
            targets = load_all_molecules(custom_tgt_file,
                                         removeHs=remove_hydrogens)
            target_labels = [f"Target_{i+1}" for i in range(len(targets))]
            print()


            print("  REFERENCE vs TARGETS:")
            print("  " + "-" * 40)
            custom_results = calculate_rmsd_batch(
                reference_mol, targets,
                labels=target_labels,
                smarts_pattern=smarts_pattern
            )
            custom_results['Type'] = 'Target'
            custom_results['Comparison'] = 'Custom'
            all_result_frames.append(custom_results)
            print()
        
    
        # Combine results
        if not all_result_frames:
            print("❌ No comparisons were performed.")
            sys.exit(1)

        all_results = pd.concat(all_result_frames, ignore_index=True)

        # Reorder Columns
        columns = ['Comparison', 'Type', 'Structure_ID', 'Label', 'RMSD (Å)', 'Atoms_Used']
        for col in columns:
            if col not in all_results.columns:
                all_results[col] = None
        all_results = all_results[columns]

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
        print("=" * 70)
        print("STATISTICS")
        print("=" * 70)
        print()

        for comparison_name, group_df in all_results.groupby('Comparison'):
            rmsd_values = group_df['RMSD (Å)'].dropna()
            print_statistics(comparison_name, rmsd_values, group_df)

        print(f"  ✓ Results saved to: {output_file}")
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