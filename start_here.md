# RMSD Calculator Package **summary**

This package contains everything you need to calculate RMSD between your X-ray ligand structure and computational predictions.

## 📄 File Descriptions(Pick one):

### Main Scripts

1. **'rmsd_calculator.py'** ⭐ RECOMMENDED FOR BEGINNERS

- Type: Interactive script
- Best for: First-time users, step-by-step guidance
- How to use:
  <pre>'python rmsd_calculator.py'<pre>

**What it does:**

- Asks you for file paths one by one
- Loads all your structures
- Calculates RMSD for each conformer and docking pose
- Saves results to CSV
- Shows summary statistics

2. **'rmsd_calculator_cli.py'** FOR ADVANCED USERS

- Type: Command-line script
- Best for: Users comfortable with terminal, batch processing
- How to use:
    <pre>'python rmsd_calculator_cli.py xray.sdf conformers.sdf docking.sdf output.csv'<pre>

**What it does:** Same as above, but faster for repeated use

---
