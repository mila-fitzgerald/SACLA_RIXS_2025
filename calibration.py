"""
Mila Fitzgerald, Oct 2025

Synthetic spectrum generation

"""

try:
    import xraylib as xl
except:
    lookup = True
import pandas as pd
import numpy as np

LINE_DB = {
    # Values obtained from https://fffred.gitlab.io/phytools/PeriodicTable.html
    "Ag": {"Ka1": 22.162, "Ka2": 21.990, "Kb1": 24.941},
    "Au": {"Ka1": 68.805, "Ka2": 67.000, "Kb1": 77.107},  
    "Co": {"Ka1": 6.93032, "Ka2": 6.91539, "Kb1": 7.64943},
    "Cu": {"Ka1": 8.048, "Ka2": 8.028, "Kb1": 8.905},
    "Fe": {"Ka1": 6.40384, "Ka2": 6.39084, "Kb1": 7.05798},
    "Mo": {"Ka1": 17.479, "Ka2": 17.374, "Kb1": 19.608},
    "Ni": {"Ka1": 7.47815, "Ka2": 7.46089, "Kb1": 8.26466},
    # very soft x-ray
    "C":  {"Ka1": 0.277},  
    "O":  {"Ka1": 0.525}
}

def parse_formula_simple(chem):
    """
    Very simple formula parser:
    Accepts either single element symbol ("Fe") or a formula "Fe2O3".
    Returns list of (symbol, mass_fraction) approximated by atom counts.
    """
    import re
    tokens = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', chem)
    if not tokens:
        raise ValueError("Cannot parse formula")
    elems = []
    for sym, num in tokens:
        n = float(num) if num else 1.0
        elems.append((sym, n))
    total = sum(n for _, n in elems)
    return [(sym, n/total) for sym, n in elems]

def get_emission_lines_lookup(chem, incident_energy_keV=10.0, min_intensity=0.0):
    """
    Fallback spectral line generator using LINE_DB.
    Returns DataFrame columns: Element, Line, Energy_keV, RelIntensity
    """
    parsed = parse_formula_simple(chem)
    rows = []
    for sym, frac in parsed:
        if sym not in LINE_DB:
            # skip elements not in our small DB
            continue
        lines = LINE_DB[sym]
        for line_name, energy_keV in lines.items():
            # simple excitation check: require incident energy > line energy
            if incident_energy_keV <= energy_keV:
                continue
            # naive relative intensity:
            # larger Z -> stronger K-lines; scale with frac and (energy_keV)**-0.5 as a crude factor
            rel = frac * (energy_keV ** -0.5) * (1.0 + 0.2 * np.log1p(energy_keV))
            rows.append({
                "Element": sym,
                "Line": line_name,
                "Energy_keV": float(energy_keV),
                "RelIntensity": rel
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No lines (fallback DB) for {chem} at {incident_energy_keV} keV.")
        return df
    df["RelIntensity"] /= df["RelIntensity"].max()
    df = df[df["RelIntensity"] > min_intensity].sort_values("Energy_keV").reset_index(drop=True)
    return df

def get_emission_lines(chem, incident_energy_keV=10.0, min_intensity=0):
    """
    Compute expected X-ray emission lines for a given compound.

    Parameters
    ----------
    chem : str
        Chemical formula (e.g. "Fe", "Cu", "Fe2O3")
    incident_energy_keV : float
        Incident photon energy (keV)
    min_intensity : float
        Minimum relative intensity to include (filters weak lines)

    Returns
    -------
    lines_df : pandas.DataFrame
        Table of emission lines with columns:
        ["Element", "Line", "Energy_keV", "RelIntensity", "CrossSection_cm2_g"]
    """
    # Parse composition
    comp = xl.CompoundParser(chem)
    elements = comp['Elements']
    fractions = comp['massFractions']

    # Collect line data
    data = []

    for Z, frac in zip(elements, fractions):
        rho = frac * xl.ElementDensity(Z)
        sym = xl.AtomicNumberToSymbol(Z)

        # Loop over standard line sets
        for line_set in ['Ka1', 'Ka2', 'Kb1', 'Lb1', 'La1', 'La2', 'Lg1', 'Lg2']:
            try:
                line_id = getattr(xl, f'LINE_{line_set.upper()}')
                edge_shell = line_set[0].upper() + '_SHELL'
                
                # Skip if below excitation edge
                edge_energy = xl.EdgeEnergy(Z, getattr(xl, edge_shell))
                if incident_energy_keV < edge_energy / 1000:
                    continue

                energy_keV = xl.LineEnergy(Z, line_id)
                cross_section = xl.CS_FluorLine_Kissel_Cascade(Z, line_id, incident_energy_keV * 1000)

                if cross_section == 0:
                    continue

                data.append({
                    "Element": sym,
                    "Line": line_set,
                    "Energy_keV": energy_keV / 1000,
                    "RelIntensity": rho * cross_section,
                    "CrossSection_cm2_g": cross_section
                })
            except Exception:
                continue

    df = pd.DataFrame(data)
    if df.empty:
        print(f"No emission lines found for {chem} at {incident_energy_keV} keV.")
        return df

    # Normalize intensities
    df["RelIntensity"] /= df["RelIntensity"].max()
    df = df[df["RelIntensity"] > min_intensity].sort_values("Energy_keV")
    df.reset_index(drop=True, inplace=True)

    return df