"""
Pure-Python (numpy/pandas) PWM/PPM generation.

No PyRosetta or AF3 dependency.
"""

import os
import numpy as np
import pandas as pd


def pwm_from_energies_with_temperature(WT_seq, WT_energy, mutation_dict,
                                       tau=1.0, value='ddg'):
    """
    Build a PPM and PWM from a dict of mutation energies using a
    Boltzmann (softmax) distribution with temperature *tau*.

    Parameters
    ----------
    WT_seq : str
        Wild-type DNA sequence.
    WT_energy : float
        Wild-type interface energy.
    mutation_dict : pd.DataFrame
        DataFrame with columns ``tag`` (mutation identifier) and *value*.
        Tag format expected: ``..._<1-based-pos>_<OLDtoNEW>_...``
    tau : float
        Temperature parameter (lower = sharper distribution).
    value : str
        Column name for the energy value.

    Returns
    -------
    energies : np.ndarray, shape (L, 4)
    PPM : np.ndarray, shape (L, 4)
    PWM : np.ndarray, shape (L, 4)
    """
    bases = ["A", "C", "G", "T"]
    L = len(WT_seq)

    energies = np.full((L, 4), np.nan)

    # Fill WT energy
    for i, base in enumerate(WT_seq):
        energies[i][bases.index(base)] = WT_energy

    # Fill mutant energies
    for _, row in mutation_dict.iterrows():
        tag_parts = os.path.basename(row['tag']).split('_')
        pos = int(tag_parts[1]) - 1          # 0-based
        mut_key = tag_parts[2]               # e.g. "CtoG"
        _, new_base = mut_key.split("to")
        b_idx = bases.index(new_base.upper())
        energies[pos][b_idx] = row[value]

    # Fill remaining NaN with WT energy
    for i in range(L):
        for j in range(4):
            if np.isnan(energies[i][j]):
                energies[i][j] = WT_energy

    deltaE = energies - energies.min(axis=1, keepdims=True)
    PPM = np.exp(-deltaE / tau)
    PPM /= PPM.sum(axis=1, keepdims=True)
    PWM = np.log2(PPM / 0.25)

    return energies, PPM, PWM


def pwm_from_hybrid_csv(csv_path, tau=1.5, value='mut_ddg'):
    """
    Generate PPM/PWM from a ``pwm_results_hybrid.csv`` produced by
    :func:`pwm_hybrid.pipeline.generate_pwm_hybrid`.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    tau : float
        Boltzmann temperature parameter.
    value : str
        Column to use for energies (``'mut_ddg'`` or ``'ddG'``).

    Returns
    -------
    seq : str
    energies : np.ndarray, shape (L, 4)
    PPM : np.ndarray, shape (L, 4)
    PWM : np.ndarray, shape (L, 4)
    """
    bases = ['A', 'C', 'G', 'T']

    df = pd.read_csv(csv_path)

    if 'wt_ddg' in df.columns:
        wt_energy = df['wt_ddg'].iloc[0]
    else:
        wt_energy = df[value].min()

    seq_df = df.drop_duplicates(subset='position').sort_values('position')
    seq = ''.join(seq_df['original'].tolist())
    L = len(seq)

    print(f"Sequence: {seq}")
    print(f"Length: {L}")
    print(f"WT energy: {wt_energy:.3f}")

    energies = np.full((L, 4), np.nan)

    for i, base in enumerate(seq):
        energies[i][bases.index(base)] = wt_energy

    for _, row in df.iterrows():
        pos = int(row['position']) - 1
        mut_base = row['mutant']
        orig_base = row['original']
        if mut_base != orig_base:
            b_idx = bases.index(mut_base)
            energies[pos][b_idx] = row[value]

    energies = np.nan_to_num(energies, nan=0.0)

    deltaE = energies - energies.min(axis=1, keepdims=True)
    PPM = np.exp(-deltaE / tau)
    PPM /= PPM.sum(axis=1, keepdims=True)
    PWM = np.log2(PPM / 0.25)

    return seq, energies, PPM, PWM
