"""Analysis functions for evaluating causal dictionary specialization.

Computes atom-rule affinity, specialization scores, reconstruction
ratios for composition events, and Jaccard similarity of activated atoms.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class DictionaryModel(Protocol):
    """Protocol for dictionary models with encode/reconstruct interface."""

    n_atoms: int

    def encode(self, data: np.ndarray) -> np.ndarray: ...
    def reconstruction_error(self, data: np.ndarray) -> np.ndarray: ...


def atom_rule_affinity(
    sd: DictionaryModel,
    rule_data: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute mean activation of each atom on each rule's data.

    For each rule, encodes the data with the trained dictionary and
    computes the mean absolute activation per atom.

    Args:
        sd: A trained SparseDictionary instance.
        rule_data: Mapping from rule name to encoded data arrays,
            each of shape (N_rule, input_dim).

    Returns:
        Affinity matrix of shape (n_atoms, n_rules), where entry (i, j)
        is the mean |activation| of atom i on rule j's data.
    """
    rule_names = list(rule_data.keys())
    n_rules = len(rule_names)
    n_atoms = sd.n_atoms
    affinity = np.zeros((n_atoms, n_rules))

    for j, rule_name in enumerate(rule_names):
        data = rule_data[rule_name]
        codes = sd.encode(data)
        affinity[:, j] = np.mean(np.abs(codes), axis=0)

    return affinity


def specialization_scores(
    sd: DictionaryModel,
    rule_data: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute per-atom specialization scores.

    Specialization is defined as max(affinity_row) / sum(affinity_row).
    A score of 1.0 means the atom responds to exactly one rule.
    A score of 1/n_rules means equal response to all rules.

    Args:
        sd: A trained SparseDictionary instance.
        rule_data: Mapping from rule name to encoded data arrays.

    Returns:
        Specialization scores of shape (n_atoms,), each in
        [1/n_rules, 1.0].
    """
    affinity = atom_rule_affinity(sd, rule_data)
    row_sums = affinity.sum(axis=1)
    # Avoid division by zero for atoms that never activate
    row_sums = np.maximum(row_sums, 1e-12)
    row_maxes = affinity.max(axis=1)
    return row_maxes / row_sums


def composition_reconstruction_ratio(
    sd: DictionaryModel,
    single_data: np.ndarray,
    comp_data: np.ndarray,
) -> float:
    """Compute reconstruction error ratio for composition vs single-rule.

    A ratio near 1.0 means composition events are reconstructed as
    well as single-rule events (good compositionality). A high ratio
    means the dictionary struggles with compositions.

    Args:
        sd: A trained SparseDictionary instance.
        single_data: Encoded single-rule events, shape (N, input_dim).
        comp_data: Encoded composition events, shape (M, input_dim).

    Returns:
        Ratio of mean_error(comp_data) / mean_error(single_data).
    """
    single_err = float(np.mean(sd.reconstruction_error(single_data)))
    comp_err = float(np.mean(sd.reconstruction_error(comp_data)))
    return comp_err / max(single_err, 1e-12)


def atom_union_jaccard(
    sd: DictionaryModel,
    rule_a_data: np.ndarray,
    rule_b_data: np.ndarray,
    comp_data: np.ndarray,
    activation_threshold: float = 0.1,
) -> float:
    """Compute Jaccard similarity of activated atom sets.

    Measures whether composition events activate the union of atoms
    from individual rules, testing true compositionality.

    Args:
        sd: A trained SparseDictionary instance.
        rule_a_data: Encoded data for rule A, shape (N, input_dim).
        rule_b_data: Encoded data for rule B, shape (M, input_dim).
        comp_data: Encoded composition data, shape (K, input_dim).
        activation_threshold: Minimum mean activation to consider
            an atom as "active" for a dataset.

    Returns:
        Jaccard similarity |C inter (A union B)| / |C union (A union B)|,
        where A, B, C are sets of active atom indices.
    """
    codes_a = sd.encode(rule_a_data)
    codes_b = sd.encode(rule_b_data)
    codes_c = sd.encode(comp_data)

    set_a = set(np.where(np.mean(np.abs(codes_a), axis=0) > activation_threshold)[0])
    set_b = set(np.where(np.mean(np.abs(codes_b), axis=0) > activation_threshold)[0])
    set_c = set(np.where(np.mean(np.abs(codes_c), axis=0) > activation_threshold)[0])

    union_ab = set_a | set_b
    intersection = set_c & union_ab
    union_all = set_c | union_ab

    if len(union_all) == 0:
        return 0.0

    return len(intersection) / len(union_all)
