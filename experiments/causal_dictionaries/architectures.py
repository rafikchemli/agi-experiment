"""Alternative architectures for causal dictionary learning.

Each architecture addresses a structural limitation of vanilla sparse coding:

1. ProductOfExperts: Separates "what rule" from "where it happens" via
   factored latent space. Composition = activating multiple rule factors.

2. SlotDictionary: Fixed slots that bind to rules via competition.
   Inspired by slot attention (Locatello et al., 2020). Composition
   fills multiple slots rather than blending atoms.

3. ContrastiveDictionary: ISTA + contrastive loss that pushes atoms
   to fire on one rule and NOT fire on others. Direct specialization
   pressure without relying on sparsity alone.
"""

from __future__ import annotations

import numpy as np


class ProductOfExperts:
    """Factored dictionary: D = rule_codebook x position_codebook.

    Why? Standard dictionaries allocate atoms to cover the joint
    (rule, position) space. Gravity has many positions → many atoms.
    By factoring into rule codes (3-5 dims) and position codes (3-5 dims),
    each rule gets a compact representation regardless of positional
    diversity. Composition = combining rule codes from different rules.

    Architecture:
        x → [rule_code, pos_code] via two parallel ISTA settlers
        reconstruction = rule_decoder(rule_code) + pos_decoder(pos_code)

    The rule codes are what we evaluate for compositionality.
    """

    def __init__(
        self,
        n_atoms: int = 8,
        n_rule_atoms: int = 4,
        n_pos_atoms: int = 4,
        sparsity: float = 0.02,
        infer_rate: float = 0.1,
        learn_rate: float = 0.02,
        n_settle: int = 50,
        seed: int = 42,
    ) -> None:
        self.n_atoms = n_rule_atoms + n_pos_atoms
        self.n_rule_atoms = n_rule_atoms
        self.n_pos_atoms = n_pos_atoms
        self.sparsity = sparsity
        self.infer_rate = infer_rate
        self.learn_rate = learn_rate
        self.n_settle = n_settle
        self._rng = np.random.default_rng(seed)
        self._D_rule: np.ndarray | None = None
        self._D_pos: np.ndarray | None = None

    def train(
        self,
        data: np.ndarray,
        epochs: int = 80,
        batch_size: int = 64,
    ) -> list[dict[str, float | int]]:
        """Train both codebooks jointly."""
        n, d = data.shape
        # Initialize both dictionaries
        self._D_rule = self._rng.standard_normal((d, self.n_rule_atoms))
        self._D_rule /= np.linalg.norm(self._D_rule, axis=0, keepdims=True) + 1e-8
        self._D_pos = self._rng.standard_normal((d, self.n_pos_atoms))
        self._D_pos /= np.linalg.norm(self._D_pos, axis=0, keepdims=True) + 1e-8

        history: list[dict[str, float | int]] = []
        for epoch in range(1, epochs + 1):
            idx = self._rng.permutation(n)
            epoch_loss = 0.0
            nb = 0
            for start in range(0, n, batch_size):
                x = data[idx[start : start + batch_size]]
                b = x.shape[0]

                # Parallel ISTA settle for both codebooks
                z_rule = self._ista(x, self._D_rule, self.n_rule_atoms)
                z_pos = self._ista(x, self._D_pos, self.n_pos_atoms)

                # Combined reconstruction
                recon = z_rule @ self._D_rule.T + z_pos @ self._D_pos.T
                residual = x - recon

                # Separate Hebbian updates
                # Rule dictionary gets residual signal
                self._D_rule += self.learn_rate * (residual.T @ z_rule) / b
                self._D_rule /= np.linalg.norm(self._D_rule, axis=0, keepdims=True) + 1e-8
                # Position dictionary gets residual signal
                self._D_pos += self.learn_rate * (residual.T @ z_pos) / b
                self._D_pos /= np.linalg.norm(self._D_pos, axis=0, keepdims=True) + 1e-8

                epoch_loss += float(np.mean(residual**2))
                nb += 1
            history.append({"epoch": epoch, "loss": epoch_loss / max(nb, 1)})
        return history

    def _ista(self, x: np.ndarray, d: np.ndarray, n_atoms: int) -> np.ndarray:
        """ISTA settling for one codebook."""
        n = x.shape[0]
        z = np.zeros((n, n_atoms))
        for _ in range(self.n_settle):
            residual = x - z @ d.T
            drive = residual @ d
            z = z + self.infer_rate * drive
            z = np.maximum(0.0, z - self.sparsity * self.infer_rate)
            np.minimum(z, 5.0, out=z)
        return z

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Return concatenated [rule_codes, pos_codes]."""
        assert self._D_rule is not None and self._D_pos is not None  # noqa: S101
        z_rule = self._ista(data, self._D_rule, self.n_rule_atoms)
        z_pos = self._ista(data, self._D_pos, self.n_pos_atoms)
        return np.hstack([z_rule, z_pos])

    def reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """Per-sample MSE."""
        assert self._D_rule is not None and self._D_pos is not None  # noqa: S101
        z_rule = self._ista(data, self._D_rule, self.n_rule_atoms)
        z_pos = self._ista(data, self._D_pos, self.n_pos_atoms)
        recon = z_rule @ self._D_rule.T + z_pos @ self._D_pos.T
        return np.mean((data - recon) ** 2, axis=1)

    @property
    def dictionary(self) -> np.ndarray:
        """Combined dictionary [D_rule | D_pos]."""
        assert self._D_rule is not None and self._D_pos is not None  # noqa: S101
        return np.hstack([self._D_rule, self._D_pos])


class SlotDictionary:
    """Slot-based dictionary with competitive binding.

    Why? Standard sparse coding blends atoms via superposition. But physical
    rules don't blend — they compose discretely (an object either falls OR
    doesn't). Slots provide discrete binding: each slot captures one factor,
    and composition = filling multiple slots.

    Inspired by Slot Attention (Locatello et al., 2020) but simplified:
    - K slots, each with a learnable prototype vector
    - Soft-competitive assignment: each input dimension routes to best slot
    - Slots are updated via weighted mean of assigned inputs

    For composition evaluation, we check which slots activate (not atoms).
    """

    def __init__(
        self,
        n_atoms: int = 8,
        n_settle: int = 3,
        seed: int = 42,
        **kwargs: object,
    ) -> None:
        self.n_atoms = n_atoms
        self.n_settle = n_settle
        self._rng = np.random.default_rng(seed)
        self._slots: np.ndarray | None = None  # (n_atoms, input_dim)
        self._scale: float = 1.0

    def train(
        self,
        data: np.ndarray,
        epochs: int = 80,
        batch_size: int = 64,
    ) -> list[dict[str, float | int]]:
        """Train slot prototypes via iterative refinement."""
        n, d = data.shape
        # Initialize slots from data (k-means++ style)
        self._slots = np.zeros((self.n_atoms, d))
        self._slots[0] = data[self._rng.integers(n)]
        for k in range(1, self.n_atoms):
            dists = np.min(
                np.linalg.norm(
                    data[:, None, :] - self._slots[None, :k, :], axis=2,
                ),
                axis=1,
            )
            probs = dists**2
            probs /= probs.sum() + 1e-12
            self._slots[k] = data[self._rng.choice(n, p=probs)]

        self._scale = 1.0 / np.sqrt(d)
        history: list[dict[str, float | int]] = []

        for epoch in range(1, epochs + 1):
            idx = self._rng.permutation(n)
            epoch_loss = 0.0
            nb = 0
            for start in range(0, n, batch_size):
                x = data[idx[start : start + batch_size]]

                # Iterative slot refinement
                slots = self._slots.copy()
                for _ in range(self.n_settle):
                    # Attention: (batch, n_atoms)
                    attn_logits = x @ slots.T * self._scale
                    attn = _softmax(attn_logits, axis=-1)
                    # Weighted update: each slot gets its weighted mean input
                    for k in range(self.n_atoms):
                        weights = attn[:, k]  # (batch,)
                        w_sum = weights.sum() + 1e-8
                        slots[k] = (weights[:, None] * x).sum(axis=0) / w_sum

                # Reconstruction: each sample reconstructs from its slot mixture
                recon = attn @ slots
                residual = x - recon
                batch_loss = float(np.mean(residual**2))

                # Update slots with momentum
                self._slots = 0.9 * self._slots + 0.1 * slots
                epoch_loss += batch_loss
                nb += 1
            history.append({"epoch": epoch, "loss": epoch_loss / max(nb, 1)})
        return history

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Return slot attention weights as 'activation codes'."""
        assert self._slots is not None  # noqa: S101
        slots = self._slots.copy()
        for _ in range(self.n_settle):
            attn_logits = data @ slots.T * self._scale
            attn = _softmax(attn_logits, axis=-1)
            for k in range(self.n_atoms):
                weights = attn[:, k]
                w_sum = weights.sum() + 1e-8
                slots[k] = (weights[:, None] * data).sum(axis=0) / w_sum
        return _softmax(data @ slots.T * self._scale, axis=-1)

    def reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """Per-sample reconstruction MSE."""
        assert self._slots is not None  # noqa: S101
        codes = self.encode(data)
        recon = codes @ self._slots
        return np.mean((data - recon) ** 2, axis=1)

    @property
    def dictionary(self) -> np.ndarray:
        """Slot prototypes transposed to (input_dim, n_atoms)."""
        assert self._slots is not None  # noqa: S101
        return self._slots.T


class ContrastiveDictionary:
    """ISTA dictionary with contrastive specialization pressure.

    Why? Standard ISTA optimizes reconstruction only. Atoms specialize
    as a side effect of sparsity, but this is weak. Contrastive learning
    adds direct pressure: for each batch, we know which rule generated
    each event. We penalize atoms that fire on multiple rules.

    This requires rule labels during training (available since we generate
    the data), but NOT during inference. The dictionary itself is rule-agnostic;
    the contrastive loss just shapes it during learning.

    Loss = reconstruction_error + lambda * cross_rule_activation_penalty
    """

    def __init__(
        self,
        n_atoms: int = 8,
        sparsity: float = 0.02,
        contrastive_weight: float = 0.5,
        infer_rate: float = 0.1,
        learn_rate: float = 0.02,
        n_settle: int = 50,
        seed: int = 42,
        **kwargs: object,
    ) -> None:
        self.n_atoms = n_atoms
        self.sparsity = sparsity
        self.contrastive_weight = contrastive_weight
        self.infer_rate = infer_rate
        self.learn_rate = learn_rate
        self.n_settle = n_settle
        self._rng = np.random.default_rng(seed)
        self._D: np.ndarray | None = None

    def train_with_labels(
        self,
        rule_data: dict[str, np.ndarray],
        epochs: int = 80,
        batch_size: int = 64,
    ) -> list[dict[str, float | int]]:
        """Train with rule labels for contrastive pressure."""
        rules = list(rule_data.keys())
        n_rules = len(rules)

        # Build labeled dataset: (data, rule_index)
        all_x = []
        all_labels = []
        for ri, rule in enumerate(rules):
            all_x.append(rule_data[rule])
            all_labels.extend([ri] * len(rule_data[rule]))
        data = np.vstack(all_x)
        labels = np.array(all_labels)
        n, d = data.shape

        self._D = self._rng.standard_normal((d, self.n_atoms))
        self._D /= np.linalg.norm(self._D, axis=0, keepdims=True) + 1e-8

        history: list[dict[str, float | int]] = []
        for epoch in range(1, epochs + 1):
            idx = self._rng.permutation(n)
            epoch_loss = 0.0
            nb = 0
            for start in range(0, n, batch_size):
                bi = idx[start : start + batch_size]
                x = data[bi]
                y = labels[bi]
                b = x.shape[0]

                # ISTA settle
                z = self._ista_settle(x)

                # Reconstruction update
                residual = x - z @ self._D.T
                recon_grad = (residual.T @ z) / b

                # Contrastive penalty: for each atom, compute mean activation
                # per rule in this batch. Penalize if atom fires on >1 rule.
                contrast_grad = np.zeros_like(self._D)
                for j in range(self.n_atoms):
                    rule_means = np.zeros(n_rules)
                    for ri in range(n_rules):
                        mask = y == ri
                        if mask.any():
                            rule_means[ri] = np.mean(np.abs(z[mask, j]))
                    # Penalty: push toward max-only activation
                    total = rule_means.sum() + 1e-12
                    target = np.zeros(n_rules)
                    target[np.argmax(rule_means)] = total
                    penalty = rule_means - target  # negative for non-max rules
                    # Translate to dictionary gradient: reduce dictionary column
                    # alignment with non-specialized rules
                    for ri in range(n_rules):
                        if penalty[ri] > 0:
                            mask = y == ri
                            if mask.any():
                                contrast_grad[:, j] -= (
                                    penalty[ri]
                                    * x[mask].mean(axis=0)
                                    * 0.1
                                )

                # Combined update
                self._D += self.learn_rate * (
                    recon_grad + self.contrastive_weight * contrast_grad
                )
                self._D /= np.linalg.norm(self._D, axis=0, keepdims=True) + 1e-8

                epoch_loss += float(np.mean(residual**2))
                nb += 1
            history.append({"epoch": epoch, "loss": epoch_loss / max(nb, 1)})
        return history

    def train(
        self,
        data: np.ndarray,
        epochs: int = 80,
        batch_size: int = 64,
    ) -> list[dict[str, float | int]]:
        """Standard train (no labels) — falls back to regular ISTA."""
        n, d = data.shape
        self._D = self._rng.standard_normal((d, self.n_atoms))
        self._D /= np.linalg.norm(self._D, axis=0, keepdims=True) + 1e-8
        history: list[dict[str, float | int]] = []
        for epoch in range(1, epochs + 1):
            idx = self._rng.permutation(n)
            epoch_loss = 0.0
            nb = 0
            for start in range(0, n, batch_size):
                x = data[idx[start : start + batch_size]]
                b = x.shape[0]
                z = self._ista_settle(x)
                residual = x - z @ self._D.T
                self._D += self.learn_rate * (residual.T @ z) / b
                self._D /= np.linalg.norm(self._D, axis=0, keepdims=True) + 1e-8
                epoch_loss += float(np.mean(residual**2))
                nb += 1
            history.append({"epoch": epoch, "loss": epoch_loss / max(nb, 1)})
        return history

    def _ista_settle(self, x: np.ndarray) -> np.ndarray:
        """ISTA settling."""
        assert self._D is not None  # noqa: S101
        b = x.shape[0]
        z = np.zeros((b, self.n_atoms))
        for _ in range(self.n_settle):
            residual = x - z @ self._D.T
            drive = residual @ self._D
            z = z + self.infer_rate * drive
            z = np.maximum(0.0, z - self.sparsity * self.infer_rate)
            np.minimum(z, 5.0, out=z)
        return z

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode via ISTA."""
        return self._ista_settle(data)

    def reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """Per-sample MSE."""
        z = self.encode(data)
        assert self._D is not None  # noqa: S101
        return np.mean((data - z @ self._D.T) ** 2, axis=1)

    @property
    def dictionary(self) -> np.ndarray:
        assert self._D is not None  # noqa: S101
        return self._D


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / (e_x.sum(axis=axis, keepdims=True) + 1e-12)
