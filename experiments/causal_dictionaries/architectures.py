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

4. ContrastiveProductOfExperts: Combines PoE's factored codebooks with
   contrastive specialization pressure on the rule codebook. The position
   codebook learns from reconstruction alone, while the rule codebook
   receives both reconstruction and contrastive gradients.
"""

from __future__ import annotations

import numpy as np

# Hard cap on sparse code activations.
_Z_CLIP: float = 5.0


def _run_ista(
    x: np.ndarray,
    d: np.ndarray,
    n_atoms: int,
    n_settle: int,
    infer_rate: float,
    sparsity: float,
) -> np.ndarray:
    """Run ISTA settling to infer sparse codes for a single codebook.

    Args:
        x: Input batch of shape (batch, input_dim).
        d: Dictionary matrix of shape (input_dim, n_atoms).
        n_atoms: Number of dictionary atoms.
        n_settle: Number of settling iterations.
        infer_rate: Step size for sparse inference.
        sparsity: Soft-threshold level.

    Returns:
        Sparse codes z of shape (batch, n_atoms), non-negative,
        clipped at ``_Z_CLIP``.
    """
    z = np.zeros((x.shape[0], n_atoms))
    for _ in range(n_settle):
        residual = x - z @ d.T
        z = z + infer_rate * (residual @ d)
        z = np.maximum(0.0, z - sparsity * infer_rate)
        np.minimum(z, _Z_CLIP, out=z)
    return z


class ProductOfExperts:
    """Factored dictionary: D = rule_codebook + position_codebook.

    Why? Standard dictionaries allocate atoms to cover the joint
    (rule, position) space. Gravity has many positions -> many atoms.
    By factoring into rule codes (``n_rule_atoms`` dims) and position
    codes (``n_pos_atoms`` dims), each rule gets a compact representation
    regardless of positional diversity. Composition = combining rule codes
    from different rules.

    Architecture:
        x -> [rule_code, pos_code] via two parallel ISTA settlers
        reconstruction = rule_decoder(rule_code) + pos_decoder(pos_code)

    The rule codes are what we evaluate for compositionality.

    Args:
        n_rule_atoms: Number of atoms in the rule codebook.
        n_pos_atoms: Number of atoms in the position codebook.
        sparsity: Sparsity penalty coefficient.
        infer_rate: Step size for ISTA inference.
        learn_rate: Step size for Hebbian dictionary updates.
        n_settle: Number of ISTA settling iterations.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
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
        """Train both codebooks jointly on unlabelled data.

        Args:
            data: Training data of shape (N, input_dim).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.

        Returns:
            List of dicts with keys ``"epoch"`` and ``"loss"`` per epoch.
        """
        n, d = data.shape
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

                z_rule = _run_ista(
                    x, self._D_rule, self.n_rule_atoms,
                    self.n_settle, self.infer_rate, self.sparsity,
                )
                z_pos = _run_ista(
                    x, self._D_pos, self.n_pos_atoms,
                    self.n_settle, self.infer_rate, self.sparsity,
                )

                recon = z_rule @ self._D_rule.T + z_pos @ self._D_pos.T
                residual = x - recon

                self._D_rule += self.learn_rate * (residual.T @ z_rule) / b
                self._D_rule /= np.linalg.norm(self._D_rule, axis=0, keepdims=True) + 1e-8
                self._D_pos += self.learn_rate * (residual.T @ z_pos) / b
                self._D_pos /= np.linalg.norm(self._D_pos, axis=0, keepdims=True) + 1e-8

                epoch_loss += float(np.mean(residual**2))
                nb += 1
            history.append({"epoch": epoch, "loss": epoch_loss / max(nb, 1)})
        return history

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode inputs as concatenated [rule_codes, pos_codes].

        Args:
            data: Input data of shape (N, input_dim).

        Returns:
            Concatenated codes of shape (N, n_rule_atoms + n_pos_atoms).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._D_rule is None or self._D_pos is None:
            msg = "Model not trained yet. Call train() first."
            raise RuntimeError(msg)
        z_rule = _run_ista(
            data, self._D_rule, self.n_rule_atoms,
            self.n_settle, self.infer_rate, self.sparsity,
        )
        z_pos = _run_ista(
            data, self._D_pos, self.n_pos_atoms,
            self.n_settle, self.infer_rate, self.sparsity,
        )
        return np.hstack([z_rule, z_pos])

    def reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """Compute per-sample mean squared reconstruction error.

        Args:
            data: Input data of shape (N, input_dim).

        Returns:
            Per-sample MSE of shape (N,).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._D_rule is None or self._D_pos is None:
            msg = "Model not trained yet. Call train() first."
            raise RuntimeError(msg)
        z_rule = _run_ista(
            data, self._D_rule, self.n_rule_atoms,
            self.n_settle, self.infer_rate, self.sparsity,
        )
        z_pos = _run_ista(
            data, self._D_pos, self.n_pos_atoms,
            self.n_settle, self.infer_rate, self.sparsity,
        )
        recon = z_rule @ self._D_rule.T + z_pos @ self._D_pos.T
        return np.mean((data - recon) ** 2, axis=1)

    @property
    def dictionary(self) -> np.ndarray:
        """Combined dictionary [D_rule | D_pos] of shape (input_dim, n_atoms).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._D_rule is None or self._D_pos is None:
            msg = "Model not trained yet. Call train() first."
            raise RuntimeError(msg)
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

    Args:
        n_atoms: Number of slots.
        n_settle: Number of slot-refinement iterations.
        seed: Random seed for reproducibility.
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
        """Train slot prototypes via iterative refinement.

        Args:
            data: Training data of shape (N, input_dim).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.

        Returns:
            List of dicts with keys ``"epoch"`` and ``"loss"`` per epoch.
        """
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
                    attn_logits = x @ slots.T * self._scale
                    attn = _softmax(attn_logits, axis=-1)
                    # Vectorized slot update: (n_atoms, d)
                    w_norm = attn.sum(axis=0, keepdims=True) + 1e-8  # (1, n_atoms)
                    slots = (attn.T @ x) / w_norm.T

                recon = attn @ slots
                residual = x - recon
                batch_loss = float(np.mean(residual**2))

                self._slots = 0.9 * self._slots + 0.1 * slots
                epoch_loss += batch_loss
                nb += 1
            history.append({"epoch": epoch, "loss": epoch_loss / max(nb, 1)})
        return history

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Return slot attention weights as activation codes.

        Args:
            data: Input data of shape (N, input_dim).

        Returns:
            Attention weights of shape (N, n_atoms).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._slots is None:
            msg = "Model not trained yet. Call train() first."
            raise RuntimeError(msg)
        slots = self._slots.copy()
        for _ in range(self.n_settle):
            attn_logits = data @ slots.T * self._scale
            attn = _softmax(attn_logits, axis=-1)
            w_norm = attn.sum(axis=0, keepdims=True) + 1e-8
            slots = (attn.T @ data) / w_norm.T
        return _softmax(data @ slots.T * self._scale, axis=-1)

    def reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction MSE.

        Args:
            data: Input data of shape (N, input_dim).

        Returns:
            Per-sample MSE of shape (N,).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._slots is None:
            msg = "Model not trained yet. Call train() first."
            raise RuntimeError(msg)
        codes = self.encode(data)
        recon = codes @ self._slots
        return np.mean((data - recon) ** 2, axis=1)

    @property
    def dictionary(self) -> np.ndarray:
        """Slot prototypes transposed to (input_dim, n_atoms).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._slots is None:
            msg = "Model not trained yet. Call train() first."
            raise RuntimeError(msg)
        return self._slots.T


class ContrastiveDictionary:
    """ISTA dictionary with contrastive specialization pressure.

    Why? Standard ISTA optimizes reconstruction only. Atoms specialize
    as a side effect of sparsity, but this is weak. Contrastive learning
    adds direct pressure: for each batch, we know which rule generated
    each event. We penalize atoms that fire on multiple rules.

    This requires rule labels during training (available since we generate
    the data), but NOT during inference. The dictionary itself is
    rule-agnostic; the contrastive loss just shapes it during learning.

    Loss = reconstruction_error + lambda * cross_rule_activation_penalty

    Args:
        n_atoms: Number of dictionary atoms.
        sparsity: Sparsity penalty coefficient.
        contrastive_weight: Weight for the contrastive specialization loss.
        infer_rate: Step size for ISTA inference.
        learn_rate: Step size for Hebbian dictionary updates.
        n_settle: Number of ISTA settling iterations.
        seed: Random seed for reproducibility.
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
        """Train with rule labels for contrastive specialization pressure.

        Args:
            rule_data: Mapping from rule name to encoded event arrays,
                each of shape (N_rule, input_dim).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.

        Returns:
            List of dicts with keys ``"epoch"`` and ``"loss"`` per epoch.
        """
        rules = list(rule_data.keys())
        n_rules = len(rules)

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

                z = _run_ista(
                    x, self._D, self.n_atoms,
                    self.n_settle, self.infer_rate, self.sparsity,
                )

                residual = x - z @ self._D.T
                recon_grad = (residual.T @ z) / b

                # Vectorized contrastive gradient.
                # rule_means_mat[ri, j] = mean |activation| of atom j on rule ri.
                rule_means_mat = np.zeros((n_rules, self.n_atoms))
                for ri in range(n_rules):
                    mask = y == ri
                    if mask.any():
                        rule_means_mat[ri] = np.abs(z[mask]).mean(axis=0)

                # penalty[ri, j] = rule_means_mat[ri, j] for non-max rules, 0 otherwise.
                max_rule = np.argmax(rule_means_mat, axis=0)  # (n_atoms,)
                penalty = rule_means_mat.copy()
                penalty[max_rule, np.arange(self.n_atoms)] = 0.0

                # Push atoms away from non-dominant rule data.
                contrast_grad = np.zeros_like(self._D)
                for ri in range(n_rules):
                    mask = y == ri
                    if mask.any() and penalty[ri].max() > 1e-12:
                        x_mean = x[mask].mean(axis=0)   # (input_dim,)
                        contrast_grad -= (x_mean[:, None] * penalty[ri][None, :]) * 0.1

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
        """Train without labels — falls back to standard ISTA (no contrastive loss).

        Args:
            data: Training data of shape (N, input_dim).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.

        Returns:
            List of dicts with keys ``"epoch"`` and ``"loss"`` per epoch.
        """
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
                z = _run_ista(
                    x, self._D, self.n_atoms,
                    self.n_settle, self.infer_rate, self.sparsity,
                )
                residual = x - z @ self._D.T
                self._D += self.learn_rate * (residual.T @ z) / b
                self._D /= np.linalg.norm(self._D, axis=0, keepdims=True) + 1e-8
                epoch_loss += float(np.mean(residual**2))
                nb += 1
            history.append({"epoch": epoch, "loss": epoch_loss / max(nb, 1)})
        return history

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode inputs via ISTA settling.

        Args:
            data: Input data of shape (N, input_dim).

        Returns:
            Sparse codes of shape (N, n_atoms).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._D is None:
            msg = "Model not trained yet. Call train() or train_with_labels() first."
            raise RuntimeError(msg)
        return _run_ista(
            data, self._D, self.n_atoms,
            self.n_settle, self.infer_rate, self.sparsity,
        )

    def reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """Compute per-sample mean squared reconstruction error.

        Args:
            data: Input data of shape (N, input_dim).

        Returns:
            Per-sample MSE of shape (N,).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._D is None:
            msg = "Model not trained yet. Call train() or train_with_labels() first."
            raise RuntimeError(msg)
        z = self.encode(data)
        return np.mean((data - z @ self._D.T) ** 2, axis=1)

    @property
    def dictionary(self) -> np.ndarray:
        """The dictionary matrix D of shape (input_dim, n_atoms).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._D is None:
            msg = "Model not trained yet. Call train() or train_with_labels() first."
            raise RuntimeError(msg)
        return self._D


class ContrastiveProductOfExperts:
    """PoE with contrastive specialization on the rule codebook.

    Combines two architectural ideas:
    - PoE: factor into rule_codebook (what rule) and pos_codebook (where)
    - Contrastive: push rule atoms to specialize to individual rules

    The position codebook learns from reconstruction only. The rule
    codebook gets both reconstruction gradient AND contrastive gradient
    that penalizes atoms firing on multiple rules.

    Requires rule labels during training (available since we generate
    data), but NOT during inference.

    Args:
        n_rule_atoms: Number of atoms in the rule codebook.
        n_pos_atoms: Number of atoms in the position codebook.
        sparsity: Sparsity penalty coefficient.
        contrastive_weight: Weight for the contrastive specialization loss.
        infer_rate: Step size for ISTA inference.
        learn_rate: Step size for Hebbian dictionary updates.
        n_settle: Number of ISTA settling iterations.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_rule_atoms: int = 5,
        n_pos_atoms: int = 5,
        sparsity: float = 0.02,
        contrastive_weight: float = 1.0,
        infer_rate: float = 0.1,
        learn_rate: float = 0.02,
        n_settle: int = 50,
        seed: int = 42,
    ) -> None:
        self.n_atoms = n_rule_atoms + n_pos_atoms
        self.n_rule_atoms = n_rule_atoms
        self.n_pos_atoms = n_pos_atoms
        self.sparsity = sparsity
        self.contrastive_weight = contrastive_weight
        self.infer_rate = infer_rate
        self.learn_rate = learn_rate
        self.n_settle = n_settle
        self._rng = np.random.default_rng(seed)
        self._D_rule: np.ndarray | None = None
        self._D_pos: np.ndarray | None = None

    def train_with_labels(
        self,
        rule_data: dict[str, np.ndarray],
        epochs: int = 80,
        batch_size: int = 64,
    ) -> list[dict[str, float | int]]:
        """Train with rule labels for contrastive pressure on the rule codebook.

        Args:
            rule_data: Mapping from rule name to encoded event arrays,
                each of shape (N_rule, input_dim).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.

        Returns:
            List of dicts with keys ``"epoch"`` and ``"loss"`` per epoch.
        """
        rules = list(rule_data.keys())
        n_rules = len(rules)

        all_x = []
        all_labels = []
        for ri, rule in enumerate(rules):
            all_x.append(rule_data[rule])
            all_labels.extend([ri] * len(rule_data[rule]))
        data = np.vstack(all_x)
        labels = np.array(all_labels)
        n, d = data.shape

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
                bi = idx[start : start + batch_size]
                x = data[bi]
                y = labels[bi]
                b = x.shape[0]

                z_rule = _run_ista(
                    x, self._D_rule, self.n_rule_atoms,
                    self.n_settle, self.infer_rate, self.sparsity,
                )
                z_pos = _run_ista(
                    x, self._D_pos, self.n_pos_atoms,
                    self.n_settle, self.infer_rate, self.sparsity,
                )

                recon = z_rule @ self._D_rule.T + z_pos @ self._D_pos.T
                residual = x - recon

                # Position codebook: pure reconstruction update.
                self._D_pos += self.learn_rate * (residual.T @ z_pos) / b
                self._D_pos /= np.linalg.norm(self._D_pos, axis=0, keepdims=True) + 1e-8

                # Rule codebook: reconstruction + vectorized contrastive gradient.
                recon_grad = (residual.T @ z_rule) / b

                rule_means_mat = np.zeros((n_rules, self.n_rule_atoms))
                for ri in range(n_rules):
                    mask = y == ri
                    if mask.any():
                        rule_means_mat[ri] = np.abs(z_rule[mask]).mean(axis=0)

                max_rule = np.argmax(rule_means_mat, axis=0)
                penalty = rule_means_mat.copy()
                penalty[max_rule, np.arange(self.n_rule_atoms)] = 0.0

                contrast_grad = np.zeros_like(self._D_rule)
                for ri in range(n_rules):
                    mask = y == ri
                    if mask.any() and penalty[ri].max() > 1e-12:
                        x_mean = x[mask].mean(axis=0)
                        contrast_grad -= (x_mean[:, None] * penalty[ri][None, :]) * 0.1

                self._D_rule += self.learn_rate * (
                    recon_grad + self.contrastive_weight * contrast_grad
                )
                self._D_rule /= np.linalg.norm(self._D_rule, axis=0, keepdims=True) + 1e-8

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
        """Train without labels — falls back to PoE without contrastive loss.

        Args:
            data: Training data of shape (N, input_dim).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.

        Returns:
            List of dicts with keys ``"epoch"`` and ``"loss"`` per epoch.
        """
        n, d = data.shape
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
                z_rule = _run_ista(
                    x, self._D_rule, self.n_rule_atoms,
                    self.n_settle, self.infer_rate, self.sparsity,
                )
                z_pos = _run_ista(
                    x, self._D_pos, self.n_pos_atoms,
                    self.n_settle, self.infer_rate, self.sparsity,
                )
                recon = z_rule @ self._D_rule.T + z_pos @ self._D_pos.T
                residual = x - recon
                self._D_rule += self.learn_rate * (residual.T @ z_rule) / b
                self._D_rule /= np.linalg.norm(self._D_rule, axis=0, keepdims=True) + 1e-8
                self._D_pos += self.learn_rate * (residual.T @ z_pos) / b
                self._D_pos /= np.linalg.norm(self._D_pos, axis=0, keepdims=True) + 1e-8
                epoch_loss += float(np.mean(residual**2))
                nb += 1
            history.append({"epoch": epoch, "loss": epoch_loss / max(nb, 1)})
        return history

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode inputs as concatenated [rule_codes, pos_codes].

        Args:
            data: Input data of shape (N, input_dim).

        Returns:
            Concatenated codes of shape (N, n_rule_atoms + n_pos_atoms).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._D_rule is None or self._D_pos is None:
            msg = "Model not trained yet. Call train() or train_with_labels() first."
            raise RuntimeError(msg)
        z_rule = _run_ista(
            data, self._D_rule, self.n_rule_atoms,
            self.n_settle, self.infer_rate, self.sparsity,
        )
        z_pos = _run_ista(
            data, self._D_pos, self.n_pos_atoms,
            self.n_settle, self.infer_rate, self.sparsity,
        )
        return np.hstack([z_rule, z_pos])

    def reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """Compute per-sample mean squared reconstruction error.

        Args:
            data: Input data of shape (N, input_dim).

        Returns:
            Per-sample MSE of shape (N,).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._D_rule is None or self._D_pos is None:
            msg = "Model not trained yet. Call train() or train_with_labels() first."
            raise RuntimeError(msg)
        z_rule = _run_ista(
            data, self._D_rule, self.n_rule_atoms,
            self.n_settle, self.infer_rate, self.sparsity,
        )
        z_pos = _run_ista(
            data, self._D_pos, self.n_pos_atoms,
            self.n_settle, self.infer_rate, self.sparsity,
        )
        recon = z_rule @ self._D_rule.T + z_pos @ self._D_pos.T
        return np.mean((data - recon) ** 2, axis=1)

    @property
    def dictionary(self) -> np.ndarray:
        """Combined dictionary [D_rule | D_pos] of shape (input_dim, n_atoms).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._D_rule is None or self._D_pos is None:
            msg = "Model not trained yet. Call train() or train_with_labels() first."
            raise RuntimeError(msg)
        return np.hstack([self._D_rule, self._D_pos])


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute numerically stable softmax.

    Args:
        x: Input array.
        axis: Axis along which to compute softmax.

    Returns:
        Softmax output with the same shape as ``x``.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / (e_x.sum(axis=axis, keepdims=True) + 1e-12)
