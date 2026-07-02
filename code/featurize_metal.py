"""
featurize_metal.py — matminer-based featurization for aluminum alloy compositions.

Supports implicit Al-balance mode (Al = 1.0 - sum(solutes)) and multiple
process parameters for wrought aluminum alloy property prediction.

Usage
-----
    from featurize_metal import featurize_metal_df

    df = pd.read_excel('Wrought_Al_alloy.xlsx')
    mass_frac_cols = ['Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Zn', 'V',
                      'Ti', 'Zr', 'Li', 'Ni', 'Be', 'Sc']

    X, y, col_names = featurize_metal_df(
        df,
        mass_frac_cols=mass_frac_cols,
        process_cols=['Tsol', 'Tage', 'tage'],
        target_cols=['YS', 'UTS', 'El'],
        add_al_balance=True,
    )
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pymatgen.core import Composition
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def mass_fractions_to_composition(
    mass_dict: Dict[str, float],
    normalize: bool = True,
    add_al_balance: bool = False,
) -> Composition:
    """
    Convert a dictionary of mass fractions into a pymatgen Composition.

    Parameters
    ----------
    mass_dict : dict
        Mapping from element symbol to mass fraction, e.g.
        ``{'Fe': 0.95, 'C': 0.002, 'Mn': 0.015}``.
        Values should be fractions (0–1 scale) summing to ~1.
        For aluminum alloys with Al as implicit balance, pass solute
        fractions only with ``add_al_balance=True``.
    normalize : bool
        If True, re-normalize fractions so they sum exactly to 1.
        Ignored when ``add_al_balance=True`` (normalization happens
        automatically after Al is added).
    add_al_balance : bool
        If True, compute Al = 1.0 − sum(solute fractions), add Al to the
        dict (clipped at ≥ 0), and normalize to sum = 1. Use this for
        aluminum alloys where Al is not an explicit column.

    Returns
    -------
    comp : pymatgen.Composition
    """
    if add_al_balance:
        solute_sum = sum(mass_dict.values())
        al_frac = max(0.0, 1.0 - solute_sum)
        mass_dict = {**mass_dict, 'Al': al_frac}
        # Normalize to fix any small numerical errors
        total = sum(mass_dict.values())
        if total <= 0:
            raise ValueError("Sum of mass fractions is zero.")
        mass_dict = {k: v / total for k, v in mass_dict.items()}
    elif normalize:
        total = sum(mass_dict.values())
        if total <= 0:
            raise ValueError("Sum of mass fractions is zero.")
        mass_dict = {k: v / total for k, v in mass_dict.items()}

    return Composition.from_weight_dict(mass_dict)


def _extract_element_map(mass_frac_cols: List[str]) -> Dict[str, str]:
    """
    Map column names to element symbols (e.g., 'Si' -> 'Si', 'Fe' -> 'Fe').
    Also supports legacy '_frac' suffix convention for backward compatibility.
    """
    element_map = {}
    for col in mass_frac_cols:
        if col.endswith("_frac"):
            elem = col[:-5].title()  # strip '_frac' and capitalize
        elif len(col) <= 2:
            # Direct element symbol (element symbols are 1-2 characters)
            elem = col
        else:
            elem = col.title()
        element_map[col] = elem
    return element_map


# ---------------------------------------------------------------------------
# Main featurization entry point
# ---------------------------------------------------------------------------

def featurize_metal_df(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
    process_cols: Optional[List[str]] = None,
    time_col: Optional[str] = None,
    target_col: Optional[str] = None,
    target_cols: Optional[List[str]] = None,
    use_magpie: bool = True,
    use_stoichiometry: bool = True,
    use_valence: bool = True,
    keep_raw_frac: bool = True,
    add_al_balance: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Featurize a DataFrame of metal alloy compositions (mass fractions).

    Each row is one measurement: composition + process → target property.
    Supports both single-target and multi-target regression.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ``mass_frac_cols``, process columns, and
        target column(s).
    mass_frac_cols : list of str
        Column names for elemental mass fractions, e.g.
        ``['Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr']``.
    process_cols : list of str, optional
        Column names for process parameters, e.g.
        ``['Tsol', 'Tage', 'tage']``.  Appended as features.
    time_col : str, optional
        Single time/process column (backward compatible).
        Merged into ``process_cols`` if both are provided.
    target_col : str, optional
        Name of a single target column (backward compatible).
        Ignored if ``target_cols`` is provided.
    target_cols : list of str, optional
        Names of multiple target columns for multi-output regression,
        e.g. ``['YS', 'UTS', 'El']``.
    use_magpie : bool
        Include Magpie element-property features (~132 dim).
    use_stoichiometry : bool
        Include stoichiometric attributes (~7 dim).
    use_valence : bool
        Include valence-orbital features (~10 dim).
    keep_raw_frac : bool
        Append the raw mass-fraction columns themselves as features.
    add_al_balance : bool
        If True, treat Al as the implicit balance element:
        compute Al = 1 − sum(solute fractions) before featurization.
        Use for aluminum alloys where Al is not an explicit column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (rows = samples, columns = features).
    y : pd.DataFrame
        Target values — single column for single target, multiple columns
        for multi-target (aligned with X).
    col_names : list of str
        Ordered list of feature column names (for inspection).
    """
    # ------------------------------------------------------------------
    # 0. Resolve process columns (merge time_col for backward compat)
    # ------------------------------------------------------------------
    if process_cols is None:
        if time_col is not None:
            process_cols = [time_col]
        else:
            process_cols = []
    else:
        process_cols = list(process_cols)  # copy to avoid mutating caller
    if time_col is not None and time_col not in process_cols:
        process_cols = [time_col] + process_cols
    # ------------------------------------------------------------------
    # 1. Build element map from column names
    # ------------------------------------------------------------------
    element_map = _extract_element_map(mass_frac_cols)

    # ------------------------------------------------------------------
    # 2. Collect matminer featurizers (instantiated once for label lookup)
    # ------------------------------------------------------------------
    featurizers = []
    if use_magpie:
        featurizers.append(ElementProperty.from_preset("magpie"))
    if use_stoichiometry:
        featurizers.append(Stoichiometry())
    if use_valence:
        featurizers.append(ValenceOrbital(props=["avg"]))

    # ------------------------------------------------------------------
    # 3. Iterate over rows
    # NOTE: Row-by-row loop — acceptable for N < ~10k rows. For larger
    # datasets, consider batched featurization via Composition batch methods.
    # ------------------------------------------------------------------
    feature_list: List[List[float]] = []
    skipped_rows: List[int] = []
    ok_indices: List[int] = []

    for idx, row in df.iterrows():
        # 3a. Build mass-fraction dict for this row
        mass_dict = {element_map[col]: float(row[col]) for col in mass_frac_cols}

        # 3b. Convert to pymatgen Composition
        try:
            comp = mass_fractions_to_composition(
                mass_dict, normalize=True, add_al_balance=add_al_balance
            )
        except Exception:
            skipped_rows.append(idx)
            continue

        # 3c. Apply each featurizer
        row_feats: List[float] = []
        for fz in featurizers:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vals = fz.featurize(comp)
            # Some featurizers may return NaN for exotic cases; keep as-is
            # (the caller can later impute).
            row_feats.extend(vals)

        # 3d. Optionally append raw mass fractions
        if keep_raw_frac:
            row_feats.extend(float(row[col]) for col in mass_frac_cols)

        # 3e. Append process features
        for pc in process_cols:
            row_feats.append(float(row[pc]))

        feature_list.append(row_feats)
        ok_indices.append(idx)

    if skipped_rows:
        warnings.warn(
            f"{len(skipped_rows)} row(s) skipped during featurize_metal_df() "
            f"due to pymatgen Composition conversion failure. "
            f"Affected indices: {skipped_rows}. "
            f"These rows will have zero-filled Magpie/Stoich/Valence features "
            f"in the final output (filled by featurize_metal_all). "
            f"Check input data for NaN, negative, or all-zero mass fractions."
        )

    # ------------------------------------------------------------------
    # 4. Build column names
    # ------------------------------------------------------------------
    col_names: List[str] = []

    if use_magpie:
        magpie_labels = ElementProperty.from_preset("magpie").feature_labels()
        col_names.extend(magpie_labels)
    if use_stoichiometry:
        stoich_labels = Stoichiometry().feature_labels()
        col_names.extend(stoich_labels)
    if use_valence:
        valence_labels = ValenceOrbital(props=["avg"]).feature_labels()
        col_names.extend(valence_labels)
    if keep_raw_frac:
        col_names.extend(mass_frac_cols)
    col_names.extend(process_cols)

    # ------------------------------------------------------------------
    # 5. Assemble output
    # ------------------------------------------------------------------
    X = pd.DataFrame(feature_list, columns=col_names, index=ok_indices)

    # Handle potential NaN from matminer featurizers
    if X.isna().any().any():
        na_cols = X.columns[X.isna().any()].tolist()
        print(f"NOTE: NaN values detected in columns {na_cols}. "
              f"Filling with column median.")
        X = X.fillna(X.median())

    # Determine target column(s)
    if target_cols is not None:
        y_cols = target_cols
    elif target_col is not None:
        y_cols = [target_col]
    else:
        y_cols = []

    if y_cols:
        y = df.loc[ok_indices, y_cols].reset_index(drop=True)
        y.columns = y_cols
    else:
        y = pd.DataFrame(index=X.index)

    return X, y, col_names


# ---------------------------------------------------------------------------
# Convenience: manual alloy thermodynamic descriptors (optional)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DNNRegressor: sklearn-compatible PyTorch neural network (used in N01 MICE)
# ---------------------------------------------------------------------------

class DNNRegressor:
    """
    sklearn-compatible PyTorch neural network regressor.
    Used as a candidate estimator in MICE imputation (N01).

    Parameters
    ----------
    hidden_dims : list of int
        Hidden layer sizes (default [16] — single layer, ~2500 params).
    lr : float
        Learning rate for Adam optimizer (default 1e-3).
    epochs : int
        Max training epochs (default 300; early stopping may stop sooner).
    batch_size : int
        Mini-batch size (default 32 — smaller batches add noise = regularization).
    dropout : float
        Dropout rate (default 0.15 — helps prevent overfitting on small data).
    patience : int
        Early stopping patience (default 30 — stop if val loss doesn't improve
        for 30 consecutive epochs).
    verbose : bool
        Print training progress.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        hidden_dims=None,
        lr=1e-3,
        epochs=300,
        batch_size=32,
        dropout=0.15,
        patience=30,
        verbose=False,
        random_state=42,
    ):
        if hidden_dims is None:
            hidden_dims = [16]
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state

    def _build_model(self, input_dim, output_dim):
        import torch
        import torch.nn as nn

        class _Network(nn.Module):
            def __init__(self, in_dim, hidden, out_dim, dropout):
                super().__init__()
                def _fc_block(d_in, d_out):
                    return nn.Sequential(
                        nn.Linear(d_in, d_out),
                        nn.Dropout(p=dropout),
                        nn.LeakyReLU(),
                    )
                layers = [
                    nn.Sequential(
                        nn.Linear(in_dim, hidden[0]),
                        nn.Dropout(p=dropout),
                        nn.LeakyReLU(),
                    )
                ]
                for a, b in zip(hidden[:-1], hidden[1:]):
                    layers.append(_fc_block(a, b))
                layers.append(nn.Linear(hidden[-1], out_dim))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        return _Network(input_dim, self.hidden_dims, output_dim, self.dropout)

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        # Set seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

        n_samples, n_features = X_np.shape
        n_targets = y_np.shape[1]

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Scale targets per column
        y_tensor = torch.as_tensor(y_np, dtype=torch.float32)
        self._y_mean = torch.mean(y_tensor, dim=0)
        self._y_std = torch.std(y_tensor, dim=0)
        self._y_std = torch.where(self._y_std < 1e-8, torch.ones_like(self._y_std), self._y_std)
        y_scaled = (y_tensor - self._y_mean) / self._y_std

        # Build model
        self.model_ = self._build_model(n_features, n_targets).to(device)

        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        # ── Internal validation split for early stopping ──
        n_val = max(1, int(n_samples * 0.15))
        n_tr = n_samples - n_val
        indices = np.arange(n_samples)
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(indices)
        tr_idx, va_idx = indices[:n_tr], indices[n_tr:]
        X_tr = torch.as_tensor(X_np[tr_idx], dtype=torch.float32)
        y_tr = y_scaled[tr_idx]
        X_va = torch.as_tensor(X_np[va_idx], dtype=torch.float32).to(device)
        y_va = y_scaled[va_idx].to(device)

        tr_loader = DataLoader(TensorDataset(X_tr, y_tr),
                               batch_size=min(self.batch_size, n_tr), shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        self.model_.train()
        for epoch in range(self.epochs):
            # Training
            train_loss = 0.0
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self.model_(Xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * Xb.size(0)

            # Validation (no gradient)
            self.model_.eval()
            with torch.no_grad():
                val_loss = criterion(self.model_(X_va), y_va).item()
            self.model_.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if self.verbose:
                        print(f"  DNN early stop @ epoch {epoch+1}, best_val_loss={best_val_loss:.4f}")
                    break

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"  DNN epoch {epoch+1}/{self.epochs}, "
                      f"train_loss={train_loss/n_tr:.4f}, val_loss={val_loss:.4f}")

        # Restore best weights
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(self, X):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_np = np.asarray(X, dtype=np.float32)

        self.model_.eval()
        with torch.no_grad():
            Xt = torch.as_tensor(X_np, dtype=torch.float32).to(device)
            pred_scaled = self.model_(Xt).cpu()
        pred = pred_scaled * self._y_std + self._y_mean
        result = pred.numpy()
        if result.shape[1] == 1:
            result = result.ravel()
        return result

    def get_params(self, deep=True):
        return {
            "hidden_dims": self.hidden_dims,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "patience": self.patience,
            "verbose": self.verbose,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self


# ============================================================================
# Unified feature engineering for wrought aluminum alloys
# ============================================================================
# 4 groups, 330 features (v3 final: Delta-to-Al moved to D for El isolation)
#
# Data sources: pymatgen Element + matminer MagpieData + CRC Handbook fallback.
#
# Group A: Composition (71 features) — "what the alloy IS"
#   sol_ x14     raw solute mass fractions
#   mag_ x30     Magpie (15 props x 2 stats: mean+range)
#   stoich x7    matminer Stoichiometry
#   valence x4   matminer ValenceOrbital
#   gbl_ x10     global composition statistics
#   orb_ x6      electronic orbital features
#   cls_ x7      alloy family classification (2xxx/6xxx/7xxx/8xxx)
#   sol_ x4      solid solution strengthening (Labusch, size/modulus misfit, Fleischer)
#
# Group B: Precipitation (198 features) — "what happens during AGING"
#   int_ x14     named metallurgical interactions
#   int_pair_ x91 all solute pair products (C(14,2)=91)
#   int_ratio_ x14 solute-to-Al ratios
#   ppt_ x13     precipitation potentials + excess_Si + excess_Zn
#   ppt_ x8      precipitation kinetics
#   cxp_ x45     Composition x Process for all 14 elements
#   wrt_ x13     wrought-specific descriptors
#
# Group C: Process (16 features) — "how it's MADE"
#   prc_ x3      raw process parameters (Tsol, Tage, tage)
#   prc_ x13     process engineering transforms (Larson-Miller, etc.)
#
# Group D: Physics (45 features) — "fundamental PHYSICAL properties"
#   thm_ x9      thermodynamic descriptors (Miedema mixing enthalpy)
#   el_ x7       elastic modulus mismatch
#   kin_ x5      kinetic/diffusion features
#   fal_ x8      full-alloy properties (archive v7: includes Al in averages)
#   delta_ x16   delta-to-Al features (archive v7 B_delta_to_Al — KEY for El)
#
# Hardcoded sources:
#   Density/Bulk/Shear/Boiling/Fusion/Vaporization: CRC Handbook (97th ed.)
#   Covalent radius/Electron affinity: CRC Handbook
#   Larson-Miller C=20: ASM Handbook Vol.4 (1991)
#   exp(-T/573): Hatch, Aluminum, ASM (1984)
#   Solidus coefficients: ASM Handbook Vol.3 (1992)
#   Wrought features: Nes (2002), Morinaga (1993), Schulthess (2009)
#   Labusch/Fleischer: Gypen & Deruyttere (1977), Fleischer (1963), Tiryakioğlu (2010)
#   Full-alloy features adapted from archive v7 (build_features, K1-K8)
# ============================================================================

# ---------------------------------------------------------------------------
# Helper: build extended element property table
# ---------------------------------------------------------------------------

def _build_element_database(element_symbols: List[str]) -> pd.DataFrame:
    """
    Unified element property database. 30+ properties per element.

    Data priority (cascading fallback):
    1. pymatgen Element (ab-initio + curated crystallographic data)
    2. matminer MagpieData (element property statistics for ML)
    3. Literature values from CRC Handbook 97th Ed., Smithells 8th Ed.,
       Cordero et al. (2008), Hotop & Lineberger (1985), NIST ASD (2022).
       See _fill_missing_props() for per-property citations.

    Returns DataFrame indexed by element symbol with 30+ property columns.
    """
    from pymatgen.core import Element as PymElement

    records = []
    for sym in element_symbols:
        el = PymElement(sym)
        rec = {
            'element': sym,
            'atomic_num': el.Z,
            'atomic_weight': el.atomic_mass,
            'group': el.group,
            'row': el.row,
            'VEC': _get_vec(sym),
            'chi': el.X if el.X is not None else np.nan,
            'atomic_radius': el.atomic_radius if el.atomic_radius is not None else np.nan,
            'atomic_volume': _get_atomic_volume(sym),
            'Tm': el.melting_point if el.melting_point is not None else np.nan,
        }

        # Try to get additional properties from MagpieData
        try:
            from matminer.utils.data import MagpieData
            mg = MagpieData()
            props_to_get = [
                'Density', 'BulkModulus', 'ShearModulus', 'BoilingT',
                'HeatFusion', 'HeatVaporization', 'FirstIonizationEnergy',
                'CovalentRadius', 'ElectronAffinity',
            ]
            for prop in props_to_get:
                try:
                    # MagpieData has per-element access
                    val = mg.get_elemental_property(el.Z, prop)
                    rec[prop.lower()] = float(val) if val is not None and not np.isnan(float(val)) else np.nan
                except Exception:
                    rec[prop.lower()] = np.nan
        except ImportError:
            pass

        # Fill gaps with hardcoded values for common Al alloy elements
        _fill_missing_props(rec)

        records.append(rec)

    return pd.DataFrame(records).set_index('element')


def _get_vec(sym: str) -> float:
    """Valence electron count: total s+d electrons for transition metals,
    s+p electrons for main-group elements (ground-state free-atom configuration).

    Reference: NIST Atomic Spectra Database (ver. 5.10, 2022)
    https://www.nist.gov/pml/atomic-spectra-database
    For transition metals, s+d count follows the Hume-Rothery convention
    (Hume-Rothery, W., The Metallic State, Oxford University Press, 1931).
    """
    vec_map = {
        'Al': 3, 'Si': 4, 'Fe': 8, 'Cu': 11, 'Mn': 7, 'Mg': 2, 'Cr': 6,
        'Zn': 12, 'V': 5, 'Ti': 4, 'Zr': 4, 'Li': 1, 'Ni': 10, 'Be': 2,
        'Sc': 3, 'C': 4, 'N': 5, 'P': 5, 'S': 6, 'B': 3,
        'Mo': 6, 'Co': 9, 'W': 6, 'Nb': 5, 'Ca': 2, 'Sr': 2,
        'Ag': 11, 'Sn': 4, 'Pb': 4, 'Bi': 5, 'Sb': 5, 'Cd': 12,
        'In': 3, 'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Te': 6,
        'Pd': 10, 'Pt': 10, 'Au': 11, 'Hg': 12, 'Tl': 3, 'Ba': 2,
        'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1, 'Hf': 4, 'Ta': 5,
        'Re': 7, 'Os': 8, 'Ir': 9, 'Ru': 8, 'Rh': 9, 'Y': 3,
        'La': 3, 'Ce': 4, 'Nd': 3, 'Sm': 3, 'Gd': 3, 'Dy': 3,
    }
    return vec_map.get(sym, 0)


def _get_atomic_volume(sym: str) -> float:
    """Atomic volume (cm³/mol) at 293 K, calculated from atomic weight / density.

    Reference: CRC Handbook of Chemistry and Physics, 97th Edition (2016-2017),
    W.M. Haynes (Ed.), CRC Press, Boca Raton, FL.
    - Density values: Table "Density of Elements" (Section 4, pp. 4-12 to 4-15)
    - Atomic weights: Table "Atomic Weights and Isotopic Compositions" (Section 1, pp. 1-11 to 1-14)
    Atomic volume = atomic_weight / density.

    Values verified against WebElements (https://www.webelements.com, Mark Winter,
    University of Sheffield, accessed 2024) for consistency.
    """
    av = {
        'Al': 10.0, 'Si': 12.1, 'Fe': 7.1, 'Cu': 7.1, 'Mn': 7.4, 'Mg': 14.0,
        'Cr': 7.2, 'Zn': 9.2, 'V': 8.3, 'Ti': 10.6, 'Zr': 14.0, 'Li': 13.1, 'Ni': 6.6, 'Be': 5.0, 'Sc': 15.0, 'C': 5.3, 'N': 17.3, 'P': 17.0,
        'S': 15.5, 'B': 4.6, 'Mo': 9.4, 'Co': 6.7, 'W': 9.5, 'Nb': 10.8,
        'Sn': 16.3, 'Pb': 18.3, 'Ca': 29.9, 'Ag': 10.3, 'Au': 10.2,
        'Pt': 9.1, 'Pd': 8.9, 'Cd': 13.1, 'In': 15.7, 'Sb': 18.2,
        'Bi': 21.3, 'Hg': 14.8, 'Tl': 17.2,
    }
    return av.get(sym, 10.0)


def _fill_missing_props(rec: dict):
    """Fill missing element property values from authoritative literature tables.

    These values serve as fallbacks when pymatgen and matminer MagpieData
    cannot provide a value (e.g., exotic elements, missing database entries).

    References (in priority order per property):
    - [CRC]  CRC Handbook of Chemistry and Physics, 97th Ed. (2016-2017),
             W.M. Haynes (Ed.), CRC Press.
             - density: Section 4, Table "Density of Elements" (pp. 4-12 to 4-15)
             - boilingt: Section 4, Table "Boiling Points of the Elements"
             - heatfusion: Section 4, Table "Enthalpy of Fusion of the Elements"
             - firstionizationenergy: Section 10, Table "Ionization Energies of
               Atoms and Atomic Ions" (pp. 10-201 to 10-205)
    - [SMT]  Smithells Metals Reference Book, 8th Ed. (2004), W.F. Gale &
             T.C. Totemeier (Eds.), Elsevier Butterworth-Heinemann.
             - bulkmodulus, shearmodulus: Chapter 15, Table 15.1
               "Elastic Constants of Metals" (pp. 15-5 to 15-9)
    - [COR]  Cordero, B. et al., Dalton Transactions (2008) 2832-2838.
             DOI: 10.1039/B801115J
             - covalentradius: Table 1 "Covalent radii revisited"
    - [HOT]  Hotop, H. & Lineberger, W.C., J. Phys. Chem. Ref. Data 14 (1985)
             731-750. DOI: 10.1063/1.555749
             - electronaffinity: Table II "Electron Affinities of Atoms"
             (Values for Mg, Zn, Be = 0.0 per experimental upper bounds)
    - [NIST] NIST Atomic Spectra Database (ver. 5.10, 2022).
             https://www.nist.gov/pml/atomic-spectra-database
             - firstionizationenergy: verified against NIST values
    """
    sym = rec['element']
    fallback = {
        'Al': {'density': 2.70, 'bulkmodulus': 76, 'shearmodulus': 26, 'boilingt': 2792,
               'heatfusion': 10.71, 'firstionizationenergy': 5.986, 'covalentradius': 125, 'electronaffinity': 0.441},
        'Si': {'density': 2.33, 'bulkmodulus': 100, 'shearmodulus': 51, 'boilingt': 3173,
               'heatfusion': 50.21, 'firstionizationenergy': 8.152, 'covalentradius': 111, 'electronaffinity': 1.385},
        'Fe': {'density': 7.87, 'bulkmodulus': 170, 'shearmodulus': 82, 'boilingt': 3134,
               'heatfusion': 13.81, 'firstionizationenergy': 7.902, 'covalentradius': 125, 'electronaffinity': 0.163},
        'Cu': {'density': 8.96, 'bulkmodulus': 140, 'shearmodulus': 48, 'boilingt': 2835,
               'heatfusion': 13.26, 'firstionizationenergy': 7.726, 'covalentradius': 128, 'electronaffinity': 1.228},
        'Mn': {'density': 7.47, 'bulkmodulus': 120, 'shearmodulus': 47, 'boilingt': 2334,
               'heatfusion': 12.91, 'firstionizationenergy': 7.434, 'covalentradius': 127, 'electronaffinity': 0.0},
        'Mg': {'density': 1.74, 'bulkmodulus': 45, 'shearmodulus': 17, 'boilingt': 1363,
               'heatfusion': 8.48, 'firstionizationenergy': 7.646, 'covalentradius': 141, 'electronaffinity': 0.0},
        'Cr': {'density': 7.19, 'bulkmodulus': 160, 'shearmodulus': 115, 'boilingt': 2944,
               'heatfusion': 21.0, 'firstionizationenergy': 6.767, 'covalentradius': 128, 'electronaffinity': 0.666},
        'Zn': {'density': 7.14, 'bulkmodulus': 70, 'shearmodulus': 42, 'boilingt': 1180,
               'heatfusion': 7.32, 'firstionizationenergy': 9.394, 'covalentradius': 131, 'electronaffinity': 0.0},
        'V':  {'density': 6.11, 'bulkmodulus': 160, 'shearmodulus': 47, 'boilingt': 3680,
               'heatfusion': 21.5, 'firstionizationenergy': 6.746, 'covalentradius': 134, 'electronaffinity': 0.525},
        'Ti': {'density': 4.51, 'bulkmodulus': 110, 'shearmodulus': 44, 'boilingt': 3560,
               'heatfusion': 14.15, 'firstionizationenergy': 6.828, 'covalentradius': 136, 'electronaffinity': 0.079},
        'Zr': {'density': 6.51, 'bulkmodulus': 95, 'shearmodulus': 33, 'boilingt': 4682,
               'heatfusion': 14.0, 'firstionizationenergy': 6.634, 'covalentradius': 148, 'electronaffinity': 0.426},
        'Li': {'density': 0.53, 'bulkmodulus': 11, 'shearmodulus': 3, 'boilingt': 1615,
               'heatfusion': 3.0, 'firstionizationenergy': 5.392, 'covalentradius': 134, 'electronaffinity': 0.618},
        'Ni': {'density': 8.91, 'bulkmodulus': 180, 'shearmodulus': 76, 'boilingt': 3186,
               'heatfusion': 17.48, 'firstionizationenergy': 7.640, 'covalentradius': 124, 'electronaffinity': 1.156},
        'Be': {'density': 1.85, 'bulkmodulus': 130, 'shearmodulus': 132, 'boilingt': 3243,
               'heatfusion': 12.2, 'firstionizationenergy': 9.323, 'covalentradius': 96, 'electronaffinity': 0.0},
        'Sc': {'density': 2.99, 'bulkmodulus': 57, 'shearmodulus': 29, 'boilingt': 3109,
               'heatfusion': 14.1, 'firstionizationenergy': 6.561, 'covalentradius': 144, 'electronaffinity': 0.188},
    }
    if sym in fallback:
        for k, v in fallback[sym].items():
            if np.isnan(rec.get(k, np.nan)): rec[k] = v

    # Rename to internal property codes for consistency
    for old_k, new_k in [('shearmodulus','G_mod'),('bulkmodulus','B_mod'),('firstionizationenergy','IE1'),
                          ('electronaffinity','EA'),('covalentradius','r_cov'),('boilingt','Tb'),
                          ('heatfusion','heat_fus')]:
        if old_k in rec and np.isnan(rec.get(new_k, np.nan)): rec[new_k] = rec[old_k]


# ---------------------------------------------------------------------------
# Category functions
# ---------------------------------------------------------------------------

def features_wrought(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
) -> pd.DataFrame:
    """
    Wrought/forged aluminum alloy specific features (13 features).

    References:
    - Zener pinning: Nes, E. et al., Prog. Mater. Sci. 47 (2002) 163-282.
      DOI: 10.1016/S0079-6425(01)00005-6
    - SFE proxy: Morinaga, M. & Kamado, S., Modelling Simul. Mater. Sci. Eng.
      1 (1993) 151. Also: Schulthess, T. et al., Acta Mater. 57 (2009)
      1396-1406 — Al SFE = 120-166 mJ/m² from DFT.
    - Solidus depression (ΔT per wt fraction):
      Binary Al-X phase diagrams from ASM Handbook Vol. 3: Alloy Phase Diagrams
      (1992), ASM International, Materials Park, OH.
      - Si: ~600 °C/wt-frac (Al-Si eutectic at 577 °C, 12.6 wt% Si)
      - Cu: ~340 °C/wt-frac (Al-Cu eutectic at 548 °C, 33.2 wt% Cu)
      - Mg: ~450 °C/wt-frac (Al-Mg eutectic at 450 °C, 35 wt% Mg)
      Values are linear approximations near the Al-rich end of each binary.
    """
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)

    def _v(sym):
        for c, e in element_map.items():
            if e == sym: return df[c].values
        return np.zeros(len(df))

    Cr, Mn, Zr = _v('Cr'), _v('Mn'), _v('Zr')
    Sc, V, Ti = _v('Sc'), _v('V'), _v('Ti')
    Ni, Cu, Fe, Si = _v('Ni'), _v('Cu'), _v('Fe'), _v('Si')
    Mg, Zn, Li = _v('Mg'), _v('Zn'), _v('Li')

    # Zener pinning: dispersoid formers slow grain boundary migration
    out['wrt_zener_pinning'] = Cr + Mn + Zr + Sc + V

    # Recrystallization resistance via fine dispersoids
    out['wrt_rex_resistance'] = Cr + Mn + Zr

    # SFE proxy: Cu, Ni, Zn lower stacking fault energy in Al
    # Lower SFE = wider stacking faults, easier planar slip
    out['wrt_sfe_proxy'] = Cu + Ni + Zn

    # (wrt_age_hardening removed — overlaps with ppt_S_phase_potential)
    # (wrt_solute_drag removed — overlaps with wrt_zener_pinning)

    # Solidus depression from binary phase diagrams (ASM 1992)
    # Delta_T per wt fraction for key solutes
    out['wrt_solidus_depression'] = Si * 600 + Cu * 340 + Mg * 450

    # Si/Mg ratio for 6xxx series wrought alloys
    out['wrt_Si_over_Mg'] = Si / np.maximum(Mg, 1e-9)

    # Total transition metal (affects quench sensitivity)
    out['wrt_transition_metal_total'] = Cr + Mn + Fe + Zr + V + Ti

    # ── Ductility / elongation specific features ──
    # Total solute content (higher solute → more obstacles → lower ductility)
    out['wrt_total_solute'] = (Si + Fe + Cu + Mn + Mg + Cr + Zn +
                               V + Ti + Zr + Li + Ni)

    # Mg+Si+Cu combined (major strengthening solutes in 2xxx/6xxx/7xxx)
    out['wrt_major_strengtheners'] = Mg + Si + Cu

    # Fe+Si intermetallic formers (brittle particles → reduce ductility)
    out['wrt_brittle_intermetallic'] = Fe + Si

    # Mn+Cr+Zr dispersoid formers (refine grain → improve ductility)
    out['wrt_dispersoid_total'] = Mn + Cr + Zr

    # Zn/Mg ratio (7xxx series indicator, affects η vs T phase balance)
    out['wrt_Zn_over_Mg'] = Zn / np.maximum(Mg, 1e-9)

    # Cu/Mg ratio (2xxx vs 6xxx series indicator)
    out['wrt_Cu_over_Mg'] = Cu / np.maximum(Mg, 1e-9)

    # Purity indicator (lower Fe+Si = higher purity = better ductility)
    out['wrt_purity_index'] = 1.0 - np.minimum(Fe + Si, 0.99)

    return out



def features_input(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
    process_cols: List[str],
) -> pd.DataFrame:
    """sol_ + prc_: Raw composition + process parameters (17 features)."""
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)
    for col, elem in element_map.items():
        out[f'sol_{elem}'] = df[col].values
    for pc in process_cols:
        out[f'prc_{pc}'] = df[pc].values
    return out


def features_interaction(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
) -> pd.DataFrame:
    """B_interaction: Metallurgical interactions + pair products (~14 features).

    No X/Al ratios — for Al-base alloys (Al ~85-99 wt%), X/Al ≈ X and these
    are nearly collinear with raw solute fractions. Pair products (e.g. Mg×Si)
    replace them — these are the actual drivers of precipitate volume fraction.
    """
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)

    def _v(sym):
        for c, e in element_map.items():
            if e == sym:
                return df[c].values
        return np.zeros(len(df))

    eps = 1e-9

    Mg, Si, Cu, Zn = _v('Mg'), _v('Si'), _v('Cu'), _v('Zn')
    Mn, Fe, Cr, Zr = _v('Mn'), _v('Fe'), _v('Cr'), _v('Zr')
    Ti, Sc, Be = _v('Ti'), _v('Sc'), _v('Be')

    # --- Named metallurgical interactions ---
    out['int_Cu_sq'] = Cu ** 2
    out['int_Mg_sq'] = Mg ** 2
    out['int_Zn_over_Mg'] = np.divide(Zn, np.maximum(Mg, eps))
    out['int_Cu_over_MgZn'] = np.divide(Cu, np.maximum(Mg + Zn, eps))
    out['int_grain_refiner_total'] = Zr + Ti + Sc
    out['int_Fe_over_Si'] = np.divide(Fe, np.maximum(Si, eps))
    out['int_Mn_over_FeSi'] = np.divide(Mn, np.maximum(Fe + Si, eps))
    out['int_MgZn_over_Cu'] = np.divide(Mg + Zn, np.maximum(Cu, eps))
    out['int_Sc_over_Zr'] = np.divide(Sc, np.maximum(Sc + Zr, eps))
    out['int_Be_times_MgZn'] = Be * (Mg + Zn)

    # --- Pair products: precipitate volume-fraction drivers ---
    # These were previously removed as "noise on small data," but Mg×Si,
    # Mg×Zn, Cu×Mg are the direct drivers of β'', η', and S-phase respectively —
    # the three dominant strengthening precipitate systems in Al alloys.
    out['int_MgSi_product'] = Mg * Si       # Mg2Si / β'' driver (6xxx)
    out['int_MgZn_product'] = Mg * Zn       # MgZn2 / η' driver (7xxx)
    out['int_CuMg_product'] = Cu * Mg       # Al2CuMg / S-phase driver (2xxx)
    out['int_ZnCu_product'] = Zn * Cu       # Zn-Cu synergism in 7xxx

    # --- All solute pair products (archive v7: D_pair_X_times_Y) ---
    # C(14,2) = 91 pairs. Metallurgically meaningful examples for El:
    # Fe×Si → AlFeSi brittle intermetallics, Fe×Mn → dispersoids, etc.
    # The archive found these necessary; N03 LassoCV selects the useful ones.
    from itertools import combinations
    for a, b in combinations(list(element_map.values()), 2):
        out[f'int_pair_{a}_x_{b}'] = _v(a) * _v(b)

    # --- Solute-to-Al ratios (archive v7: D_ratio_X_to_Al) ---
    al_balance = np.maximum(1.0 - df[mass_frac_cols].sum(axis=1).values, eps)
    for s in list(element_map.values()):
        out[f'int_ratio_{s}_to_Al'] = _v(s) / al_balance

    return out


def features_thermo(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
    miedema_matrix: Optional[pd.DataFrame] = None,
    add_al_balance: bool = True,
) -> pd.DataFrame:
    """D_thermo: Miedema-model thermodynamic descriptors (9 features).

    NOTE: Row-by-row loop — acceptable for N < ~10k rows. For larger datasets,
    consider vectorizing the weighted-average computations via numpy.einsum.
    """
    element_map = _extract_element_map(mass_frac_cols)
    prop_df = _build_element_database(list(element_map.values()) + ['Al'])
    R = 8.314

    records = []
    for _, row in df.iterrows():
        mass_dict = {element_map[col]: float(row[col]) for col in mass_frac_cols}
        if add_al_balance:
            s_sum = sum(mass_dict.values())
            mass_dict['Al'] = max(0.0, 1.0 - s_sum)

        total = sum(mass_dict.values())
        if total <= 0:
            records.append({f'thm_{k}': np.nan for k in ['dH','dS','VEC','delta','dChi','Omega','logOm','Tm','SS']})
            continue

        w = {k: v / total for k, v in mass_dict.items()}
        elems = list(w.keys())
        weights = np.array([w[e] for e in elems])

        # Property vectors
        get_p = lambda p: np.array([
            prop_df.loc[e, p]
            if e in prop_df.index and not pd.isna(prop_df.loc[e, p]) else np.nan
            for e in elems])

        chi = get_p('chi')
        r_atom = get_p('atomic_radius')
        vec = get_p('VEC')
        tm = get_p('Tm')

        # D2: entropy of mixing
        valid = weights > 0
        delta_S = -R * np.sum(weights[valid] * np.log(weights[valid]))

        # D1: mixing enthalpy (real Miedema or 0)
        delta_H = 0.0
        if miedema_matrix is not None:
            n = len(elems)
            for i in range(n):
                for j in range(i + 1, n):
                    e1, e2 = elems[i], elems[j]
                    if e1 in miedema_matrix.index and e2 in miedema_matrix.columns:
                        h = miedema_matrix.loc[e1, e2]
                        if h != 0:
                            delta_H += 4.0 * float(h) * weights[i] * weights[j]

        # D3: mixed VEC
        vec_clean = vec[~np.isnan(vec)]
        w_clean = weights[~np.isnan(vec)]
        D3 = float(np.average(vec_clean, weights=w_clean)) if len(w_clean) > 0 else np.nan

        # D4: atomic size mismatch (delta x 100)
        r_clean = r_atom[~np.isnan(r_atom)]
        w_r = weights[~np.isnan(r_atom)]
        if len(w_r) > 1:
            r_mean = np.average(r_clean, weights=w_r)
            D4 = 100.0 * np.sqrt(np.sum(w_r * (1.0 - r_clean / r_mean) ** 2))
        else:
            D4 = 0.0

        # D5: electronegativity mismatch
        chi_clean = chi[~np.isnan(chi)]
        w_chi = weights[~np.isnan(chi)]
        if len(w_chi) > 1:
            chi_mean = np.average(chi_clean, weights=w_chi)
            D5 = np.sqrt(np.sum(w_chi * (chi_clean - chi_mean) ** 2))
        else:
            D5 = 0.0

        # D8: mixed melting temperature
        tm_clean = tm[~np.isnan(tm)]
        w_tm = weights[~np.isnan(tm)]
        D8 = float(np.average(tm_clean, weights=w_tm)) if len(w_tm) > 0 else np.nan

        # D6: Omega = Tm * dS / |dH|
        eps = 1e-9
        D6 = D8 * delta_S / (abs(delta_H) + eps) if not np.isnan(D8) else np.nan
        # D7: log Omega
        D7 = np.log(max(D6 + eps, eps)) if not np.isnan(D6) else np.nan
        # D9: solid solution parameter
        D9 = D4 * abs(delta_H) / (D8 * delta_S + eps) if not np.isnan(D8) else np.nan

        records.append({
            'thm_dH': delta_H,
            'thm_dS': delta_S,
            'thm_VEC': D3,
            'thm_delta': D4,
            'thm_dChi': D5,
            'thm_Omega': D6,
            'thm_logOm': D7,
            'thm_Tm': D8,
            'thm_SS': D9,
        })

    return pd.DataFrame(records)


def features_process(
    df: pd.DataFrame,
    process_cols: List[str],
) -> pd.DataFrame:
    """C_process: Process engineering transforms (13 features)."""
    out = pd.DataFrame(index=df.index)
    if 'Tage' not in df.columns or 'tage' not in df.columns:
        warnings.warn(
            "features_process: 'Tage' or 'tage' column missing — "
            "all 13 process transform features will be empty. "
            "Ensure process_cols=['Tsol','Tage','tage'] is passed."
        )
        return out

    Tage = df['Tage'].values
    tage = df['tage'].values
    Tage_K = Tage + 273.15
    eps = 1e-9

    out['prc_log_tage'] = np.log(np.maximum(tage, eps))
    out['prc_sqrt_tage'] = np.sqrt(np.maximum(tage, 0.0))
    out['prc_inv_Tage_K'] = 1.0 / Tage_K
    # Larson-Miller parameter: C=20 for Al alloys (ASM Handbook Vol.4, 1991)
    out['prc_larson_miller'] = Tage_K * (np.log10(np.maximum(tage, eps)) + 20.0)
    out['prc_Tage_log_tage'] = Tage * np.log(np.maximum(tage, eps))
    out['prc_Tage_sq'] = Tage ** 2
    out['prc_Tsol_sq'] = df['Tsol'].values ** 2 if 'Tsol' in df.columns else 0
    if 'Tsol' in df.columns:
        Tsol = df['Tsol'].values
        out['prc_Tsol_minus_Tage'] = Tsol - Tage
        out['prc_deltaT_log_tage'] = (Tsol - Tage) * np.log(np.maximum(tage, eps))
        out['prc_Tsol_Tage'] = Tsol * Tage
    # Diffusion reference at 300 C (573 K): typical Al aging temperature baseline
    # Source: J.E. Hatch, Aluminum: Properties and Physical Metallurgy, ASM (1984), Ch.5
    out['prc_exp_neg_Tage'] = np.exp(-Tage_K / 573.0)
    out['prc_inv_tage'] = 1.0 / np.maximum(tage, eps)
    out['prc_log_Tsol_K'] = np.log(Tsol + 273.15) if 'Tsol' in df.columns else 0

    return out


def features_global(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
    process_cols: List[str],
) -> pd.DataFrame:
    """G_global: Global composition statistics (10 features)."""
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)

    vals = df[mass_frac_cols].values
    out['gbl_total_solute'] = vals.sum(axis=1)
    out['gbl_active_solute_count'] = (vals > 0.0005).sum(axis=1)  # >0.05 wt% = 0.0005 fraction
    out['gbl_max_solute'] = vals.max(axis=1)
    # Second largest
    part = -np.partition(-vals, 1, axis=1)
    out['gbl_second_largest'] = part[:, 1] if vals.shape[1] > 1 else 0
    out['gbl_solute_std'] = vals.std(axis=1)

    def _v(sym):
        for c, e in element_map.items():
            if e == sym:
                return df[c].values
        return np.zeros(len(df))
    Mg, Zn, Cu, Si = _v('Mg'), _v('Zn'), _v('Cu'), _v('Si')
    Fe, Zr, Ti, Sc = _v('Fe'), _v('Zr'), _v('Ti'), _v('Sc')

    out['gbl_main_hardener_total'] = Mg + Zn + Cu + Si
    out['gbl_grain_refiner_x_Tsol'] = (Zr + Ti + Sc) * df.get('Tsol', pd.Series(500, index=df.index)).values
    out['gbl_impurity_total'] = Fe + Si
    out['gbl_MgZn_total'] = Mg + Zn

    # Equivalent solute density (mass-fraction weighted)
    props = _build_element_database(list(element_map.values()))

    def _get_density(elem_symbol):
        """Return element density with fallback to 7.0 g/cm³."""
        if elem_symbol in props.index:
            val = props.loc[elem_symbol, 'density']
            if not pd.isna(val):
                return float(val)
        return 7.0  # fallback for elements missing from database

    density_vec = np.array([_get_density(e) for e in element_map.values()])
    out['gbl_equiv_solute_density'] = np.sum(vals * density_vec[None, :], axis=1) / np.maximum(
        out['gbl_total_solute'].values, 1e-9)

    return out


def features_elastic(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
) -> pd.DataFrame:
    """el_elastic: Elastic modulus mismatch features (7 features).

    NOTE: Row-by-row loop — acceptable for N < ~10k rows. For larger datasets,
    consider vectorizing via pre-computed element-property arrays.
    """
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)
    props = _build_element_database(list(element_map.values()) + ['Al'])

    def get_p(sym, prop):
        if sym not in props.index: return np.nan
        v = props.loc[sym, prop]
        return np.nan if pd.isna(v) else float(v)

    eps = 1e-9

    for idx in df.index:
        mass_dict = {element_map[col]: float(df[col].loc[idx]) for col in mass_frac_cols}
        s_sum = sum(mass_dict.values())
        mass_dict['Al'] = max(0.0, 1.0 - s_sum)
        total = max(sum(mass_dict.values()), eps)
        w = {k: v / total for k, v in mass_dict.items()}
        elems = list(w.keys())
        weights = np.array([w[e] for e in elems])

        G = np.array([get_p(e, 'shearmodulus') for e in elems])
        B = np.array([get_p(e, 'bulkmodulus') for e in elems])
        dens = np.array([get_p(e, 'density') for e in elems])

        # Use approximation: E ≈ 9BG/(3B+G) if not available
        g_valid, b_valid = ~np.isnan(G), ~np.isnan(B)
        G_fill = np.where(g_valid, G, 40)
        B_fill = np.where(b_valid, B, 80)
        E_calc = 9 * B_fill * G_fill / (3 * B_fill + G_fill + eps)

        G_mean = np.average(G_fill, weights=weights)
        B_mean = np.average(B_fill, weights=weights)
        E_mean = np.average(E_calc, weights=weights)
        dens_mean = np.average(np.where(~np.isnan(dens), dens, 7), weights=weights)

        Al_G = get_p('Al', 'shearmodulus') or 26
        Al_B = get_p('Al', 'bulkmodulus') or 76
        Al_dens = get_p('Al', 'density') or 2.7

        out.loc[idx, 'el_pugh_ratio'] = G_mean / (B_mean + eps)
        out.loc[idx, 'el_cauchy_pressure'] = B_mean - G_mean
        out.loc[idx, 'el_density_mismatch'] = abs(dens_mean - Al_dens)
        out.loc[idx, 'el_specific_E'] = E_mean / (dens_mean + eps)
        out.loc[idx, 'el_specific_G'] = G_mean / (dens_mean + eps)
        G_diff = np.sqrt(np.sum(weights * ((G_fill - Al_G) / (Al_G + eps)) ** 2))
        B_diff = np.sqrt(np.sum(weights * ((B_fill - Al_B) / (Al_B + eps)) ** 2))
        out.loc[idx, 'el_shear_mismatch'] = G_diff
        out.loc[idx, 'el_bulk_mismatch'] = B_diff

    return out


def features_orbital(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
) -> pd.DataFrame:
    """orb_orbital: Electronic orbital features (6 features).

    NOTE: Row-by-row loop — acceptable for N < ~10k rows. For larger datasets,
    consider vectorizing via pre-computed element-configuration arrays.
    """
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)

    # s, p, d electron counts for each element (common Al alloying elements)
    e_config = {
        'Al': (2, 1, 0), 'Si': (2, 2, 0), 'Fe': (2, 6, 6), 'Cu': (1, 0, 10),
        'Mn': (2, 6, 5), 'Mg': (2, 6, 0), 'Cr': (1, 5, 5), 'Zn': (2, 0, 10),
        'V': (2, 6, 3), 'Ti': (2, 6, 2), 'Zr': (2, 6, 2), 'Li': (1, 0, 0),
        'Ni': (2, 0, 8), 'Be': (2, 0, 0), 'Sc': (2, 6, 1),
        'C': (2, 2, 0), 'N': (2, 3, 0), 'B': (2, 1, 0), 'P': (2, 3, 0),
        'S': (2, 4, 0), 'Mo': (1, 5, 5), 'Co': (2, 6, 7), 'W': (2, 6, 4),
        'Nb': (1, 4, 4), 'Sn': (2, 2, 10),
    }

    for idx in df.index:
        mass_dict = {element_map[col]: float(df[col].loc[idx]) for col in mass_frac_cols}
        s_sum = sum(mass_dict.values())
        mass_dict['Al'] = max(0.0, 1.0 - s_sum)
        total = max(sum(mass_dict.values()), 1e-9)
        w = {k: v / total for k, v in mass_dict.items()}
        elems = list(w.keys())
        weights = np.array([w[e] for e in elems])

        s_cnt = np.array([e_config.get(e, (1, 0, 0))[0] for e in elems])
        p_cnt = np.array([e_config.get(e, (1, 0, 0))[1] for e in elems])
        d_cnt = np.array([e_config.get(e, (1, 0, 0))[2] for e in elems])
        total_e = s_cnt + p_cnt + d_cnt

        d_mean = np.average(d_cnt, weights=weights)
        d_std = np.sqrt(np.average((d_cnt - d_mean) ** 2, weights=weights))
        s_frac = np.average(s_cnt / np.maximum(total_e, 1e-9), weights=weights)
        p_frac = np.average(p_cnt / np.maximum(total_e, 1e-9), weights=weights)
        d_frac = np.average(d_cnt / np.maximum(total_e, 1e-9), weights=weights)
        sp_sum = np.average((s_cnt + p_cnt) / np.maximum(total_e, 1e-9), weights=weights)

        out.loc[idx, 'orb_d_electron_fraction'] = d_frac
        out.loc[idx, 'orb_sp_over_d'] = sp_sum / max(d_frac, 1e-9)
        out.loc[idx, 'orb_d_electron_count'] = d_mean
        out.loc[idx, 'orb_s_electron_fraction'] = s_frac
        out.loc[idx, 'orb_p_electron_fraction'] = p_frac
        out.loc[idx, 'orb_d_electron_std'] = d_std

    return out


def features_kinetic(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
    process_cols: List[str],
) -> pd.DataFrame:
    """kin_kinetic: Kinetic / diffusion features (5 features).

    NOTE: Row-by-row loop — acceptable for N < ~10k rows. For larger datasets,
    consider vectorizing via pre-computed element-property arrays.
    """
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)
    props = _build_element_database(list(element_map.values()) + ['Al'])
    R = 8.314

    for idx in df.index:
        mass_dict = {element_map[col]: float(df[col].loc[idx]) for col in mass_frac_cols}
        s_sum = sum(mass_dict.values())
        mass_dict['Al'] = max(0.0, 1.0 - s_sum)
        total = max(sum(mass_dict.values()), 1e-9)
        w = {k: v / total for k, v in mass_dict.items()}
        elems = list(w.keys())
        weights = np.array([w[e] for e in elems])

        hf = np.array([
            props[props.index == e]['heatfusion'].values[0]
            if e in props.index.values and not pd.isna(
                props[props.index == e]['heatfusion'].values[0])
            else 10.0 for e in elems])
        tm = np.array([
            props[props.index == e]['Tm'].values[0]
            if e in props.index.values and not pd.isna(
                props[props.index == e]['Tm'].values[0])
            else 1000.0 for e in elems])

        H_fus = np.average(hf, weights=weights)  # kJ/mol
        Tm_mix = np.average(tm, weights=weights)
        Tm_Al = 933.0  # K
        Tage = df['Tage'].loc[idx] if 'Tage' in df.columns else 298
        Tage_K = Tage + 273.15

        out.loc[idx, 'kin_heat_of_fusion_mix'] = H_fus
        out.loc[idx, 'kin_Tm_ratio_to_Al'] = Tm_mix / Tm_Al
        out.loc[idx, 'kin_homologous_T'] = Tage_K / max(Tm_mix, 1e-9)
        J4 = H_fus * 1000.0 / (R * Tage_K + 1e-9)  # activation term
        out.loc[idx, 'kin_activation_term'] = J4
        out.loc[idx, 'kin_diffusion_exp'] = np.exp(-np.clip(J4, 0, 700))

    return out


def features_precipitation(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
) -> pd.DataFrame:
    """B_precipitation: Al-specific precipitation potentials (11 features).

    Note: ppt_Al_balance_wt_pct, ppt_Al_weight_fraction removed (redundant
    with gbl_total_solute in composition group). ppt_solute_weight_fraction
    removed (identical to gbl_total_solute).
    """
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)
    eps = 1e-9

    # Atomic weights for wt% -> at% conversion (matching NB7)
    all_elems = list(element_map.values()) + ['Al']
    prop_df = _build_element_database(all_elems)
    at_wt = {e: prop_df.loc[e, 'atomic_weight'] if e in prop_df.index else 1.0 for e in all_elems}

    def _n(sym):
        """Approximate atomic mole fraction from wt% mass fraction."""
        for c, e in element_map.items():
            if e == sym:
                return df[c].values / at_wt.get(sym, 1.0)
        return np.zeros(len(df))

    Mg, Si, Cu, Zn = _n('Mg'), _n('Si'), _n('Cu'), _n('Zn')
    Mn, Fe, Cr, Zr = _n('Mn'), _n('Fe'), _n('Cr'), _n('Zr')
    Ti, Sc = _n('Ti'), _n('Sc')

    # Precipitation stoichiometric features
    out['ppt_Mg_over_Si_ratio'] = np.divide(Mg, np.maximum(Si, eps))
    out['ppt_Mg2Si_potential'] = np.minimum(Mg / 2.0, Si)
    out['ppt_excess_Mg'] = np.maximum(0, Mg - 2.0 * Si)
    out['ppt_excess_Si'] = np.maximum(0, Si - 0.5 * Mg)   # free Si → brittle particles → hurts El
    out['ppt_Zn_over_Mg_ratio'] = np.divide(Zn, np.maximum(Mg, eps))
    out['ppt_MgZn2_potential'] = np.minimum(Mg, Zn / 2.0)
    out['ppt_excess_Zn'] = np.maximum(0, Zn - 2.0 * Mg)    # excess Zn → grain-boundary η → hurts El
    out['ppt_Cu_over_Mg_ratio'] = np.divide(Cu, np.maximum(Mg, eps))
    out['ppt_S_phase_potential'] = np.minimum(Cu, Mg)
    out['ppt_Sc_over_ScZr'] = np.divide(Sc, np.maximum(Sc + Zr, eps))
    out['ppt_dispersoid_total'] = Sc + Zr + Ti + _n('V') + Cr
    out['ppt_Fe_over_Mn_ratio'] = np.divide(Fe, np.maximum(Mn, eps))
    out['ppt_sludge_factor'] = Fe + 2.0 * Mn + 3.0 * Cr

    return out


# ---------------------------------------------------------------------------
# Precipitation kinetics features (added — previously missing physics)
# ---------------------------------------------------------------------------

def features_precipitation_kinetics(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
    process_cols: List[str],
) -> pd.DataFrame:
    """Precipitation kinetics features for age-hardenable Al alloys (8 features).

    Moves beyond stoichiometric limits (ppt_Mg2Si_potential etc.) to capture
    actual precipitation driving force and kinetics — the dominant
    strengthening mechanism in 2xxx/6xxx/7xxx alloys.

    References:
    - JMA kinetics: Johnson & Mehl (1939), Avrami (1939–41)
    - Al alloy aging: Hatch, Aluminum, ASM (1984) Ch.5
    - Quench sensitivity: Morinaga & Kamado (1993)
    """
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)
    eps = 1e-9

    def _v(sym):
        for c, e in element_map.items():
            if e == sym:
                return df[c].values
        return np.zeros(len(df))

    Mg, Si, Cu, Zn = _v('Mg'), _v('Si'), _v('Cu'), _v('Zn')
    Cr, Mn, Zr = _v('Cr'), _v('Mn'), _v('Zr')
    V = _v('V')
    Tage = df['Tage'].values if 'Tage' in df.columns else np.ones(len(df))
    tage = df['tage'].values if 'tage' in df.columns else np.ones(len(df))
    Tsol = df['Tsol'].values if 'Tsol' in df.columns else np.full(len(df), 500.0)

    # --- Pair products: precipitate volume-fraction drivers ---
    # These are smoother alternatives to min(Mg/2, Si) etc. — the product
    # captures joint availability of BOTH elements for precipitation.
    out['ppt_MgSi_product'] = Mg * Si          # Mg2Si / β'' driver (6xxx)
    out['ppt_MgZn_product'] = Mg * Zn          # MgZn2 / η' driver (7xxx)
    out['ppt_CuMg_product'] = Cu * Mg          # S-phase driver (2xxx)
    out['ppt_ZnMgCu_product'] = Zn * Mg * Cu   # η-phase with Cu (7xxx)

    # --- Aging state indicators ---
    # Homologous temperature ratio — diffusion speed at aging temp
    out['ppt_aging_temp_ratio'] = Tage / np.maximum(Tsol, eps)
    # Quench supersaturation — thermodynamic driving force for precipitation
    out['ppt_supersaturation'] = (Tsol - Tage) / np.maximum(Tsol, eps)
    # Coupled thermodynamic-kinetic aging extent
    out['ppt_log_tage_weighted'] = np.log(np.maximum(tage, eps)) * (Tsol - Tage)

    # --- Quench sensitivity ---
    # Cr, Mn, Zr, V form sub-μm dispersoids that promote heterogeneous
    # precipitation during quenching, consuming solute and reducing
    # subsequent aging response (quench sensitivity effect).
    out['ppt_quench_sensitivity'] = Cr + Mn + Zr + V

    return out


# ---------------------------------------------------------------------------
# Alloy classification features (added — family-specific strengthening context)
# ---------------------------------------------------------------------------

def features_alloy_class(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
    process_cols: List[str],
) -> pd.DataFrame:
    """Alloy family classification indicators (7 features).

    Different alloy families (2xxx/6xxx/7xxx/8xxx) have fundamentally
    different strengthening mechanisms. These binary indicators let the
    model learn family-specific composition-property relationships without
    having to infer the alloy family from raw composition first.
    """
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)
    eps = 1e-9

    def _v(sym):
        for c, e in element_map.items():
            if e == sym:
                return df[c].values
        return np.zeros(len(df))

    Cu, Mg, Zn, Si, Li = _v('Cu'), _v('Mg'), _v('Zn'), _v('Si'), _v('Li')
    Sc, Zr = _v('Sc'), _v('Zr')

    # Binary alloy family indicators (tree models split naturally on these)
    out['cls_is_2xxx'] = ((Cu > 0.01) & ((Mg + eps) / (Cu + eps) < 1.0)).astype(float)
    out['cls_is_6xxx'] = ((Mg > 0.003) & (Si > 0.003) & (Cu < Mg)).astype(float)
    out['cls_is_7xxx'] = (Zn > 0.02).astype(float)
    out['cls_is_8xxx'] = (Li > 0.005).astype(float)

    # Microalloying / special processing indicators
    out['cls_has_ScZr'] = ((Sc + Zr) > 0.001).astype(float)

    # Heat treatment state (natural aging vs artificial aging vs as-solutionized)
    Tage = df['Tage'].values if 'Tage' in df.columns else np.zeros(len(df))
    out['cls_is_heat_treated'] = (Tage > 60).astype(float)

    # High-solute vs dilute (separates wrought heat-treatable from non-heat-treatable)
    total_solute = df[mass_frac_cols].sum(axis=1).values
    out['cls_high_solute'] = (total_solute > 0.02).astype(float)

    return out


# ---------------------------------------------------------------------------
# Solid solution strengthening features (added — Labusch model)
# ---------------------------------------------------------------------------

def features_solid_solution(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
) -> pd.DataFrame:
    """Solid solution strengthening features (4 features).

    Each solute species contributes differently to strength when dissolved
    (not precipitated). The Labusch model accounts for the nonlinear
    concentration dependence: Δσ ∝ c^(2/3).

    Also includes the Fleischer combined parameter (from archive v7) which
    joins size + modulus misfit into a single kernel — often more
    predictive than separate terms on small datasets.

    References:
    - Labusch, R., Phys. Stat. Sol. 41 (1970) 659
    - Fleischer, R.L., Acta Metall. 11 (1963) 203
    - Gypen & Deruyttere, J. Mater. Sci. 12 (1977) 1028
    - For Al: Tiryakioğlu et al., MSE A 527 (2010) — k coefficients

    NOTE: Row-by-row loop — acceptable for N < ~10k rows.
    """
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)
    prop_df = _build_element_database(list(element_map.values()) + ['Al'])
    eps = 1e-9

    # Labusch strengthening coefficients k (MPa per at%^(2/3)) for Al
    k_labusch = {
        'Mg': 15.0, 'Cu': 46.0, 'Zn': 3.0, 'Si': 25.0,
        'Mn': 30.0, 'Cr': 7.0, 'Fe': 8.0, 'Zr': 12.0,
        'Ti': 20.0, 'Li': 2.0, 'Ni': 10.0, 'V': 12.0,
        'Sc': 25.0, 'Be': 5.0,
    }

    # Al reference values for misfit calculation
    r_Al = 1.43   # atomic radius (Angstrom)
    G_Al = 26.0   # shear modulus (GPa)
    E_Al = 70.0   # Young's modulus (GPa) — approximate for pure Al

    for idx in df.index:
        mass_dict = {element_map[col]: float(df[col].loc[idx]) for col in mass_frac_cols}
        s_sum = sum(mass_dict.values())
        mass_dict['Al'] = max(0.0, 1.0 - s_sum)
        total_mass = max(sum(mass_dict.values()), eps)

        labusch_sum = 0.0
        size_misfit_sum = 0.0
        modulus_misfit_sum = 0.0
        fleischer_sum = 0.0

        for elem, mass_frac in mass_dict.items():
            if elem == 'Al' or mass_frac < 1e-9:
                continue
            # Convert mass fraction → atomic fraction (approximate, single-solute)
            at_wt_elem = (prop_df.loc[elem, 'atomic_weight']
                          if elem in prop_df.index and not pd.isna(prop_df.loc[elem, 'atomic_weight'])
                          else 50.0)
            at_frac = (mass_frac / at_wt_elem) / (
                mass_frac / at_wt_elem + (1.0 - mass_frac) / 26.98)

            # Labusch rate: k × c^(2/3), c in at%
            k = k_labusch.get(elem, 5.0)
            labusch_sum += k * (at_frac * 100.0) ** (2.0 / 3.0)

            # Size misfit and modulus misfit
            if elem in prop_df.index:
                r_elem = prop_df.loc[elem, 'atomic_radius']
                if not pd.isna(r_elem):
                    radius_delta = (r_elem - r_Al) / max(r_Al, eps)
                    size_misfit_sum += abs(radius_delta) * at_frac
                else:
                    radius_delta = 0.0

                # Get Young's modulus for Fleischer parameter
                G_elem = np.nan
                if 'shearmodulus' in prop_df.columns:
                    G_elem = prop_df.loc[elem, 'shearmodulus']
                if not pd.isna(G_elem):
                    modulus_misfit_sum += abs(G_elem - G_Al) / max(G_Al, eps) * at_frac

                # Fleischer combined solid-solution kernel:
                # kernel = (delta_r² + 3·delta_G²)^(2/3)
                # This is the archive v7 C9_solid_solution_parameter
                G_val = G_elem if not (isinstance(G_elem, float) and np.isnan(G_elem)) else G_Al
                modulus_delta = (G_val - G_Al) / max(G_Al, eps)
                fleischer_kernel = (radius_delta ** 2 + 3.0 * modulus_delta ** 2) ** (2.0 / 3.0)
                fleischer_sum += at_frac * fleischer_kernel

        out.loc[idx, 'sol_labusch_rate'] = labusch_sum
        out.loc[idx, 'sol_size_misfit'] = size_misfit_sum
        out.loc[idx, 'sol_modulus_misfit'] = modulus_misfit_sum
        out.loc[idx, 'sol_fleischer_combined'] = fleischer_sum

    return out


# ---------------------------------------------------------------------------
# Composition × Process interactions (added — core precipitation physics)
# ---------------------------------------------------------------------------

def features_comp_process_v2(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
    process_cols: List[str],
) -> pd.DataFrame:
    """Composition × Process interactions — all solute elements (42 features).

    Archive v7 approach: generate ALL element×process interactions and let
    feature selection pick the useful ones. The archive found this necessary
    — some trace elements (e.g. Zr×Tage for dispersoid precipitation) affect
    properties through process-dependent mechanisms.

    Three interaction types per element:
    - × Tage (precipitation driving force)
    - × log(tage) (precipitation kinetics)
    - × Tsol (solution treatment effect)
    """
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)
    eps = 1e-9

    def _v(sym):
        for c, e in element_map.items():
            if e == sym:
                return df[c].values
        return np.zeros(len(df))

    Tage = df['Tage'].values if 'Tage' in df.columns else np.ones(len(df))
    tage = df['tage'].values if 'tage' in df.columns else np.ones(len(df))
    Tsol = df['Tsol'].values if 'Tsol' in df.columns else np.full(len(df), 500.0)
    log_tage = np.log(np.maximum(tage, eps))

    # --- Per-element × process (archive v7: F_{element}_times_X) ---
    for col, elem in element_map.items():
        v = df[col].values
        out[f'cxp_{elem}_x_Tage'] = v * Tage
        out[f'cxp_{elem}_x_log_tage'] = v * log_tage
        out[f'cxp_{elem}_x_Tsol'] = v * Tsol

    # --- Global C×P features (archive v7) ---
    Mg, Zn, Cu = _v('Mg'), _v('Zn'), _v('Cu')
    Tage_K = Tage + 273.15
    larson_miller = Tage_K * (np.log10(np.maximum(tage, eps)) + 20.0)
    out['cxp_MgZn_x_LMP'] = (Mg + Zn) * larson_miller
    out['cxp_Cu_x_LMP'] = Cu * larson_miller
    out['cxp_MgZn_x_Tage'] = Mg * Zn * Tage

    return out


# ---------------------------------------------------------------------------
# Delta-to-Al features (archive v7 — key for El prediction)
# ---------------------------------------------------------------------------

def features_delta_to_al(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
) -> pd.DataFrame:
    """Element property delta-to-Al features (archive v7 B_delta_to_Al, ~20 features).

    Computes at%-weighted average of (property_i - property_Al) for key
    element properties. This captures \"how different from pure Al\" the
    alloy is — directly relevant to ductility (more different = more
    obstacles to dislocation motion = less ductile).

    NOTE: Row-by-row loop — acceptable for N < ~10k rows.
    """
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)
    prop_df = _build_element_database(list(element_map.values()) + ['Al'])
    eps = 1e-9

    # Properties relevant to mechanical behavior (archive v7 DELTA_AL_PROPS subset)
    DELTA_PROPS = [
        'atomic_radius',   # size mismatch -> solid solution -> ductility
        'chi',             # electronegativity -> intermetallic formation
        'VEC',             # valence electron concentration -> phase stability
        'Tm',              # melting temperature -> diffusion -> coarsening
        'density',         # specific properties
    ]

    # Additional properties if available in the database
    for extra in ['covalentradius', 'firstionizationenergy', 'electronaffinity']:
        if extra not in DELTA_PROPS:
            DELTA_PROPS.append(extra)

    for idx in df.index:
        mass_dict = {element_map[col]: float(df[col].loc[idx]) for col in mass_frac_cols}
        s_sum = sum(mass_dict.values())
        mass_dict['Al'] = max(0.0, 1.0 - s_sum)
        total_mass = max(sum(mass_dict.values()), eps)

        # Approximate atomic fractions from mass fractions
        at_frac = {}
        for elem, mf in mass_dict.items():
            at_wt = (prop_df.loc[elem, 'atomic_weight']
                     if elem in prop_df.index and not pd.isna(prop_df.loc[elem, 'atomic_weight'])
                     else 50.0)
            at_frac[elem] = (mf / at_wt) / sum(
                mass_dict[e2] / (prop_df.loc[e2, 'atomic_weight']
                                 if e2 in prop_df.index and not pd.isna(prop_df.loc[e2, 'atomic_weight'])
                                 else 50.0)
                for e2 in mass_dict)

        for prop in DELTA_PROPS:
            # Map to property_df column names
            prop_col = prop
            prop_name = prop
            if prop == 'covalentradius':
                prop_name = 'r_cov'
            elif prop == 'firstionizationenergy':
                prop_name = 'IE1'
            elif prop == 'electronaffinity':
                prop_name = 'EA'

            al_val = (prop_df.loc['Al', prop_col]
                      if 'Al' in prop_df.index and prop_col in prop_df.columns
                         and not pd.isna(prop_df.loc['Al', prop_col])
                      else 0.0)

            delta_sum = 0.0
            abs_delta_sum = 0.0
            for elem, frac in at_frac.items():
                if elem == 'Al' or frac < eps:
                    continue
                if elem in prop_df.index and prop_col in prop_df.columns:
                    val = prop_df.loc[elem, prop_col]
                    if not pd.isna(val):
                        delta_sum += frac * (float(val) - float(al_val))
                        abs_delta_sum += frac * abs(float(val) - float(al_val))

            out.loc[idx, f'delta_{prop_name}_to_Al'] = delta_sum
            out.loc[idx, f'delta_{prop_name}_abs_to_Al'] = abs_delta_sum

    return out


# ---------------------------------------------------------------------------
# Full-alloy properties (added from archive v7 — includes Al in averages)
# ---------------------------------------------------------------------------

def features_full_alloy(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
) -> pd.DataFrame:
    """Full-alloy thermodynamic properties including Al (8 features).

    Unlike the solute-only averages in the physics group, these include Al
    in the weighted averaging — giving a different perspective on overall
    alloy character. From archive v7 K1-K8.

    NOTE: Row-by-row loop — acceptable for N < ~10k rows.
    """
    out = pd.DataFrame(index=df.index)
    element_map = _extract_element_map(mass_frac_cols)
    prop_df = _build_element_database(list(element_map.values()) + ['Al'])
    R = 8.314
    eps = 1e-9

    all_elements = ['Al'] + list(element_map.values())
    at_wt = {e: (prop_df.loc[e, 'atomic_weight']
                 if e in prop_df.index and not pd.isna(prop_df.loc[e, 'atomic_weight'])
                 else 50.0) for e in all_elements}

    for idx in df.index:
        # Build full-alloy mass dictionary (Al + all solutes)
        mass_dict = {element_map[col]: float(df[col].loc[idx]) for col in mass_frac_cols}
        s_sum = sum(mass_dict.values())
        mass_dict['Al'] = max(0.0, 1.0 - s_sum)

        # Convert to mole fractions
        mole_dict = {e: mass_dict[e] / at_wt[e] for e in all_elements}
        total_mole = sum(mole_dict.values())
        if total_mole <= 0:
            for k in ['fal_Al_wt_pct', 'fal_total_solute_at_frac',
                       'fal_VEC', 'fal_entropy', 'fal_size_mismatch',
                       'fal_Tm', 'fal_chi', 'fal_atomic_radius']:
                out.loc[idx, k] = 0.0
            continue

        at_frac = {e: mole_dict[e] / total_mole for e in all_elements}
        weights = np.array([at_frac[e] for e in all_elements])

        # K1-K3: Al balance and solute fraction
        out.loc[idx, 'fal_Al_wt_pct'] = mass_dict['Al'] * 100.0
        out.loc[idx, 'fal_total_solute_at_frac'] = 1.0 - at_frac['Al']

        # Property vectors for all elements (including Al)
        def _get_prop(prop_name):
            vals = []
            for e in all_elements:
                if e in prop_df.index and not pd.isna(prop_df.loc[e, prop_name]):
                    vals.append(float(prop_df.loc[e, prop_name]))
                else:
                    vals.append(np.nan)
            return np.array(vals)

        # K4: Full-alloy VEC
        vec = _get_prop('VEC')
        vec_valid = ~np.isnan(vec)
        if vec_valid.any():
            out.loc[idx, 'fal_VEC'] = np.average(vec[vec_valid], weights=weights[vec_valid])

        # K5: Full-alloy entropy of mixing
        valid_pos = weights > eps
        if valid_pos.any():
            out.loc[idx, 'fal_entropy'] = -R * np.sum(weights[valid_pos] * np.log(weights[valid_pos]))

        # K6: Full-alloy atomic size mismatch
        r_atom = _get_prop('atomic_radius')
        r_valid = ~np.isnan(r_atom)
        if r_valid.sum() > 1:
            r_mean = np.average(r_atom[r_valid], weights=weights[r_valid])
            out.loc[idx, 'fal_size_mismatch'] = 100.0 * np.sqrt(
                np.sum(weights[r_valid] * (1.0 - r_atom[r_valid] / max(r_mean, eps)) ** 2))

        # K7: Full-alloy mixed melting temperature
        tm = _get_prop('Tm')
        tm_valid = ~np.isnan(tm)
        if tm_valid.any():
            out.loc[idx, 'fal_Tm'] = np.average(tm[tm_valid], weights=weights[tm_valid])

        # K8: Full-alloy mean electronegativity
        chi = _get_prop('chi')
        chi_valid = ~np.isnan(chi)
        if chi_valid.any():
            out.loc[idx, 'fal_chi'] = np.average(chi[chi_valid], weights=weights[chi_valid])

        # Additional: full-alloy mean atomic radius
        if r_valid.any():
            out.loc[idx, 'fal_atomic_radius'] = np.average(r_atom[r_valid], weights=weights[r_valid])

    return out


# ---------------------------------------------------------------------------
# Miedema mixing enthalpy matrix (shared across categories)
# ---------------------------------------------------------------------------

def compute_miedema_enthalpy(
    elements: List[str],
) -> Optional[pd.DataFrame]:
    """
    Build a Miedema binary mixing enthalpy matrix for the given elements.

    Uses matminer's MixingEnthalpy featurizer to query real pair-specific
    mixing enthalpies. Includes Al explicitly since it is the base element.

    Parameters
    ----------
    elements : list of str
        Element symbols present in the alloy system (e.g., ['Si','Fe','Cu']).

    Returns
    -------
    pd.DataFrame or None
        Square DataFrame of mixing enthalpies (kJ/mol), indexed by element
        symbol. Returns None if matminer MixingEnthalpy is not available.
    """
    try:
        from matminer.featurizers.composition.alloy import MixingEnthalpy
    except ImportError:
        return None

    all_elems = sorted(set(elements) | {'Al'})
    miedema = MixingEnthalpy()
    n = len(all_elems)
    matrix = pd.DataFrame(np.zeros((n, n)), index=all_elems, columns=all_elems)

    from pymatgen.core import Element as PymElement
    for i, e1 in enumerate(all_elems):
        for j, e2 in enumerate(all_elems):
            if i < j:
                try:
                    val = miedema.get_mixing_enthalpy(PymElement(e1), PymElement(e2))
                    if val is not None and not np.isnan(val):
                        matrix.loc[e1, e2] = float(val)
                        matrix.loc[e2, e1] = float(val)
                except Exception:
                    pass
    return matrix


# ---------------------------------------------------------------------------
# Unified entry point — 4 semantic groups (~370 features)
# ---------------------------------------------------------------------------

def featurize_metal_all(
    df: pd.DataFrame,
    mass_frac_cols: List[str],
    process_cols: Optional[List[str]] = None,
    target_cols: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
    add_al_balance: bool = True,
    use_magpie: bool = True,
    use_stoichiometry: bool = True,
    use_valence: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str], Dict[str, List[str]]]:
    """
    Unified feature engineering — 4 groups, 330 features (v3 final).

    v3 structure (Delta-to-Al moved to Physics for El isolation):
    - Composition (A, 71): solutes, Magpie, global, orbital, alloy class, solid solution
    - Precipitation (B, 198): interactions, pair products, ratios, precipitates,
      kinetics, CxP (all 14 elements), wrought
    - Process (C, 16): raw params + transforms
    - Physics (D, 45): thermodynamics, elastic, kinetic, full-alloy, delta-to-Al(16)

    Key findings from ablation:
    - YS/UTS: All(330) best (+3.5%/+6.2%). Precipitation(B) & Process(C) drive this.
    - El: Delta-to-Al(16) gives +5.4% over Raw. It is the ONLY El signal.
      Buried in D(45), diluted to -0.2%. N03 per-target LassoCV extracts it.
    - Group A (Magpie stats etc.): noise for all targets.
    - Group C (16 process features): +3.7% for UTS — strongest single-group effect.

    Parameters
    ----------
    df : pd.DataFrame
    mass_frac_cols : list of str
        Element column names, e.g. ['Si','Fe','Cu',...].
    process_cols : list of str
        Process column names, e.g. ['Tsol','Tage','tage'].
    target_cols : list of str, optional
    groups : list of str, optional
        Which groups to include. Default: all 4.
        Options: 'composition', 'precipitation', 'process', 'physics'.
    add_al_balance : bool
    use_magpie, use_stoichiometry, use_valence : bool

    Returns
    -------
    X : pd.DataFrame
    y : pd.DataFrame or None
    feature_names : list of str
    group_map : dict  {group_name: [feature_names]}
    """
    if process_cols is None:
        process_cols = []
    if groups is None:
        groups = ['composition', 'precipitation', 'process', 'physics']

    element_map = _extract_element_map(mass_frac_cols)
    all_dfs = []
    group_map = {}

    # Build Miedema matrix once (used by D_thermo in physics group)
    miedema = None
    if 'physics' in groups:
        miedema = compute_miedema_enthalpy(list(element_map.values()))

    # ================================================================
    # Assemble features into 4 groups (v3: merged 7→4)
    # ================================================================
    # Build sub-DataFrames first, then merge into 4 output groups.
    # This keeps the computation modular while producing a cleaner
    # top-level grouping for ablation and interpretation.

    # --- Internal sub-frames (computed regardless of group selection) ---
    sub_frames = {}       # {sub_name: DataFrame}
    sub_names = {}        # {sub_name: [feature_names]}

    # Solute + process raw inputs
    Xa = features_input(df, mass_frac_cols, process_cols)
    solute_cols = [c for c in Xa.columns if c.startswith('sol_')]
    proc_cols_raw = [c for c in Xa.columns if c.startswith('prc_')]
    sub_frames['solute'] = Xa[solute_cols]
    sub_names['solute'] = solute_cols
    sub_frames['prc_raw'] = Xa[proc_cols_raw] if proc_cols_raw else pd.DataFrame(index=df.index)
    sub_names['prc_raw'] = proc_cols_raw

    # Magpie + Stoichiometry + ValenceOrbital
    Xb, _, bnames = featurize_metal_df(
        df, mass_frac_cols=mass_frac_cols,
        process_cols=[],
        target_cols=None,
        use_magpie=use_magpie, use_stoichiometry=use_stoichiometry,
        use_valence=use_valence, keep_raw_frac=False,
        add_al_balance=add_al_balance,
    )
    bnames_short = [n.replace('MagpieData ', 'mag_') for n in bnames]
    bnames_short = [n.replace('-norm', '').replace(' ', '_').replace('avg_', 'val_avg_')
                    for n in bnames_short]
    EXCLUDE_MAGPIE = ['NdValence', 'NfValence', 'NdUnfilled', 'NfUnfilled',
                      'GSmagmom', 'GSbandgap', 'SpaceGroupNumber']
    keep_idx = [i for i, n in enumerate(bnames_short)
                if not any(e in n for e in EXCLUDE_MAGPIE)]
    bnames_short = [bnames_short[i] for i in keep_idx]
    Xb = Xb.iloc[:, keep_idx]
    KEEP_MAGPIE_STATS = ['mean', 'range']
    magpie_keep_idx = [i for i, n in enumerate(bnames_short)
                       if any(s in n for s in KEEP_MAGPIE_STATS)]
    bnames_short = [bnames_short[i] for i in magpie_keep_idx]
    Xb = Xb.iloc[:, magpie_keep_idx]
    Xb.columns = bnames_short
    sub_frames['magpie'] = Xb
    sub_names['magpie'] = bnames_short

    # Global composition statistics
    Xg = features_global(df, mass_frac_cols, process_cols)
    sub_frames['global'] = Xg
    sub_names['global'] = list(Xg.columns)

    # Orbital (all 6 features from archive v7 I1-I6)
    Xi = features_orbital(df, mass_frac_cols)
    sub_frames['orbital'] = Xi
    sub_names['orbital'] = list(Xi.columns)

    # Alloy classification
    Xcls = features_alloy_class(df, mass_frac_cols, process_cols)
    sub_frames['alloy_cls'] = Xcls
    sub_names['alloy_cls'] = list(Xcls.columns)

    # Solid solution strengthening
    Xsol = features_solid_solution(df, mass_frac_cols)
    sub_frames['solid_sol'] = Xsol
    sub_names['solid_sol'] = list(Xsol.columns)

    # Delta-to-Al features (archive v7 — key for El)
    Xdelta = features_delta_to_al(df, mass_frac_cols)
    sub_frames['delta_al'] = Xdelta
    sub_names['delta_al'] = list(Xdelta.columns)

    # Named interactions + pair products
    Xc = features_interaction(df, mass_frac_cols)
    sub_frames['interact'] = Xc
    sub_names['interact'] = list(Xc.columns)

    # Precipitation potentials
    Xk = features_precipitation(df, mass_frac_cols)
    sub_frames['precip'] = Xk
    sub_names['precip'] = list(Xk.columns)

    # Precipitation kinetics
    Xp = features_precipitation_kinetics(df, mass_frac_cols, process_cols)
    sub_frames['precip_kin'] = Xp
    sub_names['precip_kin'] = list(Xp.columns)

    # C×P interactions
    if len(process_cols) > 0:
        Xcxp = features_comp_process_v2(df, mass_frac_cols, process_cols)
    else:
        Xcxp = pd.DataFrame(index=df.index)
    sub_frames['cxp'] = Xcxp
    sub_names['cxp'] = list(Xcxp.columns)

    # Wrought descriptors
    Xw = features_wrought(df, mass_frac_cols)
    sub_frames['wrought'] = Xw
    sub_names['wrought'] = list(Xw.columns)

    # Process transforms
    if len(process_cols) > 0:
        Xe = features_process(df, process_cols)
    else:
        Xe = pd.DataFrame(index=df.index)
    sub_frames['prc_trans'] = Xe
    sub_names['prc_trans'] = list(Xe.columns)

    # Thermodynamic descriptors
    Xd = features_thermo(df, mass_frac_cols, miedema, add_al_balance)
    sub_frames['thermo'] = Xd
    sub_names['thermo'] = list(Xd.columns)

    # Elastic mismatch
    Xh = features_elastic(df, mass_frac_cols)
    sub_frames['elastic'] = Xh
    sub_names['elastic'] = list(Xh.columns)

    # Kinetic/diffusion
    Xj = features_kinetic(df, mass_frac_cols, process_cols)
    sub_frames['kinetic'] = Xj
    sub_names['kinetic'] = list(Xj.columns)

    # Full-alloy properties (from archive v7)
    Xfal = features_full_alloy(df, mass_frac_cols)
    sub_frames['full_alloy'] = Xfal
    sub_names['full_alloy'] = list(Xfal.columns)

    # --- Align sub-frames to common index (handle Magpie row drops) ---
    common_idx = df.index
    if len(sub_frames['magpie']) < len(df):
        n_lost = len(df) - len(sub_frames['magpie'])
        lost_idx = sorted(set(df.index) - set(sub_frames['magpie'].index))
        warnings.warn(
            f"featurize_metal_all: {n_lost} row(s) dropped during Magpie "
            f"featurization (indices: {lost_idx}). "
            f"All feature groups will be aligned to the {len(sub_frames['magpie'])} surviving rows."
        )
        common_idx = sub_frames['magpie'].index
        if target_cols is not None and len(target_cols) > 0:
            df = df.loc[common_idx]
    for key in sub_frames:
        if len(sub_frames[key]) > len(common_idx):
            sub_frames[key] = sub_frames[key].loc[common_idx]

    # --- Merge into 4 output groups ---
    # Group A: Composition = solute + magpie + global + orbital + alloy_cls + solid_sol
    # Group B: Precipitation = interact + precip + precip_kin + cxp + wrought
    # Group C: Process = prc_raw + prc_trans
    # Group D: Physics = thermo + elastic + kinetic + full_alloy

    group_defs = {
        'composition':   ['solute', 'magpie', 'global', 'orbital', 'alloy_cls', 'solid_sol'],
        'precipitation': ['interact', 'precip', 'precip_kin', 'cxp', 'wrought'],
        'process':       ['prc_raw', 'prc_trans'],
        'physics':       ['thermo', 'elastic', 'kinetic', 'full_alloy', 'delta_al'],
    }

    for grp_name in groups:
        if grp_name not in group_defs:
            continue
        sub_keys = group_defs[grp_name]
        grp_dfs = [sub_frames[k].reset_index(drop=True) for k in sub_keys if k in sub_frames]
        grp_names_flat = []
        for k in sub_keys:
            if k in sub_names:
                grp_names_flat.extend(sub_names[k])
        if grp_dfs:
            X_grp = pd.concat(grp_dfs, axis=1)
            all_dfs.append(X_grp)
            group_map[grp_name] = grp_names_flat


    # ================================================================
    # Concatenate and finalize
    # ================================================================
    group_prefixes = {
        'composition':   'A_',
        'precipitation': 'B_',
        'process':       'C_',
        'physics':       'D_',
    }
    final_map = {}
    for grp_name, feat_list in group_map.items():
        pfx = group_prefixes.get(grp_name, '')
        renamed = [pfx + f for f in feat_list]
        final_map[grp_name] = renamed
        # Rename columns in the corresponding DataFrame
        if grp_name in group_map:
            idx = list(group_map.keys()).index(grp_name)
            all_dfs[idx].columns = renamed

    X = pd.concat(all_dfs, axis=1).fillna(0.0)

    all_names = []
    for names in final_map.values():
        all_names.extend(names)

    # Ensure consistent columns
    for c in all_names:
        if c not in X.columns:
            X[c] = 0.0
    X = X[all_names]

    if target_cols is not None and len(target_cols) > 0:
        y = df[target_cols].reset_index(drop=True)
    else:
        y = None

    return X, y, all_names, final_map


def align_to_template(
    X: pd.DataFrame,
    template_names: List[str],
) -> pd.DataFrame:
    """Ensure X has exactly the columns in template_names, filling missing with 0."""
    for c in template_names:
        if c not in X.columns:
            X[c] = 0.0
    return X[template_names]
