# Wrought Aluminum Alloy — ML Pipeline for Mechanical Property Prediction & Inverse Design

A 9-step machine learning pipeline that predicts yield strength (YS), ultimate tensile strength (UTS), and elongation (El) of wrought aluminum alloys from composition and heat treatment parameters, then performs inverse design via cluster-based multi-objective optimization.

## Project Structure

```
01.letter_code/
├── README.md
├── code/
│   ├── featurize_metal.py               # 4-group feature engineering (330 features)
│   ├── feature_selection.py             # VarFilter + CorrFilter pre-selection utilities
│   ├── 01_data_preprocessing.ipynb      # Load → Clean → Split → MICE imputation
│   ├── 02_feature_engineering.ipynb     # 4 groups: Composition | Precipitation | Process | Physics
│   ├── 03_feature_selection.ipynb       # VarFilter → CorrFilter → ExtraTrees rank → SFFS
│   ├── 04_model_training.ipynb          # Phase A: Optuna tuning (6 models) + Phase B: 50-split validation
│   ├── 05_model_explanation.ipynb       # SHAP: polar bee-swarm, dependence, waterfall, cross-target
│   ├── 06_composition_process_optimization.ipynb  # K-means clustering → LHS → Pareto optimization
│   ├── 07_cross_cluster_analysis.ipynb  # Cross-cluster Pareto front comparison
│   ├── 08_training_pareto_validation.ipynb  # Shen et al. (2024) method: HV ratio + extrapolation check
│   └── 09_visualization.ipynb           # Publication figures (Graphical Abstract + Fig 1–4)
├── data/                                # Raw data + intermediate outputs (.npy, .csv, .json)
├── figure/                              # Publication-quality PDF figures (600 DPI)
├── best_models/                         # Trained model artifacts (.pkl per target)
├── optimization_results/                # Pareto solutions + clustered training data (.csv)
└── reference/                           # Reference papers (PDF)
```

## Feature Engineering — 4 Groups (330 features)

| Group | Prefix | Content | Count | Principle |
|-------|--------|---------|:-----:|-----------|
| **A: Composition** | `A_` | Solutes + Magpie stats + Global stats + Orbital + Alloy class + Solid solution | 71 | Element identity, periodic trends |
| **B: Precipitation** | `B_` | Named interactions + Pair products + Solute/Al ratios + Precipitation potentials + Kinetics + C×P interactions + Wrought descriptors | 198 | Synergistic effects, phase formation, aging |
| **C: Process** | `C_` | Raw process params + Engineering transforms (Larson-Miller, etc.) | 16 | Heat treatment kinetics |
| **D: Physics** | `D_` | Thermodynamics + Elastic mismatch + Kinetics + Full-alloy properties + Delta-to-Al | 45 | Physical metallurgy principles |

All groups are **non-overlapping**. Feature selection runs per-target via SFFS to pick the best subset automatically — final models use only 5–9 features per target.

## Pipeline Flow

| # | Notebook | Input | Key Output |
|---|----------|-------|------------|
| 01 | Data Preprocessing | `Wrought_Al_alloy.xlsx` | Imputed train/holdout CSVs, `best_imputer.json` |
| 02 | Feature Engineering | Imputed CSVs | `X.npy`, `y.npy`, `feature_meta.json`, `optimal_features.json` |
| 03 | Feature Selection | `X_train.npy`, `optimal_features.json` | `selected_features.json` (5–9 features/target) |
| 04 | Model Training | `metal_cleaned.csv`, `selected_features.json` | `best_models/*.pkl`, `model_selection_results.json`, `repeated_split_results.json` |
| 05 | Model Explanation | `best_models/*.pkl`, SHAP data | SHAP figures (PDF, 600 DPI), `shap_plot_data.npz` |
| 06 | Composition-Process Optimization | `best_models/*.pkl`, `metal_cleaned.csv` | `cluster_pareto.csv` (235 solutions), `training_clustered.csv` |
| 07 | Cross-Cluster Analysis | `cluster_pareto.csv` | Cross-cluster Pareto overlay figure |
| 08 | Training-Data Pareto Validation | `cluster_pareto.csv`, `training_clustered.csv` | HV ratio + extrapolation report |
| 09 | Publication Figures | All above outputs | `Graphical_Abstract.pdf`, `Fig1–4_*.pdf` |

## Key Results

### Forward Prediction (50-Split Robust Validation)

| Target | Model | Features | R² | RMSE |
|--------|-------|:--------:|:---:|:----:|
| **YS** (Yield Strength) | ExtraTrees | 9 | 0.9032 ± 0.0192 | 38.08 ± 3.39 MPa |
| **UTS** (Ultimate Tensile Strength) | ExtraTrees | 5 | 0.8914 ± 0.0195 | 36.37 ± 2.96 MPa |
| **El** (Elongation) | LightGBM | 7 | 0.7479 ± 0.0533 | 3.71 ± 0.39 % |

### Selected Features per Target

- **YS** (9 features): Precipitation-dominated — C×P interactions (Cu×LMP, Zr×Tage, Mg×Tsol, Fe×Tsol, Li×log_tage, Cr×Tage) + Total solute + Process transforms (Tage×log_tage, Tsol×Tage)
- **UTS** (5 features): Physics-dominated — Thermodynamic δ + Larson-Miller + Specific elastic modulus + Solidus depression + Tsol×Tage
- **El** (7 features): Physics + Raw process — ΔTm, ΔIE1, Δχ (delta-to-Al) + Tage, tage, Tsol + Zn

### Inverse Design (Cluster-Based Optimization)

- **K = 5** metallurgical families identified via K-means clustering (silhouette = 0.362)
- **5,000 LHS samples** per cluster, filtered by 5 domain-knowledge constraints
- **235 Pareto-optimal** solutions across all clusters
- **Hypervolume ratio**: up to 2.05× over training Pareto (C3: Cu+Mg+Zn family)
- **Extrapolation**: < 5% for all clusters (well within training data coverage)
- **Best Balance** (C5 Cu+Mg+Zn): YS = 546 MPa, UTS = 611 MPa, El = 5.9%

## Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Imputation | MICE (ExtraTrees) | Best wNRMSE = 0.270 in 11-candidate comparison |
| Feature selection | SFFS with 1-SE rule (best-K for El) | Parsimonious models; El's higher CV variance makes 1-SE too aggressive |
| Model selection | Lowest CV RMSE among top-6 N01 models | Consistent with RMSE as primary metric throughout pipeline |
| Validation | 50 random 70/30 splits | Per-split MICE (fit on train only) prevents information leakage |
| Inverse design | K-means + per-cluster LHS | Avoids impossible alloy compositions from unconstrained LHS |
| Optimization constraints | Sc ≤ 0.3%, Fe ≤ 0.5%, Tage ≥ 100°C, Al ≥ 85%, YS ≥ 250 MPa | Domain knowledge: commercial viability + artificial aging only |

## SHAP Interpretation Highlights

- **YS** dominated by **Precipitation (66%)** + **Process (34%)** — Cu-Mg solute interactions during aging
- **UTS** controlled by **Physics (61%)** + **Process (28%)** — Thermodynamic stability and thermal history
- **El** governed by **Physics (59%)** + **Raw Process (32%)** — Electronegativity differences and melting point deviations affect grain boundary cohesion

## Dependencies

```
numpy pandas scikit-learn scipy matplotlib seaborn
pymatgen matminer torch shap pymoo mlxtend
xgboost lightgbm optuna
```

## Quick Start

```bash
cd code/
jupyter notebook
# Run 01 → 09 in order
# Notebooks 01–05 must run sequentially (each depends on prior outputs)
# Notebooks 06–08 can run after 04 (depend on best_models/ and data/)
# Notebook 09 is standalone (loads pre-computed data from all prior steps)
```

## Reference Papers

Key methodological references (see `reference/` directory):
- Bhat et al. (2024), *Metals* 14(2), 239 — Class-based regression for Al alloy property prediction
- Hu et al. (2021) — Feature engineering assisted ML for wrought Al alloys
- Shen et al. (2024), *Computational Materials Science* — Training-data Pareto validation method (N08)
