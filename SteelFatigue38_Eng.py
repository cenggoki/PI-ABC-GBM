#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steel Fatigue Strength Prediction – Final PyQt Desktop Application
Data Science & Metallurgy Integration

"""

import sys, os, json, joblib, traceback
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# PyQt compatibility
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QTableView, QFileDialog,
        QMessageBox, QProgressBar, QTextEdit, QGroupBox, QGridLayout,
        QLineEdit, QComboBox, QDoubleSpinBox, QCheckBox,
        QSplitter, QScrollArea, QHeaderView, QSizePolicy, QInputDialog
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QAbstractTableModel, QModelIndex
    from PyQt6.QtGui import QFont
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QTableView, QFileDialog,
        QMessageBox, QProgressBar, QTextEdit, QGroupBox, QGridLayout,
        QLineEdit, QComboBox, QDoubleSpinBox, QCheckBox,
        QSplitter, QScrollArea, QHeaderView, QSizePolicy, QInputDialog
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QAbstractTableModel, QModelIndex
    from PyQt5.QtGui import QFont
    PYQT_VERSION = 5

from sklearn.model_selection import (cross_val_score, KFold, RepeatedStratifiedKFold,
                                     GroupKFold, train_test_split)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error)
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import PartialDependenceDisplay
from scipy.stats import boxcox, shapiro, kstest, friedmanchisquare, studentized_range

from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import scikit_posthocs as sp

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="resource_tracker", category=UserWarning)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def read_csv_robust(path, **kwargs):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'utf-16']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"File could not be read: {path}")

def ensure_derived_features(df, model_features):
    missing = [feat for feat in model_features if feat not in df.columns]
    if not missing:
        return df

    # CE_IIW
    if 'CE_IIW' in missing and all(c in df.columns for c in ['C', 'Mn']):
        df['CE_IIW'] = df['C'] + df['Mn'] / 6.0
        if 'Cr' in df.columns: df['CE_IIW'] += df['Cr'] / 5.0
        if 'Mo' in df.columns: df['CE_IIW'] += df['Mo'] / 5.0
        if 'Ni' in df.columns: df['CE_IIW'] += df['Ni'] / 15.0
        if 'Cu' in df.columns: df['CE_IIW'] += df['Cu'] / 15.0

    # TempParam
    if 'TempParam' in missing and all(c in df.columns for c in ['TT', 'Tt']):
        mask = df['Tt'] > 0
        df['TempParam'] = 0.0
        df.loc[mask, 'TempParam'] = df.loc[mask, 'TT'] * (np.log10(df.loc[mask, 'Tt']) + 20)

    # Pcm
    if 'Pcm' in missing and 'C' in df.columns:
        Cu = df['Cu'] if 'Cu' in df.columns else 0
        Si = df['Si'] if 'Si' in df.columns else 0
        Mn = df['Mn'] if 'Mn' in df.columns else 0
        Cr = df['Cr'] if 'Cr' in df.columns else 0
        Ni = df['Ni'] if 'Ni' in df.columns else 0
        Mo = df['Mo'] if 'Mo' in df.columns else 0
        df['Pcm'] = df['C'] + Si / 30.0 + (Mn + Cu + Cr) / 20.0 + Ni / 60.0 + Mo / 15.0

    # Interaction terms
    for feat in ['Mn_C', 'Cr_C', 'TT_C']:
        if feat in missing:
            elem = feat.split('_')[0]
            if elem in df.columns and 'C' in df.columns:
                df[feat] = df[elem] * df['C']

    # HI
    if 'HI' in missing:
      
        def calculate_grossmann_di(df_input, grain_size=7):
    
            df = df_input.copy()
            
            elements = ['C', 'Mn', 'Si', 'Ni', 'Cr', 'Mo']
            for el in elements:
                if el not in df.columns:
                    df[el] = 0.0
                  
            c = df['C']
            f_c = np.sqrt(c) * (0.54 * grain_size - 0.12) if grain_size else 0.24 * np.sqrt(c)
            
            f_mn = 1.0 + 3.3333 * df['Mn']  
            f_si = 1.0 + 0.7000 * df['Si']  
            f_ni = 1.0 + 0.3630 * df['Ni']  
            f_cr = 1.0 + 2.1600 * df['Cr']  
            f_mo = 1.0 + 3.0000 * df['Mo']  
        
            df['DI'] = f_c * f_mn * f_si * f_ni * f_cr * f_mo
            
            df['DI'] = df['DI'].round(4)
            
            return df

        df = calculate_grossmann_di(df, grain_size=7)
        df['HI'] = df['DI']

    # PSI
    if 'PSI' in missing:
        def _calc_psi(row):
            ct = row.get('Ct', 0.0)
            ct_temp = row.get('CT', 0.0)
            if ct > 0 and ct_temp > 0:
                T_k = ct_temp + 273.15
                D = 0.15 * np.exp(-143000 / (8.314 * T_k))
                depth_cm = np.sqrt(D * ct * 60.0)
                return round(depth_cm * 10.0, 4)
            return 0.0
        df['PSI'] = df.apply(_calc_psi, axis=1)

    # DCI
    if 'DCI' in missing:
        def _calc_dci(row):
            ct = row.get('Ct', 0.0)
            ct_temp = row.get('CT', 0.0)
            if ct > 0 and ct_temp > 0:
                return round(ct_temp * np.log10(ct + 1.0), 2)
            return 0.0
        df['DCI'] = df.apply(_calc_dci, axis=1)

    still_missing = [feat for feat in model_features if feat not in df.columns]
    if still_missing:
        raise ValueError("Missing features in CSV: " + ", ".join(still_missing))
    return df

def draw_cd_diagram(avg_ranks, model_names, cd, figure=None):
    if figure is not None:
        plt.figure(figure.number)
    else:
        plt.figure(figsize=(8, 4))
    ax = plt.gca()
    n_models = len(model_names)
    x_positions = avg_ranks
    y_positions = np.arange(n_models)
    for i in range(n_models):
        ax.plot([1, n_models], [i, i], color='gray', linestyle='-', linewidth=0.5)
    ax.scatter(x_positions, y_positions, s=80, marker='o', color='blue', zorder=3)
    for i, name in enumerate(model_names):
        ax.text(n_models + 0.2, i, f"{name} ({avg_ranks[i]:.2f})", va='center', fontsize=9)
    best_idx = np.argmin(avg_ranks)
    best_rank = avg_ranks[best_idx]
    for i in range(n_models):
        if i != best_idx and avg_ranks[i] - best_rank <= cd:
            ax.plot([best_rank, avg_ranks[i]], [best_idx, i], 'k-', linewidth=1.5)
    ax.plot([best_rank, best_rank + cd], [best_idx, best_idx], 'r-', linewidth=2, label=f'CD = {cd:.2f}')
    ax.legend()
    ax.set_xlabel('Average Rank')
    ax.set_yticks([])
    ax.set_title('Critical Difference (CD) Diagram')
    ax.invert_yaxis()
    ax.set_xlim(0.5, n_models + 2)
    plt.tight_layout()

# =============================================================================
# TRANSFORMATION PIPELINE (PowerTransformer + fit_features)
# =============================================================================
class TransformPipeline:
    def __init__(self):
        self.power_trans = None
        self.log1p_features = []
        self.feature_order = []   # final column order for the model
        self.fit_features = []    # ALL columns seen during fit (needed for transformation)

    def fit(self, X, skew_threshold=1.0):
        self.fit_features = list(X.columns)
        self.feature_order = self.fit_features[:]
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        boxcox_cols = []
        self.log1p_features = []

        for col in numeric_cols:
            col_min = X[col].min()
            skewness = X[col].skew()
            if abs(skewness) <= skew_threshold:
                continue
            if col_min > 0:
                boxcox_cols.append(col)
            elif col_min >= 0:
                self.log1p_features.append(col)

        if boxcox_cols:
            self.power_trans = PowerTransformer(method='box-cox', standardize=False)
            self.power_trans.fit(X[boxcox_cols])
        else:
            self.power_trans = None

    def transform(self, X):
        Xt = X.copy()
        if self.power_trans is not None:
            if self.fit_features:
                transform_cols = [c for c in self.fit_features if c in Xt.columns]
            else:
                transform_cols = [c for c in self.feature_order if c in Xt.columns]

            boxcox_cols = [c for c in self.power_trans.feature_names_in_ if c in Xt.columns]
            common_cols = [c for c in boxcox_cols if c in transform_cols]
            if common_cols:
                Xt[common_cols] = self.power_trans.transform(Xt[common_cols])

        for feat in self.log1p_features:
            if feat in Xt.columns:
                Xt[feat] = np.log1p(Xt[feat].clip(lower=0))

        order = self.feature_order if self.feature_order else self.fit_features
        if order:
            available = [c for c in order if c in Xt.columns]
            remaining = [c for c in Xt.columns if c not in order]
            Xt = Xt[available + remaining]
        return Xt

    @property
    def boxcox_lambdas(self):
        if self.power_trans is not None and hasattr(self.power_trans, 'lambdas_'):
            return dict(zip(self.power_trans.feature_names_in_, self.power_trans.lambdas_))
        return {}

# =============================================================================
# DATA MANAGER
# =============================================================================
class DataManager:
    def __init__(self):
        self.raw_df = None
        self.selected_features = None
        self.transform_pipeline = None
        self.X_train_raw = None
        self.X_test_raw = None
        self.y_train = None
        self.y_test = None
        self.X_train_trans = None
        self.X_test_trans = None
        self.best_model = None
        self.abc_best_params = None
        self.shap_explainer = None
        self.group_col = None
        self.target_col = 'Fatigue'
        self.id_col = 'Sl. No.'

    def load_data(self, path):
        self.raw_df = read_csv_robust(path)
        return self.raw_df

    def get_feature_cols(self):
        if self.raw_df is None:
            return []
        return [c for c in self.raw_df.columns if c not in [self.target_col, self.id_col]]

    def get_feature_matrix(self):
        if self.raw_df is None:
            return None
        return self.raw_df.drop(columns=[self.target_col, self.id_col], errors='ignore').copy()

    @staticmethod
    def _calculate_hi(row):
        c = row.get('C', 0.0)
        if c <= 0.20:       f_c = 0.15
        elif c <= 0.30:     f_c = 0.20
        elif c <= 0.40:     f_c = 0.25
        elif c <= 0.50:     f_c = 0.30
        elif c <= 0.60:     f_c = 0.35
        else:               f_c = 0.40
        f_gs = 1.0
        f_mn = 1.0 + 4.0 * row.get('Mn', 0.0)
        f_si = 1.0 + 1.0 * row.get('Si', 0.0)
        f_ni = 1.0 + 1.0 * row.get('Ni', 0.0)
        f_cr = 1.0 + 2.0 * row.get('Cr', 0.0)
        f_mo = 1.0 + 3.0 * row.get('Mo', 0.0)
        return round(f_c * f_gs * f_mn * f_si * f_ni * f_cr * f_mo, 4)

    @staticmethod
    def _calculate_psi(row):
        ct = row.get('Ct', 0.0)
        ct_temp = row.get('CT', 0.0)
        if ct > 0 and ct_temp > 0:
            T_k = ct_temp + 273.15
            D = 0.15 * np.exp(-143000 / (8.314 * T_k))
            depth_cm = np.sqrt(D * ct * 60.0)
            return round(depth_cm * 10.0, 4)
        return 0.0

    @staticmethod
    def _calculate_dci(row):
        ct = row.get('Ct', 0.0)
        ct_temp = row.get('CT', 0.0)
        if ct > 0 and ct_temp > 0:
            return round(ct_temp * np.log10(ct + 1.0), 2)
        return 0.0

    def add_metallurgical_features(self):
        df = self.raw_df
        if all(c in df.columns for c in ['C', 'Mn']):
            df['CE_IIW'] = df['C'] + df['Mn'] / 6.0
            if 'Cr' in df.columns: df['CE_IIW'] += df['Cr'] / 5.0
            if 'Mo' in df.columns: df['CE_IIW'] += df['Mo'] / 5.0
            if 'Ni' in df.columns: df['CE_IIW'] += df['Ni'] / 15.0
            if 'Cu' in df.columns: df['CE_IIW'] += df['Cu'] / 15.0
        if 'TT' in df.columns and 'Tt' in df.columns:
            mask = df['Tt'] > 0
            df['TempParam'] = 0.0
            df.loc[mask, 'TempParam'] = df.loc[mask, 'TT'] * (np.log10(df.loc[mask, 'Tt']) + 20)
        if 'C' in df.columns:
            Cu = df['Cu'] if 'Cu' in df.columns else 0
            Si = df['Si'] if 'Si' in df.columns else 0
            Mn = df['Mn'] if 'Mn' in df.columns else 0
            Cr = df['Cr'] if 'Cr' in df.columns else 0
            Ni = df['Ni'] if 'Ni' in df.columns else 0
            Mo = df['Mo'] if 'Mo' in df.columns else 0
            df['Pcm'] = df['C'] + Si / 30.0 + (Mn + Cu + Cr) / 20.0 + Ni / 60.0 + Mo / 15.0
        if 'Mn' in df.columns and 'C' in df.columns:
            df['Mn_C'] = df['Mn'] * df['C']
        if 'Cr' in df.columns and 'C' in df.columns:
            df['Cr_C'] = df['Cr'] * df['C']
        if 'TT' in df.columns and 'C' in df.columns:
            df['TT_C'] = df['TT'] * df['C']
        df['HI'] = df.apply(self._calculate_hi, axis=1)
        df['PSI'] = df.apply(self._calculate_psi, axis=1)
        df['DCI'] = df.apply(self._calculate_dci, axis=1)
        self.raw_df = df

    def split_data(self, X, y, test_size=0.2, random_state=42):
        bins = [0, 600, 900, np.inf]
        labels = [0, 1, 2]
        y_binned = pd.cut(y, bins=bins, labels=labels).astype(int)
        stratify = y_binned
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=stratify
        )
        self.X_train_raw = X_train
        self.X_test_raw = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.transform_pipeline = None
        self.X_train_trans = None
        self.X_test_trans = None
        self.selected_features = None

    def apply_transformation(self, skew_threshold=1.0):
        if self.X_train_raw is None or self.X_test_raw is None:
            raise ValueError("Please perform train/test split first (split_data).")
        pipeline = TransformPipeline()
        pipeline.fit(self.X_train_raw, skew_threshold=skew_threshold)
        self.transform_pipeline = pipeline
        self.X_train_trans = pipeline.transform(self.X_train_raw)
        self.X_test_trans = pipeline.transform(self.X_test_raw)

    def update_train_test_with_new_features(self):
        if self.X_train_raw is None or self.X_test_raw is None:
            return
        feature_df = self.raw_df.drop(columns=[self.target_col, self.id_col], errors='ignore')
        self.X_train_raw = feature_df.loc[self.X_train_raw.index]
        self.X_test_raw = feature_df.loc[self.X_test_raw.index]
        if self.transform_pipeline is not None:
            self.X_train_trans = self.transform_pipeline.transform(self.X_train_raw)
            self.X_test_trans = self.transform_pipeline.transform(self.X_test_raw)
        if self.selected_features is not None:
            new_cols = [c for c in self.X_train_raw.columns if c not in self.selected_features]
            if new_cols:
                self.selected_features.extend(new_cols)

    def get_cv_splitter(self, X, y):
        groups = self.raw_df[self.group_col].values if self.group_col else None
        if groups is not None:
            return GroupKFold(n_splits=5)
        else:
            try:
                kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                y_binned = kbins.fit_transform(y.values.reshape(-1, 1)).ravel()
                return RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
            except:
                return KFold(n_splits=5, shuffle=True, random_state=42)

# =============================================================================
# ABC OPTIMIZER AND WORKER
# =============================================================================
class ABCOptimizer:
    def __init__(self, X_train, y_train, param_space, pop_size=10, max_cycles=30):
        self.X_train = X_train
        self.y_train = y_train
        self.param_space = param_space
        self.param_keys = list(param_space.keys())
        self.dim = len(self.param_keys)
        self.pop_size = pop_size
        self.max_cycles = max_cycles
        self.limit = pop_size * self.dim
        self.best_fitness_history = []
        self.bounds = [(0, 1) for _ in range(self.dim)]

    def _decode_params(self, x):
        params = {}
        for i, key in enumerate(self.param_keys):
            spec = self.param_space[key]
            if spec[0] == 'continuous':
                low, high = spec[1], spec[2]
                if low > 0 and high > 0:
                    params[key] = 10 ** (np.log10(low) + x[i] * (np.log10(high) - np.log10(low)))
                else:
                    params[key] = low + x[i] * (high - low)
            elif spec[0] == 'categorical':
                choices = spec[1]
                idx = int(np.floor(x[i] * len(choices)))
                idx = max(0, min(idx, len(choices) - 1))
                params[key] = choices[idx]
            else:
                low, high = spec[1], spec[2]
                params[key] = int(round(low + x[i] * (high - low)))
        return params

    def _objective(self, x):
        params = self._decode_params(x)
        model = GradientBoostingRegressor(**params, random_state=42)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X_train, self.y_train,
                                 cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        return np.mean(scores)

    def optimize(self, progress_callback=None):
        foods = np.random.rand(self.pop_size, self.dim)
        fitness = np.zeros(self.pop_size)
        trials = np.zeros(self.pop_size, dtype=int)
        for i in range(self.pop_size):
            fitness[i] = self._objective(foods[i])
        best_idx = np.argmax(fitness)
        best_solution = foods[best_idx].copy()
        best_fitness = fitness[best_idx]

        for cycle in range(self.max_cycles):
            for i in range(self.pop_size):
                j = np.random.randint(self.dim)
                k = i
                while k == i:
                    k = np.random.randint(self.pop_size)
                phi = np.random.uniform(-1, 1)
                new_sol = foods[i].copy()
                new_sol[j] = np.clip(foods[i, j] + phi * (foods[i, j] - foods[k, j]), 0, 1)
                new_fit = self._objective(new_sol)
                if new_fit > fitness[i]:
                    foods[i] = new_sol
                    fitness[i] = new_fit
                    trials[i] = 0
                    if new_fit > best_fitness:
                        best_fitness = new_fit
                        best_solution = new_sol.copy()
                else:
                    trials[i] += 1

            min_fit = np.min(fitness)
            shifted = fitness - min_fit
            probs = shifted / shifted.sum() if shifted.sum() > 0 else np.ones(self.pop_size) / self.pop_size
            for _ in range(self.pop_size):
                r = np.random.rand()
                cum = 0.0
                selected = 0
                for idx, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        selected = idx
                        break
                j = np.random.randint(self.dim)
                k = selected
                while k == selected:
                    k = np.random.randint(self.pop_size)
                phi = np.random.uniform(-1, 1)
                new_sol = foods[selected].copy()
                new_sol[j] = np.clip(foods[selected, j] + phi * (foods[selected, j] - foods[k, j]), 0, 1)
                new_fit = self._objective(new_sol)
                if new_fit > fitness[selected]:
                    foods[selected] = new_sol
                    fitness[selected] = new_fit
                    trials[selected] = 0
                    if new_fit > best_fitness:
                        best_fitness = new_fit
                        best_solution = new_sol.copy()
                else:
                    trials[selected] += 1

            for i in range(self.pop_size):
                if trials[i] > self.limit:
                    foods[i] = np.random.rand(self.dim)
                    fitness[i] = self._objective(foods[i])
                    trials[i] = 0
                    if fitness[i] > best_fitness:
                        best_fitness = fitness[i]
                        best_solution = foods[i].copy()

            self.best_fitness_history.append(best_fitness)
            if progress_callback and (cycle + 1) % 5 == 0:
                progress_callback(f"ABC Cycle {cycle+1}/{self.max_cycles} - Best MAE: {-best_fitness:.4f} MPa")

        return self._decode_params(best_solution), best_fitness

class ABCWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, object)

    def __init__(self, X_train, y_train, param_space, pop_size=10, max_cycles=30):
        QThread.__init__(self)
        self.X_train = X_train
        self.y_train = y_train
        self.param_space = param_space
        self.pop_size = pop_size
        self.max_cycles = max_cycles

    def run(self):
        def progress_cb(msg):
            self.progress.emit(msg)
        optimizer = ABCOptimizer(self.X_train, self.y_train, self.param_space,
                                 pop_size=self.pop_size, max_cycles=self.max_cycles)
        best_params, best_score = optimizer.optimize(progress_callback=progress_cb)
        self.finished.emit(best_params, best_score, optimizer.best_fitness_history)

# =============================================================================
# NESTED CV WORKER
# =============================================================================
class NestedCVWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, X_train, y_train):
        QThread.__init__(self)
        self.X_train = X_train
        self.y_train = y_train

    def run(self):
        outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
        kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        y_binned = kbins.fit_transform(self.y_train.values.reshape(-1, 1)).ravel()

        outer_scores_mae = []
        outer_scores_r2 = []
        best_params_list = []

        X_np = self.X_train.values
        y_np = self.y_train.values

        for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_np, y_binned)):
            X_inner_train = X_np[train_idx]
            y_inner_train = y_np[train_idx]
            X_val = X_np[val_idx]
            y_val = y_np[val_idx]

            param_space = {
                'n_estimators': ('int', 50, 400),
                'max_depth': ('int', 3, 15),
                'learning_rate': ('continuous', 0.001, 0.5),
                'subsample': ('continuous', 0.5, 1.0),
                'min_samples_split': ('int', 2, 20),
                'min_samples_leaf': ('int', 1, 20),
                'max_features': ('categorical', ['sqrt', 'log2', None])
            }
            optimizer = ABCOptimizer(pd.DataFrame(X_inner_train, columns=self.X_train.columns),
                                     pd.Series(y_inner_train),
                                     param_space, pop_size=10, max_cycles=30)
            best_params, _ = optimizer.optimize()
            best_params_list.append(best_params)

            model = GradientBoostingRegressor(**best_params, random_state=42)
            model.fit(X_inner_train, y_inner_train)
            y_pred_val = model.predict(X_val)
            mae_val = mean_absolute_error(y_val, y_pred_val)
            r2_val = r2_score(y_val, y_pred_val)
            outer_scores_mae.append(mae_val)
            outer_scores_r2.append(r2_val)

            self.progress.emit(f"  Fold {fold+1}: MAE={mae_val:.2f} MPa, R²={r2_val:.4f}")

        mean_mae = np.mean(outer_scores_mae)
        std_mae = np.std(outer_scores_mae)
        mean_r2 = np.mean(outer_scores_r2)
        std_r2 = np.std(outer_scores_r2)

        result = {
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'best_params_list': best_params_list,
        }
        self.finished.emit(result)

# =============================================================================
# BOOTSTRAP WORKER
# =============================================================================
class BootstrapWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)

    def __init__(self, X_train, y_train, X_test, y_test, params, n_bootstrap, confidence):
        QThread.__init__(self)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.params = params
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence

    def run(self):
        n_samples = len(self.X_train)
        bootstrap_preds = []
        for i in range(self.n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = self.X_train.iloc[indices]
            y_boot = self.y_train.iloc[indices]

            model = GradientBoostingRegressor(**self.params, random_state=i)
            model.fit(X_boot, y_boot)
            pred = model.predict(self.X_test)
            bootstrap_preds.append(pred)

            if (i + 1) % 20 == 0:
                self.progress.emit(i + 1, f"Bootstrap iteration: {i+1}/{self.n_bootstrap}")

        bootstrap_preds = np.array(bootstrap_preds)
        alpha = (1 - self.confidence) / 2
        lower = np.percentile(bootstrap_preds, 100 * alpha, axis=0)
        upper = np.percentile(bootstrap_preds, 100 * (1 - alpha), axis=0)
        mean_pred = np.mean(bootstrap_preds, axis=0)

        results_df = pd.DataFrame({
            'Actual (MPa)': self.y_test.values,
            'Predicted (mean)': np.round(mean_pred, 2),
            f'Lower Bound ({100*alpha:.1f}%)': np.round(lower, 2),
            f'Upper Bound ({100*(1-alpha):.1f}%)': np.round(upper, 2),
            'Interval Width': np.round(upper - lower, 2)
        })

        coverage = np.mean((self.y_test.values >= lower) & (self.y_test.values <= upper)) * 100
        mean_width = np.mean(upper - lower)

        result = {
            'results_df': results_df,
            'coverage': coverage,
            'mean_width': mean_width,
            'lower': lower,
            'upper': upper,
            'mean_pred': mean_pred
        }
        self.finished.emit(result)

# =============================================================================
# MODEL COMPARISON WORKER
# =============================================================================
class ModelComparisonWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)

    def __init__(self, X_train, X_test, y_train, y_test, abc_best_params, cv_splitter):
        QThread.__init__(self)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.abc_best_params = abc_best_params
        self.cv_splitter = cv_splitter

    def run(self):
        gbm_abc = GradientBoostingRegressor(**self.abc_best_params, random_state=42)

        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "GBM + ABC": gbm_abc,
            "SVR (RBF)": Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf'))]),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1, n_jobs=1)
        }

        results = []
        predictions = {}
        cv_scores_dict = {}

        for idx, (name, model) in enumerate(models.items()):
            self.progress.emit(idx + 1, f"Training: {name} ...")

            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv_splitter,
                                        scoring='neg_mean_absolute_error', n_jobs=-1)
            cv_scores_dict[name] = -cv_scores

            model.fit(self.X_train, self.y_train)
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)

            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae  = mean_absolute_error(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse  = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_mape = mean_absolute_percentage_error(self.y_train, y_pred_train) * 100
            test_mape  = mean_absolute_percentage_error(self.y_test, y_pred_test) * 100
            train_r2   = r2_score(self.y_train, y_pred_train)
            test_r2    = r2_score(self.y_test, y_pred_test)

            results.append({
                "Model": name,
                "Train MAE": round(train_mae, 3),
                "Test MAE": round(test_mae, 3),
                "Train RMSE": round(train_rmse, 3),
                "Test RMSE": round(test_rmse, 3),
                "Train R²": round(train_r2, 4),
                "Test R²": round(test_r2, 4),
                "Train MAPE (%)": round(train_mape, 2),
                "Test MAPE (%)": round(test_mape, 2)
            })
            predictions[name] = y_pred_test

        result_package = {
            'results': results,
            'predictions': predictions,
            'cv_scores_dict': cv_scores_dict,
            'y_test': self.y_test.values
        }
        self.finished.emit(result_package)

# =============================================================================
# PANDAS TABLE MODEL
# =============================================================================
class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._df = df.copy()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        col = index.column()
        row = index.row()
        value = self._df.iloc[row, col]
        if role == (Qt.ItemDataRole.DisplayRole if PYQT_VERSION == 6 else Qt.DisplayRole):
            if isinstance(value, (float, np.floating)):
                return "{:.4f}".format(value)
            return str(value)
        elif role == (Qt.ItemDataRole.TextAlignmentRole if PYQT_VERSION == 6 else Qt.TextAlignmentRole):
            return int(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == (Qt.ItemDataRole.DisplayRole if PYQT_VERSION == 6 else Qt.DisplayRole):
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            else:
                return str(self._df.index[section])
        return None

    def setDataFrame(self, df):
        self.beginResetModel()
        self._df = df.copy()
        self.endResetModel()

# =============================================================================
# TAB 1: DATA LOADING
# =============================================================================
class DataTab(QWidget):
    FEATURE_DESCRIPTIONS = {
        "Sl. No.": "Sample serial number (ID column)",
        "NT": "Normalizing temperature (°C)",
        "THT": "Through hardening temperature (°C)",
        "THt": "Through hardening time (min)",
        "THQCr": "Cooling rate for through hardening",
        "CT": "Carburizing temperature (°C)",
        "Ct": "Carburizing time (min)",
        "DT": "Diffusion temperature (°C)",
        "Dt": "Diffusion time (min)",
        "QmT": "Quenching media temperature – for carburizing (°C)",
        "TT": "Tempering temperature (°C)",
        "Tt": "Tempering time (min)",
        "TCr": "Cooling rate for tempering",
        "C": "Carbon (wt%)",
        "Si": "Silicon (wt%)",
        "Mn": "Manganese (wt%)",
        "P": "Phosphorus (wt%)",
        "S": "Sulphur (wt%)",
        "Ni": "Nickel (wt%)",
        "Cr": "Chromium (wt%)",
        "Cu": "Copper (wt%)",
        "Mo": "Molybdenum (wt%)",
        "RedRatio": "Reduction ratio (Ingot → Bar)",
        "dA": "Area proportion of inclusions deformed by plastic work",
        "dB": "Area proportion of inclusions occurring in discontinuous array",
        "dC": "Area proportion of isolated inclusions",
        "Fatigue": "Rotating bending fatigue strength at 10⁷ cycles (MPa)"
    }

    def __init__(self, data_manager, parent=None):
        QWidget.__init__(self, parent)
        self.dm = data_manager
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        self.btn_load = QPushButton("Load CSV")
        self.btn_load.setStyleSheet("font-weight: bold; padding: 8px; background-color: #1976D2; color: white;")
        self.btn_load.clicked.connect(self.load_csv)
        ctrl.addWidget(self.btn_load)
        self.lbl_info = QLabel("Dataset not loaded")
        ctrl.addWidget(self.lbl_info)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        table_group = QGroupBox("Data Table")
        tv_layout = QVBoxLayout(table_group)
        self.table_view = QTableView()
        self.table_view.setAlternatingRowColors(True)
        self.table_model = PandasModel()
        self.table_view.setModel(self.table_model)
        tv_layout.addWidget(self.table_view)
        layout.addWidget(table_group)

        stats_group = QGroupBox("Descriptive Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_table = QTableView()
        self.stats_model = PandasModel()
        self.stats_table.setModel(self.stats_model)
        self.stats_table.horizontalHeader().setStretchLastSection(False)
        self.stats_table.setAlternatingRowColors(True)
        stats_layout.addWidget(self.stats_table)
        layout.addWidget(stats_group)

        miss_group = QGroupBox("Missing Values / Duplicates")
        miss_layout = QVBoxLayout(miss_group)
        self.missing_text = QTextEdit()
        self.missing_text.setReadOnly(True)
        self.missing_text.setMaximumHeight(80)
        miss_layout.addWidget(self.missing_text)
        layout.addWidget(miss_group)

        app_group = QGroupBox("Applicability Domain")
        app_layout = QVBoxLayout(app_group)
        self.app_table = QTableView()
        self.app_model = PandasModel(pd.DataFrame())
        self.app_table.setModel(self.app_model)
        self.app_table.horizontalHeader().setStretchLastSection(False)
        app_layout.addWidget(self.app_table)
        layout.addWidget(app_group)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if path:
            try:
                df = self.dm.load_data(path)
                self.table_model.setDataFrame(df)
                self.lbl_info.setText(f"{os.path.basename(path)} | {df.shape[0]} rows x {df.shape[1]} columns")
                self.lbl_info.setStyleSheet("color: green; font-weight: bold;")
                self.update_stats(df)
                self.update_applicability_domain()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"File could not be loaded:\n{str(e)}")

    def update_stats(self, df):
        desc_df = df.describe().transpose()
        desc_df.reset_index(inplace=True)
        desc_df.rename(columns={'index': 'Feature'}, inplace=True)
        num_cols = desc_df.select_dtypes(include=[np.number]).columns
        desc_df[num_cols] = desc_df[num_cols].round(4)
        self.stats_model.setDataFrame(desc_df)

        total_missing = df.isnull().sum().sum()
        duplicated = df.duplicated().sum()
        miss_lines = [f"Total Missing Values: {total_missing}",
                      f"Duplicate Rows: {duplicated}"]
        col_missing = df.isnull().sum()
        col_missing = col_missing[col_missing > 0]
        if len(col_missing) > 0:
            miss_lines.append("Missing Values by Column:")
            for col, n in col_missing.items():
                miss_lines.append(f"  {col}: {n} (%{100*n/len(df):.2f})")
        else:
            miss_lines.append("No missing values found.")
        self.missing_text.setText("\n".join(miss_lines))

    def update_applicability_domain(self):
        X = self.dm.X_train_raw if self.dm.X_train_raw is not None else self.dm.get_feature_matrix()
        if X is None:
            return
        num_cols = X.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return
        stats = X[num_cols].describe().transpose()
        domain_df = pd.DataFrame({
            'Feature': stats.index,
            'Minimum': stats['min'].round(4),
            'Maximum': stats['max'].round(4)
        })
        domain_df['Feature Description'] = domain_df['Feature'].map(self.FEATURE_DESCRIPTIONS).fillna('—')
        domain_df = domain_df[['Feature', 'Feature Description', 'Minimum', 'Maximum']]
        self.app_model.setDataFrame(domain_df)

# =============================================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (TABBED)
# =============================================================================
class EDATab(QWidget):
    def __init__(self, data_manager, parent=None):
        QWidget.__init__(self, parent)
        self.dm = data_manager
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        self.btn_corr = QPushButton("Correlation Matrix")
        self.btn_corr.clicked.connect(self.plot_correlation)
        ctrl.addWidget(self.btn_corr)
        self.btn_box = QPushButton("Boxplot")
        self.btn_box.clicked.connect(self.plot_boxplot)
        ctrl.addWidget(self.btn_box)
        self.btn_scatter = QPushButton("Fatigue Scatter")
        self.btn_scatter.clicked.connect(self.plot_scatter)
        ctrl.addWidget(self.btn_scatter)
        self.lbl_thresh = QLabel("Corr. Threshold:")
        ctrl.addWidget(self.lbl_thresh)
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.0, 1.0)
        self.spin_thresh.setValue(0.3)
        ctrl.addWidget(self.spin_thresh)
        self.btn_top = QPushButton("Highly Correlated")
        self.btn_top.clicked.connect(self.show_top_features)
        ctrl.addWidget(self.btn_top)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.graph_tabs = QTabWidget()

        # Correlation tab
        corr_tab = QWidget()
        corr_layout = QVBoxLayout(corr_tab)
        self.corr_figure = plt.figure(figsize=(8, 6))
        self.corr_canvas = FigureCanvas(self.corr_figure)
        self.corr_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.corr_toolbar = NavigationToolbar(self.corr_canvas, self)
        corr_layout.addWidget(self.corr_toolbar)
        self.corr_scroll = QScrollArea()
        self.corr_scroll.setWidgetResizable(True)
        self.corr_scroll.setWidget(self.corr_canvas)
        corr_layout.addWidget(self.corr_scroll)
        self.graph_tabs.addTab(corr_tab, "Correlation Matrix")

        # Boxplot tab
        box_tab = QWidget()
        box_layout = QVBoxLayout(box_tab)
        self.box_figure = plt.figure(figsize=(8, 6))
        self.box_canvas = FigureCanvas(self.box_figure)
        self.box_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.box_toolbar = NavigationToolbar(self.box_canvas, self)
        box_layout.addWidget(self.box_toolbar)
        self.box_scroll = QScrollArea()
        self.box_scroll.setWidgetResizable(True)
        self.box_scroll.setWidget(self.box_canvas)
        box_layout.addWidget(self.box_scroll)
        self.graph_tabs.addTab(box_tab, "Boxplot")

        # Scatter tab
        scat_tab = QWidget()
        scat_layout = QVBoxLayout(scat_tab)
        self.scat_figure = plt.figure(figsize=(10, 8))
        self.scat_canvas = FigureCanvas(self.scat_figure)
        self.scat_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scat_toolbar = NavigationToolbar(self.scat_canvas, self)
        scat_layout.addWidget(self.scat_toolbar)
        self.scat_scroll = QScrollArea()
        self.scat_scroll.setWidgetResizable(True)
        self.scat_scroll.setWidget(self.scat_canvas)
        scat_layout.addWidget(self.scat_scroll)
        self.graph_tabs.addTab(scat_tab, "Fatigue Scatter")

        layout.addWidget(self.graph_tabs)

        self.text_output = QTextEdit()
        self.text_output.setMaximumHeight(100)
        self.text_output.setReadOnly(True)
        layout.addWidget(self.text_output)

    def check_data(self):
        if self.dm.raw_df is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return False
        return True

    def plot_correlation(self):
        if not self.check_data(): return
        df = self.dm.raw_df.drop(columns=[self.dm.id_col], errors='ignore')
        self.corr_figure.clear()
        ax = self.corr_figure.add_subplot(111)
        corr = df.corr()

        n = len(corr.columns)
        cell_size = 0.45
        fig_width = n * cell_size
        fig_height = n * cell_size * 0.95
        self.corr_figure.set_size_inches(fig_width, fig_height)

        im = ax.imshow(corr, cmap='coolwarm', aspect='equal', vmin=-1, vmax=1)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(corr.iloc[i, j]) > 0.7 else "black", fontsize=6)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=7, ha='center')
        ax.set_yticklabels(corr.columns, fontsize=7)
        self.corr_figure.colorbar(im, ax=ax)
        ax.set_title('Correlation Matrix', fontsize=12)
        self.corr_figure.subplots_adjust(bottom=0.2, left=0.2, top=0.95, right=0.95)
        self.corr_canvas.setMinimumSize(int(fig_width * self.corr_figure.dpi),
                                        int(fig_height * self.corr_figure.dpi))
        self.corr_canvas.draw()
        self.graph_tabs.setCurrentIndex(0)

    def plot_boxplot(self):
        if not self.check_data(): return
        df = self.dm.raw_df.drop(columns=[self.dm.id_col], errors='ignore')
        numeric = df.select_dtypes(include=[np.number]).columns
        n = len(numeric)
        fig_width = max(8, n * 0.4)
        fig_height = 6
        self.box_figure.set_size_inches(fig_width, fig_height)
        self.box_figure.clear()
        ax = self.box_figure.add_subplot(111)
        df[numeric].boxplot(ax=ax, rot=45, grid=True)
        ax.set_title('Boxplot')
        self.box_figure.tight_layout()
        self.box_canvas.setMinimumSize(int(fig_width * self.box_figure.dpi),
                                       int(fig_height * self.box_figure.dpi))
        self.box_canvas.draw()
        self.graph_tabs.setCurrentIndex(1)

        outlier_counts = {}
        for col in numeric:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            outlier_counts[col] = ((df[col] < lb) | (df[col] > ub)).sum()
        out = ["Outlier Counts (IQR):"] + [f"{c}: {v}" for c, v in outlier_counts.items()]
        self.text_output.setText("\n".join(out))

    def plot_scatter(self):
        if not self.check_data(): return
        df = self.dm.raw_df
        target = self.dm.target_col
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c != target]
        if not features:
            QMessageBox.information(self, "Info", "No numerical features to display.")
            return
        n = len(features)
        cols = 2
        rows = (n + cols - 1) // cols
        fig_width = max(10, cols * 4.0)
        fig_height = max(6, rows * 3.5)
        self.scat_figure.set_size_inches(fig_width, fig_height)
        self.scat_canvas.setMinimumSize(int(fig_width * self.scat_figure.dpi),
                                        int(fig_height * self.scat_figure.dpi))
        self.scat_figure.clear()
        for i, feat in enumerate(features):
            ax = self.scat_figure.add_subplot(rows, cols, i + 1)
            ax.scatter(df[feat], df[target], alpha=0.6, s=20,
                       c='steelblue', edgecolors='black', linewidth=0.3)
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel(target, fontsize=8)
            corr = df[feat].corr(df[target])
            ax.set_title(f"r = {corr:.3f}", fontsize=9)
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle='--', alpha=0.4)
        self.scat_figure.tight_layout(pad=1.2)
        self.scat_canvas.draw()
        self.graph_tabs.setCurrentIndex(2)

    def show_top_features(self):
        if not self.check_data(): return
        thresh = self.spin_thresh.value()
        df = self.dm.raw_df
        corr = df.corr()[self.dm.target_col].abs().sort_values(ascending=False)
        top = [c for c in corr.index if corr[c] > thresh and c not in [self.dm.target_col, self.dm.id_col]]
        out = [f"Correlation Threshold > {thresh}:"] + [f"  {f}: {df[f].corr(df[self.dm.target_col]):+.4f}" for f in top]
        out.append(f"Total {len(top)} features")
        self.text_output.setText("\n".join(out))

# =============================================================================
# TAB 3: FEATURE SELECTION (TABBED)
# =============================================================================
class FeatureSelectionTab(QWidget):
    def __init__(self, data_manager, parent=None):
        QWidget.__init__(self, parent)
        self.dm = data_manager
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        feat_grp = QGroupBox("Feature Selection (Training Data Only)")
        feat_layout = QVBoxLayout(feat_grp)

        ctrl = QHBoxLayout()
        self.btn_split = QPushButton("Train/Test Split (80/20)")
        self.btn_split.clicked.connect(self.split_data)
        self.btn_split.setStyleSheet("font-weight: bold; background-color: #1976D2; color: white;")
        ctrl.addWidget(self.btn_split)

        self.btn_vif = QPushButton("Calculate VIF (Training)")
        self.btn_vif.clicked.connect(self.calc_vif_train)
        ctrl.addWidget(self.btn_vif)

        self.btn_shap = QPushButton("SHAP Importance (Training)")
        self.btn_shap.clicked.connect(self.calc_importance_train)
        ctrl.addWidget(self.btn_shap)

        self.btn_eng = QPushButton("Add Metallurgical Features (All Data)")
        self.btn_eng.clicked.connect(self.add_metallurgical_features)
        ctrl.addWidget(self.btn_eng)

        self.lbl_shap_th = QLabel("SHAP Threshold:")
        ctrl.addWidget(self.lbl_shap_th)
        self.spin_shap_th = QDoubleSpinBox()
        self.spin_shap_th.setDecimals(4)
        self.spin_shap_th.setRange(0.0, 100)
        self.spin_shap_th.setValue(1.0)
        ctrl.addWidget(self.spin_shap_th)
        self.chk_auto_shap = QCheckBox("Auto (cumulative 95%)")
        self.chk_auto_shap.setChecked(True)
        ctrl.addWidget(self.chk_auto_shap)

        self.lbl_vif_th = QLabel("VIF Threshold:")
        ctrl.addWidget(self.lbl_vif_th)
        self.spin_vif_th = QDoubleSpinBox()
        self.spin_vif_th.setRange(1.0, 1000)
        self.spin_vif_th.setValue(10.0)
        ctrl.addWidget(self.spin_vif_th)

        self.btn_shap_filter = QPushButton("Filter by SHAP")
        self.btn_shap_filter.clicked.connect(self.filter_train_by_shap)
        ctrl.addWidget(self.btn_shap_filter)
        self.btn_vif_filter = QPushButton("Filter by VIF")
        self.btn_vif_filter.clicked.connect(self.filter_train_by_vif)
        ctrl.addWidget(self.btn_vif_filter)
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self.select_all_train_features)
        ctrl.addWidget(self.btn_select_all)
        ctrl.addStretch()
        feat_layout.addLayout(ctrl)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.feat_table = QTableView()
        self.feat_model = PandasModel()
        self.feat_table.setModel(self.feat_model)
        left_layout.addWidget(QLabel("Training Data Feature Importance / VIF:"))
        left_layout.addWidget(self.feat_table)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.graph_tabs = QTabWidget()

        shap_tab = QWidget()
        shap_layout = QVBoxLayout(shap_tab)
        self.shap_figure = plt.figure(figsize=(6, 5))
        self.shap_canvas = FigureCanvas(self.shap_figure)
        self.shap_toolbar = NavigationToolbar(self.shap_canvas, self)
        shap_layout.addWidget(self.shap_toolbar)
        shap_layout.addWidget(self.shap_canvas)
        self.graph_tabs.addTab(shap_tab, "SHAP Importance")

        vif_tab = QWidget()
        vif_layout = QVBoxLayout(vif_tab)
        self.vif_figure = plt.figure(figsize=(6, 5))
        self.vif_canvas = FigureCanvas(self.vif_figure)
        self.vif_toolbar = NavigationToolbar(self.vif_canvas, self)
        vif_layout.addWidget(self.vif_toolbar)
        vif_layout.addWidget(self.vif_canvas)
        self.graph_tabs.addTab(vif_tab, "VIF")

        right_layout.addWidget(self.graph_tabs)
        splitter.addWidget(right)

        feat_layout.addWidget(splitter)
        layout.addWidget(feat_grp)

    # ----------------------------------------------------------------
    def check_data(self):
        if self.dm.raw_df is None:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return False
        return True

    def _get_train_data(self):
        if self.dm.X_train_raw is None or self.dm.y_train is None:
            QMessageBox.warning(self, "Warning", "Please perform Train/Test Split first!")
            return None, None
        return self.dm.X_train_raw, self.dm.y_train

    def split_data(self):
        if not self.check_data(): return
        X = self.dm.get_feature_matrix()
        y = self.dm.raw_df[self.dm.target_col]
        self.dm.split_data(X, y, test_size=0.2, random_state=42)
        self.dm.selected_features = list(self.dm.X_train_raw.columns)
        self.log_text.clear()
        self.log_text.append("Train/Test split completed.")
        self.log_text.append(f"Train: {self.dm.X_train_raw.shape[0]} samples, Test: {self.dm.X_test_raw.shape[0]} samples")
        self.log_text.append(f"All features ({len(self.dm.selected_features)}) selected: {self.dm.selected_features}")
        try:
            pred_tab = self.window().tab_pred
            pred_tab.load_test_data(self.dm.X_test_raw, self.dm.y_test)
        except:
            pass
        try:
            self.window().tab_data.update_applicability_domain()
        except:
            pass

    def calc_importance_train(self):
        X_train, y_train = self._get_train_data()
        if X_train is None: return
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        importances = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": np.abs(shap_values).mean(axis=0)
        }).sort_values("Importance", ascending=False)
        method = "SHAP (GBM - Training)"
        self.feat_model.setDataFrame(importances)

        if self.chk_auto_shap.isChecked():
            total_imp = importances["Importance"].sum()
            cumsum = importances["Importance"].cumsum() / total_imp
            idx = np.argmax(cumsum >= 0.95)
            auto_threshold = importances["Importance"].iloc[idx]
            self.spin_shap_th.setValue(auto_threshold)
            threshold = auto_threshold
            self.log_text.append(f"Auto SHAP threshold (cumulative 95%): {auto_threshold:.4f}")
        else:
            threshold = self.spin_shap_th.value()

        low = importances[importances["Importance"] < threshold]["Feature"].tolist()
        high = importances[importances["Importance"] >= threshold]["Feature"].tolist()
        self.log_text.append(f"{method} calculated.")
        self.log_text.append(f"Below SHAP threshold ({threshold}): {low}")
        self.log_text.append(f"Above SHAP threshold ({threshold}): {high}")

        self.shap_figure.clear()
        ax = self.shap_figure.add_subplot(111)
        bars = ax.barh(importances["Feature"], importances["Importance"],
                       color='steelblue', edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, importances["Importance"]):
            ax.text(bar.get_width() + 0.01 * importances["Importance"].max(),
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=7, color='black')
        ax.set_title(f'Feature Importance ({method})\nTraining Data, {len(X_train)} samples, {len(X_train.columns)} features',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Mean |SHAP| Value', fontsize=9)
        ax.set_ylabel('Feature', fontsize=9)
        if threshold > 0:
            ax.axvline(x=threshold, color='red', linestyle='--', linewidth=1.2, alpha=0.8,
                       label=f'Threshold (< {threshold:.4f})')
            ax.legend(loc='lower right', fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.tick_params(axis='y', labelsize=8)
        self.shap_figure.tight_layout(pad=1.5)
        self.shap_canvas.draw()
        self.graph_tabs.setCurrentIndex(0)

    def calc_vif_train(self):
        X_train, _ = self._get_train_data()
        if X_train is None: return
        vif_data = pd.DataFrame({
            "Feature": X_train.columns,
            "VIF": [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
        })
        vif_data = vif_data.sort_values("VIF", ascending=False)
        self.feat_model.setDataFrame(vif_data)
        self.log_text.clear()
        self.log_text.append("VIF calculated (Training).")
        threshold = self.spin_vif_th.value()
        high_vif = vif_data[vif_data["VIF"] > threshold]["Feature"].tolist()
        self.log_text.append(f"VIF > {threshold}: {high_vif}")

        self.vif_figure.clear()
        ax = self.vif_figure.add_subplot(111)
        vif_plot = vif_data.copy()
        vif_plot["Log VIF"] = np.log1p(vif_plot["VIF"])
        colors = ['red' if v > 10 else 'orange' if v > 2 else 'green' for v in vif_plot["VIF"]]
        ax.barh(vif_plot["Feature"], vif_plot["Log VIF"], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel("Log(1+VIF)", fontsize=9)
        ax.set_ylabel("Feature", fontsize=9)
        ax.set_title("VIF Scores (High → Multicollinearity)", fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        self.vif_figure.tight_layout(pad=1.5)
        self.vif_canvas.draw()
        self.graph_tabs.setCurrentIndex(1)

    def filter_train_by_shap(self):
        X_train, _ = self._get_train_data()
        if X_train is None: return
        imp_df = self.feat_model._df
        if imp_df.empty or "Importance" not in imp_df.columns:
            QMessageBox.warning(self, "Warning", "Please calculate SHAP Importance first.")
            return
        threshold = self.spin_shap_th.value()
        low = imp_df[imp_df["Importance"] < threshold]["Feature"].tolist()
        selected = [f for f in X_train.columns if f not in low]
        self.dm.selected_features = selected
        self.log_text.append(f"SHAP filter (threshold < {threshold}), removed: {low}")
        self.log_text.append(f"Remaining ({len(selected)}): {selected}")

    def filter_train_by_vif(self):
        X_train, _ = self._get_train_data()
        if X_train is None: return
        vif_df = self.feat_model._df
        if vif_df.empty or "VIF" not in vif_df.columns:
            QMessageBox.warning(self, "Warning", "Please calculate VIF first.")
            return
        threshold = self.spin_vif_th.value()
        high = vif_df[vif_df["VIF"] > threshold]["Feature"].tolist()
        selected = [f for f in X_train.columns if f not in high]
        self.dm.selected_features = selected
        self.log_text.append(f"VIF filter (VIF > {threshold}), removed: {high}")
        self.log_text.append(f"Remaining ({len(selected)}): {selected}")

    def select_all_train_features(self):
        X_train, _ = self._get_train_data()
        if X_train is None: return
        self.dm.selected_features = list(X_train.columns)
        self.log_text.append(f"All features ({len(self.dm.selected_features)}) selected: {self.dm.selected_features}")

    def add_metallurgical_features(self):
        if not self.check_data(): return
        old_cols = set(self.dm.raw_df.columns)
        self.dm.add_metallurgical_features()
        new_cols = set(self.dm.raw_df.columns)
        added = sorted(list(new_cols - old_cols))
        self.log_text.clear()
        self.log_text.append("Metallurgical features added (to all data).")
        self.log_text.append(f"Added features: {added}")
        self.dm.update_train_test_with_new_features()
        if self.dm.X_train_raw is not None:
            self.log_text.append(f"Train/test sets updated. Current feature count: {self.dm.X_train_raw.shape[1]}")
            if self.dm.selected_features is not None:
                self.dm.selected_features = list(self.dm.X_train_raw.columns)
        QMessageBox.information(self, "Success", f"Metallurgical features added:\n{', '.join(added)}")

# =============================================================================
# TAB 4: OUTLIER DETECTION & TRANSFORMATION (test table not updated)
# =============================================================================
class OutlierTab(QWidget):
    def __init__(self, data_manager, parent=None):
        QWidget.__init__(self, parent)
        self.dm = data_manager
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        self.btn_norm = QPushButton("Normality Tests")
        self.btn_norm.clicked.connect(self.normality_tests)
        ctrl.addWidget(self.btn_norm)

        self.btn_iqr = QPushButton("IQR Analysis")
        self.btn_iqr.clicked.connect(self.iqr_analysis)
        ctrl.addWidget(self.btn_iqr)

        self.btn_skew = QPushButton("Skewness")
        self.btn_skew.clicked.connect(self.skewness_analysis)
        ctrl.addWidget(self.btn_skew)

        self.btn_transform = QPushButton("Apply Transformation (Box‑Cox, Log1p)")
        self.btn_transform.clicked.connect(self.apply_transformation)
        self.btn_transform.setStyleSheet("font-weight: bold; background-color: #E91E63; color: white;")
        ctrl.addWidget(self.btn_transform)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.table_view = QTableView()
        self.table_model = PandasModel()
        self.table_view.setModel(self.table_model)
        left_layout.addWidget(QLabel("Analysis Results:"))
        left_layout.addWidget(self.table_view)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.graph_tabs = QTabWidget()

        iqr_tab = QWidget()
        iqr_layout = QVBoxLayout(iqr_tab)
        self.iqr_figure = plt.figure(figsize=(7, 5))
        self.iqr_canvas = FigureCanvas(self.iqr_figure)
        self.iqr_toolbar = NavigationToolbar(self.iqr_canvas, self)
        iqr_layout.addWidget(self.iqr_toolbar)
        iqr_layout.addWidget(self.iqr_canvas)
        self.graph_tabs.addTab(iqr_tab, "IQR Analysis")

        skew_tab = QWidget()
        skew_layout = QVBoxLayout(skew_tab)
        self.skew_figure = plt.figure(figsize=(7, 5))
        self.skew_canvas = FigureCanvas(self.skew_figure)
        self.skew_toolbar = NavigationToolbar(self.skew_canvas, self)
        skew_layout.addWidget(self.skew_toolbar)
        skew_layout.addWidget(self.skew_canvas)
        self.graph_tabs.addTab(skew_tab, "Skewness")

        right_layout.addWidget(self.graph_tabs)
        splitter.addWidget(right)

        layout.addWidget(splitter)

    def get_selected_df(self):
        df = None
        if self.dm.X_train_raw is not None:
            df = self.dm.X_train_raw.copy()
        else:
            df = self.dm.get_feature_matrix()
        if df is not None and self.dm.selected_features is not None:
            cols = [c for c in self.dm.selected_features if c in df.columns]
            if cols:
                df = df[cols]
        return df

    def _get_raw_data(self):
        if self.dm.X_train_raw is not None:
            return self.dm.X_train_raw.copy()
        return self.dm.get_feature_matrix()

    def apply_transformation(self):
        if self.dm.X_train_raw is None:
            QMessageBox.warning(self, "Warning", "Please perform Train/Test Split in Tab 3 first!")
            return
        try:
            self.dm.apply_transformation(skew_threshold=1.0)
            self.log_text.append("Transformation (Box‑Cox, Log1p) applied successfully.")
            if self.dm.transform_pipeline:
                boxcox_feats = list(self.dm.transform_pipeline.boxcox_lambdas.keys())
                log1p_feats = self.dm.transform_pipeline.log1p_features
                self.log_text.append(f"Box‑Cox applied to: {boxcox_feats if boxcox_feats else 'None'}")
                self.log_text.append(f"Log1p applied to: {log1p_feats if log1p_feats else 'None'}")

            QMessageBox.information(self, "Success", "Transformation applied.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def normality_tests(self):
        df = self.get_selected_df()
        if df is None: return
        numeric = df.select_dtypes(include=[np.number]).columns
        results = []
        for col in numeric:
            data = df[col].dropna()
            if len(data) < 3: continue
            try: sw_stat, sw_p = shapiro(data)
            except: sw_stat, sw_p = np.nan, np.nan
            try: ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
            except: ks_stat, ks_p = np.nan, np.nan
            results.append({"Feature": col, "SW p": sw_p, "KS p": ks_p,
                            "Normal (0.05)": "Yes" if sw_p>0.05 else "No"})
        self.table_model.setDataFrame(pd.DataFrame(results))
        self.log_text.setText(f"Normality tests completed for {len(results)} features.")

    def iqr_analysis(self):
        df = self.get_selected_df()
        if df is None: return
        numeric = df.select_dtypes(include=[np.number]).columns
        results = []
        outlier_counts = {}
        for col in numeric:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            cnt = ((df[col] < lb) | (df[col] > ub)).sum()
            outlier_counts[col] = cnt
            results.append({"Feature": col, "Outlier Count": cnt})
        self.table_model.setDataFrame(pd.DataFrame(results))

        self.iqr_figure.clear()
        ax = self.iqr_figure.add_subplot(111)
        cols = list(outlier_counts.keys())[:12]
        counts = [outlier_counts[c] for c in cols]
        bars = ax.bar(cols, counts, color='coral', edgecolor='black', linewidth=0.5)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(cnt), ha='center', va='bottom', fontsize=8)
        ax.set_ylabel("Outlier Count", fontsize=9)
        ax.set_title("IQR‑Based Outlier Counts (First 12 Features)", fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        self.iqr_figure.tight_layout(pad=1.5)
        self.iqr_canvas.draw()
        self.graph_tabs.setCurrentIndex(0)
        self.log_text.setText("IQR analysis completed.")

    def skewness_analysis(self):
        raw_df = self._get_raw_data()
        if raw_df is None: return

        numeric = raw_df.select_dtypes(include=[np.number])
        skew = numeric.skew().reset_index()
        skew.columns = ["Feature", "Skewness"]
        skew = skew.sort_values("Skewness", key=abs, ascending=False)

        suggestions = []
        for _, row in skew.iterrows():
            feat = row["Feature"]
            sk = row["Skewness"]
            col_data = raw_df[feat].dropna()
            col_min = col_data.min()
            if col_min > 0 and abs(sk) > 1:
                suggestions.append("Box-Cox")
            elif col_min >= 0 and abs(sk) > 1:
                suggestions.append("Log1p")
            else:
                suggestions.append("None")
        skew["Suggested Transformation"] = suggestions

        self.table_model.setDataFrame(skew)

        self.skew_figure.clear()
        ax = self.skew_figure.add_subplot(111)
        colors = ['red' if abs(v)>1 else 'orange' if abs(v)>0.5 else 'green' for v in skew["Skewness"]]
        bars = ax.barh(skew["Feature"], skew["Skewness"], color=colors, edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.axvline(1, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='|Skew| = 1')
        ax.axvline(-1, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        for bar, val in zip(bars, skew["Skewness"]):
            x_pos = bar.get_width() + 0.1 if val > 0 else bar.get_width() - 0.3
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=7, color='black')
        ax.set_xlabel("Skewness", fontsize=9)
        ax.set_ylabel("Feature", fontsize=9)
        ax.set_title("Feature Skewness Coefficients and Suggested Transformations", fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.invert_yaxis()
        self.skew_figure.tight_layout(pad=1.5)
        self.skew_canvas.draw()
        self.graph_tabs.setCurrentIndex(1)

        boxcox_feats = skew[skew["Suggested Transformation"] == "Box-Cox"]["Feature"].tolist()
        log1p_feats = skew[skew["Suggested Transformation"] == "Log1p"]["Feature"].tolist()
        self.log_text.clear()
        self.log_text.append("Skewness analysis completed. Suggested transformations (all training columns):")
        self.log_text.append(f"Box‑Cox: {boxcox_feats if boxcox_feats else 'None'}")
        self.log_text.append(f"Log1p   : {log1p_feats if log1p_feats else 'None'}")

# =============================================================================
# TAB 5: MODELING (TABBED, THREADED)
# =============================================================================
class ModelingTab(QWidget):
    def __init__(self, data_manager, parent=None):
        QWidget.__init__(self, parent)
        self.dm = data_manager
        self.abc_worker = None
        self.nested_cv_worker = None
        self.bootstrap_worker = None
        self.comp_worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        self.btn_abc = QPushButton("ABC Optimization (GBM)")
        self.btn_abc.clicked.connect(self.abc_optimize)
        self.btn_abc.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        ctrl.addWidget(self.btn_abc)
        self.btn_compare = QPushButton("Compare Models")
        self.btn_compare.clicked.connect(self.compare_models)
        ctrl.addWidget(self.btn_compare)
        self.btn_train = QPushButton("Train Final Model")
        self.btn_train.clicked.connect(self.train_final)
        self.btn_train.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        ctrl.addWidget(self.btn_train)
        self.btn_save = QPushButton("Save Model")
        self.btn_save.clicked.connect(self.save_model)
        ctrl.addWidget(self.btn_save)

        self.btn_nested_cv = QPushButton("Evaluate with Nested CV")
        self.btn_nested_cv.clicked.connect(self.nested_cv_evaluation)
        ctrl.addWidget(self.btn_nested_cv)
        self.btn_bootstrap = QPushButton("Uncertainty Analysis (Bootstrap)")
        self.btn_bootstrap.clicked.connect(self.bootstrap_uncertainty)
        ctrl.addWidget(self.btn_bootstrap)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        left_layout.addWidget(self.results_text)
        self.comparison_table = QTableView()
        self.comp_model = PandasModel()
        self.comparison_table.setModel(self.comp_model)
        left_layout.addWidget(QLabel("Model Comparison:"))
        left_layout.addWidget(self.comparison_table)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.graph_tabs = QTabWidget()

        sc_tab = QWidget()
        sc_layout = QVBoxLayout(sc_tab)
        self.sc_figure = plt.figure(figsize=(7, 5))
        self.sc_canvas = FigureCanvas(self.sc_figure)
        self.sc_toolbar = NavigationToolbar(self.sc_canvas, self)
        sc_layout.addWidget(self.sc_toolbar)
        sc_layout.addWidget(self.sc_canvas)
        self.graph_tabs.addTab(sc_tab, "Actual vs Predicted")

        cv_tab = QWidget()
        cv_layout = QVBoxLayout(cv_tab)
        self.cv_figure = plt.figure(figsize=(7, 5))
        self.cv_canvas = FigureCanvas(self.cv_figure)
        self.cv_toolbar = NavigationToolbar(self.cv_canvas, self)
        cv_layout.addWidget(self.cv_toolbar)
        cv_layout.addWidget(self.cv_canvas)
        self.graph_tabs.addTab(cv_tab, "CV Boxplot")

        cd_tab = QWidget()
        cd_layout = QVBoxLayout(cd_tab)
        self.cd_figure = plt.figure(figsize=(7, 5))
        self.cd_canvas = FigureCanvas(self.cd_figure)
        self.cd_toolbar = NavigationToolbar(self.cd_canvas, self)
        cd_layout.addWidget(self.cd_toolbar)
        cd_layout.addWidget(self.cd_canvas)
        self.graph_tabs.addTab(cd_tab, "CD Diagram")

        abc_tab = QWidget()
        abc_layout = QVBoxLayout(abc_tab)
        self.abc_figure = plt.figure(figsize=(7, 5))
        self.abc_canvas = FigureCanvas(self.abc_figure)
        self.abc_toolbar = NavigationToolbar(self.abc_canvas, self)
        abc_layout.addWidget(self.abc_toolbar)
        abc_layout.addWidget(self.abc_canvas)
        self.graph_tabs.addTab(abc_tab, "ABC Convergence")

        boot_tab = QWidget()
        boot_layout = QVBoxLayout(boot_tab)
        self.boot_figure = plt.figure(figsize=(7, 5))
        self.boot_canvas = FigureCanvas(self.boot_figure)
        self.boot_toolbar = NavigationToolbar(self.boot_canvas, self)
        boot_layout.addWidget(self.boot_toolbar)
        boot_layout.addWidget(self.boot_canvas)
        self.graph_tabs.addTab(boot_tab, "Bootstrap")

        feat_tab = QWidget()
        feat_layout = QVBoxLayout(feat_tab)
        self.feat_figure = plt.figure(figsize=(7, 5))
        self.feat_canvas = FigureCanvas(self.feat_figure)
        self.feat_toolbar = NavigationToolbar(self.feat_canvas, self)
        feat_layout.addWidget(self.feat_toolbar)
        feat_layout.addWidget(self.feat_canvas)
        self.graph_tabs.addTab(feat_tab, "Feature Importance")

        right_layout.addWidget(self.graph_tabs)
        splitter.addWidget(right)

        layout.addWidget(splitter)

    def _get_train_test_transformed(self):
        if self.dm.X_train_trans is None or self.dm.y_train is None:
            QMessageBox.warning(self, "Warning", "Please apply Transformation in Tab 4 first!")
            return None, None, None, None
        if self.dm.selected_features is not None:
            cols = [c for c in self.dm.selected_features if c in self.dm.X_train_trans.columns]
            X_train = self.dm.X_train_trans[cols]
            X_test = self.dm.X_test_trans[cols]
        else:
            X_train = self.dm.X_train_trans
            X_test = self.dm.X_test_trans
        return X_train, X_test, self.dm.y_train, self.dm.y_test

    def compare_models(self):
        X_train, X_test, y_train, y_test = self._get_train_test_transformed()
        if X_train is None: return

        if self.dm.abc_best_params is None:
            self.results_text.append("ABC optimization not yet performed; auto‑starting...")
            self.abc_optimize(show_result=False)

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.btn_compare.setEnabled(False)

        self.results_text.append("\n=== MODEL COMPARISON STARTED ===")

        cv = self.dm.get_cv_splitter(X_train, y_train)

        self.comp_worker = ModelComparisonWorker(
            X_train, X_test, y_train, y_test,
            self.dm.abc_best_params, cv
        )
        self.comp_worker.progress.connect(self.on_comp_progress)
        self.comp_worker.finished.connect(self.on_comp_finished)
        self.comp_worker.start()

    def on_comp_progress(self, idx, msg):
        self.results_text.append(msg)

    def on_comp_finished(self, result_package):
        results = result_package['results']
        predictions = result_package['predictions']
        cv_scores_dict = result_package['cv_scores_dict']
        y_test = result_package['y_test']

        res_df = pd.DataFrame(results)
        self.comp_model.setDataFrame(res_df)
        self.results_text.append("\n=== MODEL COMPARISON (Train/Test Metrics) ===")
        self.results_text.append(res_df.to_string(index=False))

        self.sc_figure.clear()
        ax = self.sc_figure.add_subplot(111)
        colors_sc = {
            'Random Forest': 'green', 'Decision Tree': 'blue', 'GBM + ABC': 'darkorange',
            'SVR (RBF)': 'brown', 'XGBoost': 'cyan', 'LightGBM': 'magenta'
        }
        for name, pred in predictions.items():
            ax.scatter(y_test, pred, alpha=0.5, label=name, color=colors_sc.get(name, 'gray'), s=30)
        mn, mx = y_test.min(), y_test.max()
        ax.plot([mn, mx], [mn, mx], 'k--', lw=2, label='Ideal (y=x)')
        ax.set_xlabel("Actual Fatigue Strength (MPa)", fontsize=9)
        ax.set_ylabel("Predicted Fatigue Strength (MPa)", fontsize=9)
        ax.set_title("Actual vs Predicted - Model Comparison", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self.sc_figure.tight_layout(pad=1.5)
        self.sc_canvas.draw()
        self.graph_tabs.setCurrentIndex(0)

        scores_list = list(cv_scores_dict.values())
        model_names = list(cv_scores_dict.keys())
        if len(scores_list) >= 2 and all(len(s) == len(scores_list[0]) for s in scores_list):
            try:
                stat, p_value = friedmanchisquare(*scores_list)
                self.results_text.append("\n=== STATISTICAL SIGNIFICANCE TEST (Friedman) ===")
                self.results_text.append(f"Test Statistic (χ²): {stat:.4f}")
                self.results_text.append(f"p‑value: {p_value:.6f}")
                if p_value < 0.05:
                    self.results_text.append("Result: There is a STATISTICALLY SIGNIFICANT difference among models (p < 0.05).")
                    try:
                        nemenyi_result = sp.posthoc_nemenyi_friedman(
                            np.array([scores_list[i] for i in range(len(model_names))]).T
                        )
                        self.results_text.append("\n--- NEMENYI POST‑HOC TEST (p‑values) ---")
                        self.results_text.append(nemenyi_result.to_string())

                        n_models = len(model_names)
                        n_splits = len(scores_list[0])
                        ranks = np.zeros((n_models, n_splits))
                        for split in range(n_splits):
                            mae_values = [scores_list[i][split] for i in range(n_models)]
                            rank_order = np.argsort(mae_values) + 1
                            for i in range(n_models):
                                ranks[i, split] = rank_order[i]
                        avg_ranks = np.mean(ranks, axis=1)

                        alpha = 0.05
                        q = studentized_range.ppf(1 - alpha, n_models, np.inf)
                        cd = q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_splits))
                        self.results_text.append(f"\nCritical Difference (CD): {cd:.4f}")

                        self.cd_figure.clear()
                        draw_cd_diagram(avg_ranks, model_names, cd, figure=self.cd_figure)
                        self.cd_canvas.draw()
                        self.graph_tabs.setCurrentIndex(2)
                    except Exception as e:
                        self.results_text.append(f"Nemenyi test/CD diagram error: {str(e)}")
                else:
                    self.results_text.append("Result: No statistically significant difference among models (p ≥ 0.05).")
            except Exception as e:
                self.results_text.append(f"Friedman test could not be performed: {str(e)}")
        else:
            self.results_text.append("Insufficient number of models or unequal fold counts for Friedman test.")

        self.cv_figure.clear()
        ax = self.cv_figure.add_subplot(111)
        box_data = [scores_list[i] for i in range(len(model_names))]
        bp = ax.boxplot(box_data, tick_labels=model_names, patch_artist=True, showmeans=True)
        colors_box = ['lightblue', 'lightgreen', 'orange', 'pink', 'lightyellow', 'cyan']
        for patch, color in zip(bp['boxes'], colors_box[:len(model_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("MAE (MPa)", fontsize=9)
        ax.set_title("Comparison of Model Performance (CV Folds)", fontsize=11, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        self.cv_figure.tight_layout(pad=1.5)
        self.cv_canvas.draw()

        self.progress.setVisible(False)
        self.btn_compare.setEnabled(True)

    def nested_cv_evaluation(self):
        X_train, X_test, y_train, y_test = self._get_train_test_transformed()
        if X_train is None: return

        self.results_text.append("\n=== NESTED CROSS‑VALIDATION STARTED ===")
        self.progress.setVisible(True)
        self.btn_nested_cv.setEnabled(False)

        self.nested_cv_worker = NestedCVWorker(X_train, y_train)
        self.nested_cv_worker.progress.connect(self.on_nested_cv_progress)
        self.nested_cv_worker.finished.connect(self.on_nested_cv_finished)
        self.nested_cv_worker.start()

    def on_nested_cv_progress(self, msg):
        self.results_text.append(msg)

    def on_nested_cv_finished(self, result):
        mean_mae = result['mean_mae']
        std_mae = result['std_mae']
        mean_r2 = result['mean_r2']
        std_r2 = result['std_r2']
        best_params_list = result['best_params_list']

        self.results_text.append("\n--- NESTED CV RESULTS ---")
        self.results_text.append(f"Mean MAE: {mean_mae:.2f} ± {std_mae:.2f} MPa")
        self.results_text.append(f"Mean R² : {mean_r2:.4f} ± {std_r2:.4f}")

        df_params = pd.DataFrame(best_params_list)
        param_summary = {}
        for col in df_params.columns:
            if pd.api.types.is_numeric_dtype(df_params[col]):
                param_summary[col] = round(df_params[col].mean(), 2)
            else:
                mode_val = df_params[col].mode()
                param_summary[col] = mode_val[0] if not mode_val.empty else df_params[col].iloc[0]
        self.results_text.append(f"Most frequent hyperparameters (mean/mode): {param_summary}")

        self.progress.setVisible(False)
        self.btn_nested_cv.setEnabled(True)

    def bootstrap_uncertainty(self):
        if self.dm.best_model is None:
            QMessageBox.warning(self, "Warning", "Please train the final model first!")
            return

        X_train, X_test, y_train, y_test = self._get_train_test_transformed()
        if X_train is None: return

        n_bootstrap, ok = QInputDialog.getInt(self, "Bootstrap Settings", "Number of bootstrap samples (e.g., 100):",
                                              value=100, min=10, max=500, step=10)
        if not ok: return

        confidence, ok2 = QInputDialog.getDouble(self, "Confidence Level", "Confidence level (0‑1, e.g., 0.95):",
                                                value=0.95, min=0.5, max=0.99, decimals=2)
        if not ok2: return

        self.results_text.append(f"\n--- BOOTSTRAP UNCERTAINTY ANALYSIS STARTED (n={n_bootstrap}) ---")
        self.progress.setVisible(True)
        self.btn_bootstrap.setEnabled(False)

        self.bootstrap_worker = BootstrapWorker(
            X_train, y_train, X_test, y_test,
            self.dm.abc_best_params, n_bootstrap, confidence
        )
        self.bootstrap_worker.progress.connect(self.on_bootstrap_progress)
        self.bootstrap_worker.finished.connect(self.on_bootstrap_finished)
        self.bootstrap_worker.start()

    def on_bootstrap_progress(self, iteration, msg):
        self.progress.setValue(iteration)
        self.results_text.append(msg)

    def on_bootstrap_finished(self, result):
        results_df = result['results_df']
        coverage = result['coverage']
        mean_width = result['mean_width']
        lower = result['lower']
        upper = result['upper']
        mean_pred = result['mean_pred']

        self.results_text.append("\n--- BOOTSTRAP RESULTS (first 10 test samples) ---")
        self.results_text.append(results_df.head(10).to_string(index=False))
        self.results_text.append(f"\n--- SUMMARY ---")
        self.results_text.append(f"Mean Interval Width: {mean_width:.2f} MPa")
        self.results_text.append(f"Coverage Ratio of Actual Values: {coverage:.1f}%")

        self.boot_figure.clear()
        ax = self.boot_figure.add_subplot(111)
        n_plot = min(30, len(self.dm.y_test))
        indices = np.arange(n_plot)
        ax.fill_between(indices, lower[:n_plot], upper[:n_plot], alpha=0.3, color='blue', label=f'{int(self.bootstrap_worker.confidence*100)}% CI')
        ax.plot(indices, mean_pred[:n_plot], 'b-', label='Mean Prediction')
        ax.scatter(indices, self.dm.y_test.values[:n_plot], c='red', s=40, marker='o', label='Actual Value')
        ax.set_xlabel("Test Sample Index", fontsize=9)
        ax.set_ylabel("Fatigue Strength (MPa)", fontsize=9)
        ax.set_title("Bootstrap Prediction Interval", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self.boot_figure.tight_layout(pad=1.5)
        self.boot_canvas.draw()
        self.graph_tabs.setCurrentIndex(4)

        self.progress.setVisible(False)
        self.btn_bootstrap.setEnabled(True)
        QMessageBox.information(self, "Completed", f"Bootstrap analysis completed.\nMean interval width: {mean_width:.2f} MPa")

    def abc_optimize(self, show_result=True):
        X_train, _, y_train, _ = self._get_train_test_transformed()
        if X_train is None: return
        self.progress.setVisible(True)
        self.results_text.append("\nABC optimization started...")
        param_space = {
            'n_estimators': ('int', 50, 400),
            'max_depth': ('int', 3, 15),
            'learning_rate': ('continuous', 0.001, 0.5),
            'subsample': ('continuous', 0.5, 1.0),
            'min_samples_split': ('int', 2, 20),
            'min_samples_leaf': ('int', 1, 20),
            'max_features': ('categorical', ['sqrt', 'log2', None])
        }
        self.abc_worker = ABCWorker(X_train, y_train, param_space, pop_size=10, max_cycles=30)
        self.abc_worker.progress.connect(self.on_abc_progress)
        self.abc_worker.finished.connect(self.on_abc_finished)
        self.abc_worker.start()

    def on_abc_progress(self, msg):
        self.results_text.append(msg)

    def on_abc_finished(self, best_params, best_score, history):
        self.dm.abc_best_params = best_params
        self.results_text.append("\nABC Best Parameters: " + json.dumps(best_params, indent=2))
        self.results_text.append(f"ABC Best CV MAE: {-best_score:.4f} MPa")

        model = GradientBoostingRegressor(**best_params, random_state=42)
        X_train, X_test, y_train, y_test = self._get_train_test_transformed()
        if X_train is not None and X_test is not None:
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_r2 = r2_score(y_test, y_pred_test)

            self.results_text.append("\n--- ABC‑Optimized GBM – Test Performance ---")
            self.results_text.append(f"Test MAE : {test_mae:.3f} MPa")
            self.results_text.append(f"Test RMSE: {test_rmse:.3f} MPa")
            self.results_text.append(f"Test R²  : {test_r2:.4f}")

        self.progress.setVisible(False)

        self.abc_figure.clear()
        ax = self.abc_figure.add_subplot(111)
        if history:
            ax.plot(range(1, len(history)+1), [-h for h in history], 'o-', color='darkorange', markersize=4)
            ax.set_xlabel("Cycle", fontsize=9)
            ax.set_ylabel("Best MAE (MPa)", fontsize=9)
            ax.set_title("ABC Optimization Convergence Graph", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        self.abc_figure.tight_layout(pad=1.5)
        self.abc_canvas.draw()
        self.graph_tabs.setCurrentIndex(3)

    def train_final(self):
        X_train, X_test, y_train, y_test = self._get_train_test_transformed()
        if X_train is None: return
        if self.dm.abc_best_params is None:
            QMessageBox.warning(self, "Warning", "Please perform ABC optimization first!")
            return

        model = GradientBoostingRegressor(**self.dm.abc_best_params, random_state=42)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)

        self.dm.best_model = model
        self.results_text.append("\n=== FINAL MODEL (GBM) ===")
        self.results_text.append("Parameters: " + json.dumps(self.dm.abc_best_params, indent=2))
        self.results_text.append(f"Train  -> MAE: {train_mae:.3f}, R²: {train_r2:.4f}")
        self.results_text.append(f"Test   -> MAE: {test_mae:.3f}, R²: {test_r2:.4f}")

        importance = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        self.results_text.append("\nFeature Importance:")
        self.results_text.append(importance.to_string(index=False))

        self.dm.shap_explainer = shap.TreeExplainer(model)
        self.results_text.append("SHAP explainer ready. You can use the Interpretability tab.")

        self.feat_figure.clear()
        ax = self.feat_figure.add_subplot(111)
        bars = ax.barh(importance["Feature"], importance["Importance"], color='mediumseagreen', edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, importance["Importance"]):
            ax.text(bar.get_width() + 0.01 * importance["Importance"].max(),
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=7, color='black')
        ax.set_title("GBM Feature Importance (ABC‑Optimized)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Importance", fontsize=9)
        ax.set_ylabel("Feature", fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        self.feat_figure.tight_layout(pad=1.5)
        self.feat_canvas.draw()
        self.graph_tabs.setCurrentIndex(5)

        QMessageBox.information(self, "Success", "Final model trained.")

    def save_model(self):
        if self.dm.best_model is None:
            QMessageBox.warning(self, "Warning", "Please train the model first!")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", "fatigue_model.pkl", "Pickle (*.pkl)")
        if path:
            if hasattr(self.dm.best_model, 'feature_names_in_'):
                used_features = list(self.dm.best_model.feature_names_in_)
            else:
                used_features = self.dm.selected_features or list(self.dm.X_train_trans.columns)

            pipeline_data = {
                'boxcox_lambdas': self.dm.transform_pipeline.boxcox_lambdas if self.dm.transform_pipeline else {},
                'log1p_features': self.dm.transform_pipeline.log1p_features if self.dm.transform_pipeline else [],
                'feature_order': used_features,
                'fit_features': self.dm.transform_pipeline.fit_features if self.dm.transform_pipeline else [],
                'power_transformer': self.dm.transform_pipeline.power_trans if self.dm.transform_pipeline else None
            }
            model_data = {
                'model': self.dm.best_model,
                'features': used_features,
                'params': self.dm.abc_best_params,
                'pipeline': pipeline_data,
                'shap_explainer': self.dm.shap_explainer,
                'date': datetime.now().isoformat()
            }
            joblib.dump(model_data, path)
            QMessageBox.information(self, "Success", f"Model saved to:\n{path}")

# =============================================================================
# TAB 6: INTERPRETABILITY
# =============================================================================
class InterpretTab(QWidget):
    def __init__(self, data_manager, parent=None):
        QWidget.__init__(self, parent)
        self.dm = data_manager
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        self.btn_shap_summary = QPushButton("SHAP Summary Plot")
        self.btn_shap_summary.clicked.connect(self.plot_shap_summary)
        ctrl.addWidget(self.btn_shap_summary)
        self.btn_dep = QPushButton("SHAP Dependence Plot")
        self.btn_dep.clicked.connect(self.plot_shap_dependence)
        ctrl.addWidget(self.btn_dep)
        self.btn_pdp = QPushButton("Partial Dependence (PDP)")
        self.btn_pdp.clicked.connect(self.plot_pdp)
        ctrl.addWidget(self.btn_pdp)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.graph_tabs = QTabWidget()

        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        self.summary_figure = plt.figure(figsize=(10, 7))
        self.summary_canvas = FigureCanvas(self.summary_figure)
        self.summary_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.summary_toolbar = NavigationToolbar(self.summary_canvas, self)
        summary_layout.addWidget(self.summary_toolbar)
        self.summary_scroll = QScrollArea()
        self.summary_scroll.setWidgetResizable(False)
        self.summary_scroll.setWidget(self.summary_canvas)
        summary_layout.addWidget(self.summary_scroll)
        self.graph_tabs.addTab(summary_tab, "SHAP Summary Plot")

        dep_tab = QWidget()
        dep_layout = QVBoxLayout(dep_tab)
        self.dep_figure = plt.figure(figsize=(10, 7))
        self.dep_canvas = FigureCanvas(self.dep_figure)
        self.dep_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dep_toolbar = NavigationToolbar(self.dep_canvas, self)
        dep_layout.addWidget(self.dep_toolbar)
        self.dep_scroll = QScrollArea()
        self.dep_scroll.setWidgetResizable(False)
        self.dep_scroll.setWidget(self.dep_canvas)
        dep_layout.addWidget(self.dep_scroll)
        self.graph_tabs.addTab(dep_tab, "SHAP Dependence Plot")

        pdp_tab = QWidget()
        pdp_layout = QVBoxLayout(pdp_tab)
        self.pdp_figure = plt.figure(figsize=(10, 7))
        self.pdp_canvas = FigureCanvas(self.pdp_figure)
        self.pdp_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pdp_toolbar = NavigationToolbar(self.pdp_canvas, self)
        pdp_layout.addWidget(self.pdp_toolbar)
        self.pdp_scroll = QScrollArea()
        self.pdp_scroll.setWidgetResizable(False)
        self.pdp_scroll.setWidget(self.pdp_canvas)
        pdp_layout.addWidget(self.pdp_scroll)
        self.graph_tabs.addTab(pdp_tab, "Partial Dependence (PDP)")

        layout.addWidget(self.graph_tabs)

    def _get_X(self):
        if self.dm.X_train_trans is None:
            return None
        X = self.dm.X_train_trans
        if self.dm.best_model is not None and hasattr(self.dm.best_model, 'feature_names_in_'):
            cols = [c for c in self.dm.best_model.feature_names_in_ if c in X.columns]
            return X[cols]
        elif self.dm.selected_features is not None:
            cols = [c for c in self.dm.selected_features if c in X.columns]
            return X[cols]
        return X

    def _check_requirements(self):
        if self.dm.best_model is None:
            QMessageBox.warning(self, "Warning", "Please train the final model first!")
            return False
        if self.dm.X_train_trans is None:
            QMessageBox.warning(self, "Warning", "Training data not found!")
            return False
        if self.dm.shap_explainer is None:
            try:
                self.dm.shap_explainer = shap.TreeExplainer(self.dm.best_model)
            except:
                QMessageBox.critical(self, "Error", "SHAP explainer could not be created.")
                return False
        return True

    def plot_shap_summary(self):
        if not self._check_requirements(): return
        X = self._get_X()
        if X is None: return
        shap_values = self.dm.shap_explainer.shap_values(X)

        n_features = len(X.columns)
        label_size = 9
        height_per_feature = 0.40

        fig_height = max(5, n_features * height_per_feature)
        fig_width = 10

        self.summary_figure.set_size_inches(fig_width, fig_height)
        self.summary_figure.clear()
        ax = self.summary_figure.add_subplot(111)

        original_fontsize = plt.rcParams['font.size']
        plt.rcParams['font.size'] = label_size

        try:
            plt.figure(self.summary_figure.number)
            try:
                shap.summary_plot(shap_values, X, show=False, ax=ax,
                                  alpha=1.0, cmap='coolwarm')
            except TypeError:
                shap.summary_plot(shap_values, X, show=False)
        finally:
            plt.rcParams['font.size'] = original_fontsize

        ax.set_title(f'SHAP Summary Plot ({n_features} features)',
                     fontsize=label_size + 1, fontweight='bold')
        ax.tick_params(labelsize=label_size)

        self.summary_figure.tight_layout(pad=1.5)

        canvas_width  = int(fig_width * self.summary_figure.dpi)
        canvas_height = int(fig_height * self.summary_figure.dpi)
        self.summary_canvas.setMinimumSize(canvas_width, canvas_height)
        self.summary_canvas.resize(canvas_width, canvas_height)
        self.summary_canvas.draw()

        self.graph_tabs.setCurrentIndex(0)

    def plot_shap_dependence(self):
        if not self._check_requirements(): return
        X = self._get_X()
        if X is None: return
        shap_values = self.dm.shap_explainer.shap_values(X)

        features = list(X.columns)
        n = len(features)
        cols = 2
        rows = (n + cols - 1) // cols

        fig_width = max(12, cols * 4.5)
        fig_height = max(6, rows * 3.5)
        self.dep_figure.set_size_inches(fig_width, fig_height)
        self.dep_figure.clear()

        for i, feat in enumerate(features):
            ax = self.dep_figure.add_subplot(rows, cols, i + 1)
            shap.dependence_plot(feat, shap_values, X, show=False, ax=ax)
            ax.set_title(f'{feat}', fontsize=9, fontweight='bold')
            ax.tick_params(labelsize=7)
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel('SHAP value', fontsize=8)

        self.dep_figure.tight_layout(pad=1.5)
        self.dep_canvas.setMinimumSize(int(fig_width * self.dep_figure.dpi),
                                       int(fig_height * self.dep_figure.dpi))
        self.dep_canvas.resize(int(fig_width * self.dep_figure.dpi),
                               int(fig_height * self.dep_figure.dpi))
        self.dep_canvas.draw()
        self.graph_tabs.setCurrentIndex(1)

    def plot_pdp(self):
        if self.dm.best_model is None:
            QMessageBox.warning(self, "Warning", "Please train the model first!")
            return
        X = self._get_X()
        if X is None: return

        X = X.astype(np.float64)
        features = list(X.columns)
        n = len(features)
        if n == 0:
            return

        cols = 3
        rows = (n + cols - 1) // cols

        fig_width = max(12, cols * 4.5)
        fig_height = max(6, rows * 3.5)
        self.pdp_figure.set_size_inches(fig_width, fig_height)

        self.pdp_figure.clear()

        try:
            axes = self.pdp_figure.subplots(rows, cols, squeeze=False)

            for i, feat in enumerate(features):
                row = i // cols
                col = i % cols
                ax = axes[row, col]
                PartialDependenceDisplay.from_estimator(
                    self.dm.best_model, X, [feat], ax=ax
                )
                ax.set_title(f'{feat}', fontsize=9, fontweight='bold')
                ax.tick_params(labelsize=7)
                ax.set_xlabel(feat, fontsize=8)
                ax.set_ylabel('Partial dependence', fontsize=8)

            for j in range(i + 1, rows * cols):
                row = j // cols
                col = j % cols
                axes[row, col].set_visible(False)

            self.pdp_figure.tight_layout(pad=1.5)
            self.pdp_canvas.setMinimumSize(int(fig_width * self.pdp_figure.dpi),
                                           int(fig_height * self.pdp_figure.dpi))
            self.pdp_canvas.resize(int(fig_width * self.pdp_figure.dpi),
                                   int(fig_height * self.pdp_figure.dpi))
            self.pdp_canvas.draw()
            self.graph_tabs.setCurrentIndex(2)
        except Exception as e:
            QMessageBox.warning(self, "PDP Error", str(e))

# =============================================================================
# TAB 7: PREDICTION
# =============================================================================
class PredictionTab(QWidget):
    def __init__(self, data_manager, parent=None):
        QWidget.__init__(self, parent)
        self.dm = data_manager
        self.selected_actual = None
        self.loaded_pipeline = None
        self.loaded_explainer = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        ctrl = QHBoxLayout()
        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.clicked.connect(self.load_model)
        ctrl.addWidget(self.btn_load_model)
        self.btn_update_fields = QPushButton("Update Input Fields")
        self.btn_update_fields.clicked.connect(self.update_input_fields)
        ctrl.addWidget(self.btn_update_fields)
        self.btn_predict = QPushButton("Predict")
        self.btn_predict.clicked.connect(self.predict)
        self.btn_predict.setStyleSheet("background-color: #673AB7; color: white; font-weight: bold; padding: 8px;")
        ctrl.addWidget(self.btn_predict)
        self.btn_batch = QPushButton("Batch Predict (CSV)")
        self.btn_batch.clicked.connect(self.batch_predict)
        ctrl.addWidget(self.btn_batch)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.lbl_model = QLabel("Model: Not Loaded")
        self.lbl_model.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.lbl_model)

        test_group = QGroupBox("Test Data")
        test_layout = QVBoxLayout(test_group)
        self.test_table = QTableView()
        self.test_model = PandasModel()
        self.test_table.setModel(self.test_model)
        self.test_table.selectionModel().selectionChanged.connect(self.on_test_row_selected)
        test_layout.addWidget(self.test_table)
        layout.addWidget(test_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.input_widget = QWidget()
        self.input_grid = QGridLayout(self.input_widget)
        scroll.setWidget(self.input_widget)
        layout.addWidget(scroll)

        self.input_fields = {}
        self._rebuild_fields([])

        self.result_label = QLabel("Prediction Result: -")
        self.result_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2E7D32;")
        align_flag = Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter
        self.result_label.setAlignment(align_flag)
        layout.addWidget(self.result_label)

        self.actual_label = QLabel("")
        self.actual_label.setStyleSheet("font-size: 12pt; color: #555;")
        self.actual_label.setAlignment(align_flag)
        layout.addWidget(self.actual_label)

        self.batch_table = QTableView()
        self.batch_model = PandasModel()
        self.batch_table.setModel(self.batch_model)
        layout.addWidget(QLabel("Batch Prediction Results:"))
        layout.addWidget(self.batch_table)

    def update_input_fields(self):
        if self.dm.selected_features is None:
            QMessageBox.warning(self, "Warning", "Please perform feature selection in Tab 3 or load a model first.")
            return
        features = [f for f in self.dm.selected_features if f not in [self.dm.target_col, self.dm.id_col]]
        if not features:
            QMessageBox.warning(self, "Warning", "Selected feature set is empty!")
            return
        self._rebuild_fields(features)
        QMessageBox.information(self, "Updated", f"{len(features)} input fields created.")

    def _rebuild_fields(self, features):
        while self.input_grid.count():
            item = self.input_grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.input_fields.clear()
        for i, feat in enumerate(features):
            lbl = QLabel(f"{feat}:")
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(-99999, 99999)
            spin.setValue(0.0)
            spin.setSingleStep(0.1)
            spin.setMinimumWidth(120)
            self.input_fields[feat] = spin
            row = i // 3
            col = (i % 3) * 2
            self.input_grid.addWidget(lbl, row, col)
            self.input_grid.addWidget(spin, row, col+1)

    def load_test_data(self, X_test, y_test):
        test_df = X_test.copy()
        test_df[self.dm.target_col] = y_test.values
        self.test_model.setDataFrame(test_df)
        self.selected_actual = None
        self.actual_label.setText("")
        self.result_label.setText("Prediction Result: -")

    def on_test_row_selected(self, selected, deselected):
        indexes = self.test_table.selectionModel().selectedRows()
        if not indexes: return
        row = indexes[0].row()
        test_df = self.test_model._df

        row_df = test_df.iloc[[row]]

        if hasattr(self.dm.best_model, 'feature_names_in_'):
            model_features = list(self.dm.best_model.feature_names_in_)
        elif self.dm.selected_features is not None:
            model_features = self.dm.selected_features
        else:
            model_features = list(self.input_fields.keys())

        try:
            row_df = ensure_derived_features(row_df, model_features)
        except Exception:
            pass

        for col in row_df.columns:
            if col == self.dm.target_col:
                self.selected_actual = row_df[col].iloc[0]
                self.actual_label.setText(f"Actual Value: {self.selected_actual:.2f} MPa")
            elif col in self.input_fields:
                self.input_fields[col].setValue(float(row_df[col].iloc[0]))

        self.result_label.setText("Prediction Result: -")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Pickle (*.pkl)")
        if path:
            try:
                data = joblib.load(path)
                self.dm.best_model = data['model']
                features = data.get('features', [])
                self.dm.selected_features = features
                self.lbl_model.setText(f"Model: {os.path.basename(path)} | Features: {len(features)}")
                self.lbl_model.setStyleSheet("color: green; font-weight: bold;")
                pipeline_data = data.get('pipeline', {})
                pipeline = TransformPipeline()
                pipeline.power_trans = pipeline_data.get('power_transformer', None)
                pipeline.log1p_features = pipeline_data.get('log1p_features', [])
                pipeline.feature_order = pipeline_data.get('feature_order', [])
                pipeline.fit_features = pipeline_data.get('fit_features', [])
                self.loaded_pipeline = pipeline
                self.loaded_explainer = data.get('shap_explainer', None)
                self.update_input_fields()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def predict(self):
        if self.dm.best_model is None:
            QMessageBox.warning(self, "Warning", "Please load or train a model first!")
            return
        try:
            if hasattr(self.dm.best_model, 'feature_names_in_'):
                model_features = list(self.dm.best_model.feature_names_in_)
            elif self.dm.selected_features is not None:
                model_features = self.dm.selected_features
            else:
                model_features = list(self.input_fields.keys())

            # If a row is selected in the test table
            if self.selected_actual is not None and self.test_model._df is not None:
                row_idx = self.test_table.selectionModel().selectedRows()[0].row()
                row_data = self.test_model._df.iloc[row_idx]
                row_df = pd.DataFrame([row_data])
                if self.loaded_pipeline is not None and self.loaded_pipeline.fit_features:
                    transform_features = self.loaded_pipeline.fit_features
                    row_df = ensure_derived_features(row_df, transform_features)
                    X_raw = row_df[transform_features]
                else:
                    X_raw = row_df[model_features]
            else:
                # Manual entry
                missing = [f for f in model_features if f not in self.input_fields]
                if missing:
                    QMessageBox.critical(self, "Error", f"The following features are missing from input fields:\n{', '.join(missing)}")
                    return
                values = [self.input_fields[f].value() for f in model_features]
                X_raw = pd.DataFrame([values], columns=model_features)

                if self.loaded_pipeline is not None and self.loaded_pipeline.fit_features:
                    transform_features = self.loaded_pipeline.fit_features
                    for col in transform_features:
                        if col not in X_raw.columns:
                            X_raw[col] = 1.0

            # Transformation and prediction
            if self.loaded_pipeline is not None:
                X_trans = self.loaded_pipeline.transform(X_raw)
                X_model = X_trans[model_features]
            else:
                X_model = X_raw[model_features]

            pred = self.dm.best_model.predict(X_model)[0]
            self.result_label.setText(f"Predicted Fatigue Strength: {pred:.2f} MPa")
            if self.selected_actual is not None:
                self.actual_label.setText(f"Actual Value: {self.selected_actual:.2f} MPa")
            else:
                self.actual_label.setText("")

        except ValueError as e:
            QMessageBox.warning(self, "Outside Applicability Domain",
                                f"The selected data is outside the model's applicability domain.\n"
                                f"Error: {str(e)}\n\n"
                                "Please test only steels that fall within the applicability domain.")
        except Exception as e:
            QMessageBox.critical(self, "Error", traceback.format_exc())

    def batch_predict(self):
        if self.dm.best_model is None:
            QMessageBox.warning(self, "Warning", "Please load a model first!")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV (*.csv)")
        if path:
            try:
                self.batch_model.setDataFrame(pd.DataFrame())

                df = read_csv_robust(path)
                if hasattr(self.dm.best_model, 'feature_names_in_'):
                    model_features = list(self.dm.best_model.feature_names_in_)
                elif self.dm.selected_features is not None:
                    model_features = self.dm.selected_features
                else:
                    model_features = list(self.input_fields.keys())

                if self.loaded_pipeline is not None and self.loaded_pipeline.fit_features:
                    transform_features = self.loaded_pipeline.fit_features
                else:
                    transform_features = model_features

                df = ensure_derived_features(df, transform_features)

                for col in transform_features:
                    if col not in df.columns:
                        df[col] = 1.0

                X_full = df[transform_features]

                if self.loaded_pipeline is not None:
                    X_trans = self.loaded_pipeline.transform(X_full)
                else:
                    X_trans = X_full

                X_model = X_trans[model_features]

                preds = self.dm.best_model.predict(X_model)
                df['Predicted_Fatigue'] = preds
                self.batch_model.setDataFrame(df)

                save_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "predictions.csv", "CSV (*.csv)")
                if save_path:
                    df.to_csv(save_path, index=False)
                    QMessageBox.information(self, "Success", f"Predictions saved to:\n{save_path}")
            except ValueError as e:
                QMessageBox.warning(self, "Outside Applicability Domain",
                                    f"The uploaded dataset contains values outside the applicability domain.\n"
                                    f"Error: {str(e)}\n\n"
                                    "Please test only steels that fall within the applicability domain.")
            except Exception as e:
                QMessageBox.critical(self, "Error", traceback.format_exc())

# =============================================================================
# MAIN WINDOW
# =============================================================================
class MainWindow(QMainWindow):
    data_loaded = pyqtSignal(object)

    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("Steel Fatigue Strength Prediction – PyQt Desktop Application")
        self.setGeometry(50, 50, 1400, 900)
        self.dm = DataManager()
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        header = QLabel("STEEL FATIGUE STRENGTH PREDICTION | Data Science & Metallurgy Integration")
        header.setStyleSheet("font-size: 16pt; font-weight: bold; color: #1A237E; padding: 10px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter)
        layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tab_data = DataTab(self.dm)
        self.tab_eda = EDATab(self.dm)
        self.tab_feat = FeatureSelectionTab(self.dm)
        self.tab_out = OutlierTab(self.dm)
        self.tab_model = ModelingTab(self.dm)
        self.tab_interpret = InterpretTab(self.dm)
        self.tab_pred = PredictionTab(self.dm)

        self.tabs.addTab(self.tab_data, "1. Data Loading")
        self.tabs.addTab(self.tab_eda, "2. Exploratory Analysis")
        self.tabs.addTab(self.tab_feat, "3. Feature Selection")
        self.tabs.addTab(self.tab_out, "4. Outlier & Transformation")
        self.tabs.addTab(self.tab_model, "5. Modeling")
        self.tabs.addTab(self.tab_interpret, "6. Interpretability")
        self.tabs.addTab(self.tab_pred, "7. Prediction")
        layout.addWidget(self.tabs)

        footer = QLabel("2026 Materials Metallurgy & Data Science | Python + PyQt + Scikit‑learn + ABC Optimization")
        footer.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter)
        layout.addWidget(footer)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QTabWidget::pane { border: 1px solid #ccc; background: white; }
            QTabBar::tab { background: #e0e0e0; padding: 10px 20px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #1976D2; color: white; font-weight: bold; }
            QTabBar::tab:hover { background: #BBDEFB; }
            QPushButton { padding: 6px 12px; border-radius: 4px; border: 1px solid #ccc; background: #fafafa; }
            QPushButton:hover { background: #e3f2fd; border-color: #1976D2; }
            QGroupBox { font-weight: bold; border: 1px solid #bbb; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QTextEdit { border: 1px solid #ddd; background: #fafafa; }
            QTableView { gridline-color: #ddd; selection-background-color: #BBDEFB; }
            QHeaderView::section { background: #1976D2; color: white; padding: 5px; font-weight: bold; }
        """)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Exit', 'Are you sure you want to exit?',
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont("Segoe UI", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()