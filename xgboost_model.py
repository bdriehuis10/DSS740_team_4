"""
=============================================================================
DSS740 — Group 4 Team Project
File        : xgboost_model.py
Author      : Shoheb Sarwar
Model       : XGBoost Classifier
Description : Wage-Adjusted Food Inflation Classification Model.
              Predicts whether a food item's regional food-price inflation
              exceeds the global real-wage-growth benchmark (4.5%),
              signalling households face disproportionate cost pressure.

Target variable : Wage_Adjusted_High_Cost
    1 = regional food inflation > wage benchmark  (high cost pressure)
    0 = regional food inflation <= wage benchmark (manageable cost)

Wage benchmark  : 4.5% — approximate average real-wage growth across
                  developed and emerging markets (World Bank / ILO, 2025).
                  Used as a proxy where direct wage data is unavailable.

Split strategy  : City-based split — cities held out entirely from the
                  test set to prevent region-level data leakage.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import shap

# ── Constants ──────────────────────────────────────────────────────────
WAGE_BENCHMARK_PCT = 4.5    # World Bank / ILO 2025 real-wage proxy
TEST_CITY_FRACTION = 0.20
RANDOM_STATE       = 42


class WageInflationClassifier:
    """
    XGBoost binary classifier — wage-adjusted food inflation prediction.

    Attributes
    ----------
    model            : XGBClassifier (tuned)
    best_params      : dict
    encoders         : dict of LabelEncoders
    feature_cols     : list of feature names
    train_cities     : ndarray of city names in training set
    test_cities      : ndarray of city names in test set
    X_train, X_test  : pd.DataFrame
    y_train, y_test  : pd.Series
    scale_pos_weight : float (class-imbalance correction)
    """

    CATEGORICAL_COLS = ['Item_Category', 'Item_Key']

    def __init__(self):
        self.model            = None
        self.best_params      = None
        self.encoders         = {}
        self.feature_cols     = None
        self.train_cities     = None
        self.test_cities      = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.scale_pos_weight = 1.0

    # ================================================================
    # 1. LOAD & PREPROCESS
    # ================================================================
    def load_and_preprocess(self, filepath: str) -> pd.DataFrame:
        """
        Load raw CSV, engineer the binary target, encode categoricals,
        and return a fully prepared item-level DataFrame.
        """
        print("=" * 65)
        print("STEP 1 — DATA LOADING & PREPROCESSING")
        print("=" * 65)

        df = pd.read_csv(filepath)
        print(f"Raw shape       : {df.shape}")
        print(f"Missing values  : {df.isnull().sum().sum()}")
        print(f"Duplicate rows  : {df.duplicated().sum()}")

        # Feature engineering
        df['Month_Num'] = (
            pd.to_datetime(df['Month']).dt.month
            + (pd.to_datetime(df['Month']).dt.year - 2025) * 12
        )
        df['Log_Population'] = np.log1p(df['Population_Estimate'])

        # Binary target: inflation > wage benchmark
        df['Wage_Adjusted_High_Cost'] = (
            df['YoY_Inflation_Estimate_Pct'] > WAGE_BENCHMARK_PCT
        ).astype(int)

        vc = df['Wage_Adjusted_High_Cost'].value_counts()
        print(f"\nWage benchmark  : {WAGE_BENCHMARK_PCT}%  (World Bank/ILO 2025 proxy)")
        print(f"High Cost (1)   : {vc[1]:,}  ({vc[1]/len(df)*100:.1f}%)")
        print(f"Low  Cost (0)   : {vc[0]:,}  ({vc[0]/len(df)*100:.1f}%)")
        print(f"Imbalance ratio : {vc.min()/vc.max():.2f}")

        # Encode categoricals
        for col in self.CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
            print(f"Encoded '{col}': {len(le.classes_)} categories")

        # Feature set — no price columns (prevent region-level leakage)
        self.feature_cols = [
            'Item_Category',
            'Item_Key',
            'Month_Num',
            'FAO_Index_Value',
            'Log_Population'
        ]
        print(f"\nFeature set     : {self.feature_cols}")
        return df

    # ================================================================
    # 2. EDA
    # ================================================================
    def exploratory_analysis(self, df: pd.DataFrame, save_plots: bool = True):
        """EDA: inflation distribution, class balance, continent breakdown, price by category."""
        print("\n" + "=" * 65)
        print("STEP 2 — EXPLORATORY DATA ANALYSIS")
        print("=" * 65)

        city_df = df.drop_duplicates('City')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Wage-Adjusted Food Inflation — EDA", fontsize=14, fontweight='bold')

        # 1: Inflation distribution
        city_df['YoY_Inflation_Estimate_Pct'].hist(
            ax=axes[0, 0], bins=20, color='steelblue', edgecolor='white'
        )
        axes[0, 0].axvline(WAGE_BENCHMARK_PCT, color='red', linestyle='--',
                            label=f'Benchmark {WAGE_BENCHMARK_PCT}%')
        axes[0, 0].set_title('YoY Food Inflation by City')
        axes[0, 0].set_xlabel('Inflation (%)')
        axes[0, 0].legend()

        # 2: Class balance
        vc = df['Wage_Adjusted_High_Cost'].value_counts()
        axes[0, 1].bar(['Low Cost (0)', 'High Cost (1)'],
                        [vc[0], vc[1]], color=['#2ecc71', '#e74c3c'])
        axes[0, 1].set_title('Target Class Balance')
        axes[0, 1].set_ylabel('Row count')
        for i, v in enumerate([vc[0], vc[1]]):
            axes[0, 1].text(i, v + 50, str(v), ha='center', fontweight='bold')

        # 3: Inflation by continent
        cont_infl = city_df.groupby('Continent')['YoY_Inflation_Estimate_Pct'].mean()
        cont_infl.sort_values(ascending=False).plot(
            kind='bar', ax=axes[1, 0], color='steelblue', edgecolor='white'
        )
        axes[1, 0].axhline(WAGE_BENCHMARK_PCT, color='red', linestyle='--',
                            label=f'Benchmark {WAGE_BENCHMARK_PCT}%')
        axes[1, 0].set_title('Avg Inflation by Continent')
        axes[1, 0].set_ylabel('Avg Inflation (%)')
        axes[1, 0].tick_params(axis='x', rotation=30)
        axes[1, 0].legend()

        # 4: Price by item category
        cat_le = self.encoders.get('Item_Category')
        df_plot = df.copy()
        if cat_le:
            df_plot['Category'] = cat_le.inverse_transform(df_plot['Item_Category'])
        df_plot.boxplot(column='Price_USD', by='Category', ax=axes[1, 1])
        axes[1, 1].set_title('Price (USD) by Item Category')
        axes[1, 1].set_xlabel('')
        axes[1, 1].tick_params(axis='x', rotation=40)
        plt.sca(axes[1, 1])
        plt.title('Price (USD) by Item Category')

        plt.tight_layout()
        if save_plots:
            plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
            print("EDA saved: eda_plots.png")
        plt.show()
        plt.close()

        print("\nInflation by region:")
        r = df.drop_duplicates('Region')[['Region', 'YoY_Inflation_Estimate_Pct']]
        for _, row in r.sort_values('YoY_Inflation_Estimate_Pct', ascending=False).iterrows():
            flag = " ← exceeds benchmark" if row[1] > WAGE_BENCHMARK_PCT else ""
            print(f"  {row['Region']:<22} {row['YoY_Inflation_Estimate_Pct']:.1f}%{flag}")

    # ================================================================
    # 3. CITY-BASED SPLIT
    # ================================================================
    def split_data(self, df: pd.DataFrame):
        """
        Split by city (not by row) to prevent region-level label leakage.
        Cities are entirely held out from the test set.
        """
        print("\n" + "=" * 65)
        print("STEP 3 — CITY-BASED TRAIN / TEST SPLIT")
        print("=" * 65)

        cities = df['City'].unique()
        rng = np.random.default_rng(RANDOM_STATE)
        rng.shuffle(cities)

        n_test = int(len(cities) * TEST_CITY_FRACTION)
        self.test_cities  = cities[:n_test]
        self.train_cities = cities[n_test:]

        train_df = df[df['City'].isin(self.train_cities)]
        test_df  = df[df['City'].isin(self.test_cities)]

        self.X_train = train_df[self.feature_cols]
        self.y_train = train_df['Wage_Adjusted_High_Cost']
        self.X_test  = test_df[self.feature_cols]
        self.y_test  = test_df['Wage_Adjusted_High_Cost']

        vc = self.y_train.value_counts()
        self.scale_pos_weight = vc[0] / vc[1]

        print(f"Train cities    : {len(self.train_cities)}  ({self.X_train.shape[0]:,} rows)")
        print(f"Test cities     : {len(self.test_cities)}   ({self.X_test.shape[0]:,} rows)")
        print(f"scale_pos_weight: {self.scale_pos_weight:.3f}")
        print("\nHeld-out test cities:")
        for c in sorted(self.test_cities):
            row = df[df['City'] == c].iloc[0]
            print(f"  {c:<20} {row['Region']:<18} {row['YoY_Inflation_Estimate_Pct']:.1f}%")

    # ================================================================
    # 4. BASELINE MODEL
    # ================================================================
    def train_baseline(self) -> dict:
        """Train default XGBoost as performance baseline."""
        print("\n" + "=" * 65)
        print("STEP 4 — BASELINE XGBoost MODEL")
        print("=" * 65)

        baseline = XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            eval_metric='logloss', scale_pos_weight=self.scale_pos_weight
        )
        baseline.fit(self.X_train, self.y_train)
        y_pred = baseline.predict(self.X_test)
        y_prob = baseline.predict_proba(self.X_test)[:, 1]

        metrics = self._compute_metrics(self.y_test, y_pred, y_prob)
        print("\nBaseline metrics:")
        self._print_metrics(metrics)
        return metrics

    # ================================================================
    # 5. HYPERPARAMETER TUNING
    # ================================================================
    def tune_hyperparameters(self) -> dict:
        """GridSearchCV with 5-fold Stratified CV, scoring on F1."""
        print("\n" + "=" * 65)
        print("STEP 5 — HYPERPARAMETER TUNING  (GridSearchCV)")
        print("=" * 65)

        param_grid = {
            'n_estimators'    : [100, 200, 300],
            'max_depth'       : [2, 3, 4],
            'learning_rate'   : [0.01, 0.05, 0.1],
            'subsample'       : [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
            'min_child_weight': [5, 10, 20],
        }

        xgb = XGBClassifier(
            random_state=RANDOM_STATE, eval_metric='logloss',
            scale_pos_weight=self.scale_pos_weight
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            xgb, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1
        )
        print("Running GridSearchCV — this may take a few minutes…")
        grid_search.fit(self.X_train, self.y_train)

        self.best_params = grid_search.best_params_
        self.model       = grid_search.best_estimator_

        print(f"\nBest parameters:")
        for k, v in self.best_params.items():
            print(f"  {k:<22}: {v}")
        print(f"Best CV F1: {grid_search.best_score_:.4f}")

        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        metrics = self._compute_metrics(self.y_test, y_pred, y_prob)
        print("\nTuned model metrics:")
        self._print_metrics(metrics)
        return metrics

    # ================================================================
    # 6. EVALUATE
    # ================================================================
    def evaluate(self, save_plots: bool = True):
        """Classification report, confusion matrix, ROC curve, feature importance."""
        print("\n" + "=" * 65)
        print("STEP 6 — MODEL EVALUATION")
        print("=" * 65)

        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred,
                                    target_names=['Low Cost (0)', 'High Cost (1)']))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("XGBoost — Wage-Adjusted Inflation Model Evaluation",
                     fontsize=13, fontweight='bold')

        ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(self.y_test, y_pred),
            display_labels=['Low Cost', 'High Cost']
        ).plot(ax=axes[0], colorbar=False, cmap='Blues')
        axes[0].set_title('Confusion Matrix')

        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        auc = roc_auc_score(self.y_test, y_prob)
        axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.4f}')
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()

        fi = pd.Series(self.model.feature_importances_,
                       index=self.feature_cols).sort_values(ascending=True)
        fi.plot(kind='barh', ax=axes[2], color='steelblue')
        axes[2].set_title('Feature Importance (Gain)')
        axes[2].set_xlabel('Score')

        plt.tight_layout()
        if save_plots:
            plt.savefig('evaluation_plots.png', dpi=150, bbox_inches='tight')
            print("Evaluation plots saved: evaluation_plots.png")
        plt.show()
        plt.close()

    # ================================================================
    # 7. SHAP
    # ================================================================
    def shap_analysis(self, save_plots: bool = True):
        """SHAP summary plot for global feature interpretability."""
        print("\n" + "=" * 65)
        print("STEP 7 — SHAP MODEL INTERPRETATION")
        print("=" * 65)

        explainer   = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X_test,
                          feature_names=self.feature_cols, show=False)
        plt.title("SHAP Summary — Wage-Adjusted High-Cost Prediction",
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        if save_plots:
            plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
            print("SHAP plot saved: shap_summary.png")
        plt.show()
        plt.close()

        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({'Feature': self.feature_cols,
                                'Mean|SHAP|': mean_shap}
                               ).sort_values('Mean|SHAP|', ascending=False)
        print("\nTop features by Mean |SHAP|:")
        print(shap_df.to_string(index=False))

    # ================================================================
    # 8. CLASS IMBALANCE
    # ================================================================
    def check_class_imbalance(self, df: pd.DataFrame):
        """Report class balance and document the correction strategy."""
        print("\n" + "=" * 65)
        print("CLASS IMBALANCE ASSESSMENT")
        print("=" * 65)
        vc = df['Wage_Adjusted_High_Cost'].value_counts()
        ratio = vc.min() / vc.max()
        print(f"High Cost (1) : {vc[1]:,}")
        print(f"Low  Cost (0) : {vc[0]:,}")
        print(f"Ratio         : {ratio:.2f}")
        if ratio < 0.80:
            print(f"\nModerate imbalance — using scale_pos_weight = {vc[0]/vc[1]:.3f}")
            print("XGBoost up-weights the minority class during training.")
        else:
            print("\nWell-balanced — no correction needed.")

    # ================================================================
    # HELPERS
    # ================================================================
    def _compute_metrics(self, y_true, y_pred, y_prob) -> dict:
        return {
            'Accuracy' : round(accuracy_score(y_true, y_pred),  4),
            'Precision': round(precision_score(y_true, y_pred), 4),
            'Recall'   : round(recall_score(y_true, y_pred),    4),
            'F1-Score' : round(f1_score(y_true, y_pred),        4),
            'AUC-ROC'  : round(roc_auc_score(y_true, y_prob),   4),
        }

    def _print_metrics(self, m: dict):
        print(f"  {'Metric':<12} {'Value':>8}")
        print(f"  {'-'*22}")
        for k, v in m.items():
            print(f"  {k:<12} {v:>8.4f}")

    def get_evaluation_table(self) -> pd.DataFrame:
        """One-row DataFrame of final metrics for team consolidation."""
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        m = self._compute_metrics(self.y_test, y_pred, y_prob)
        df = pd.DataFrame([m])
        df.insert(0, 'Model', 'XGBoost (Tuned)')
        df.insert(1, 'Best Params', str(self.best_params))
        return df


# ======================================================================
# MAIN
# ======================================================================
if __name__ == '__main__':
    DATA_PATH = 'breakfast basket.csv'

    clf = WageInflationClassifier()
    df  = clf.load_and_preprocess(DATA_PATH)
    clf.check_class_imbalance(df)
    clf.exploratory_analysis(df)
    clf.split_data(df)
    clf.train_baseline()
    clf.tune_hyperparameters()
    clf.evaluate()
    clf.shap_analysis()

    print("\n" + "=" * 65)
    print("TEAM EVALUATION TABLE")
    print("=" * 65)
    print(clf.get_evaluation_table().to_string(index=False))
