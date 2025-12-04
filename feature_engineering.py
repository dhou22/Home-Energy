"""
Feature Engineering CORRIGÉ - Sans Data Leakage

Corrections appliquées basées sur diagnostic:
1. ❌ RETIRÉ: Sub_metering (composantes du target)
2. ❌ RETIRÉ: Global_intensity (relation déterministe avec target)
3. ❌ RETIRÉ: Voltage (idem)
4. ❌ RETIRÉ: apparent_power (calcul direct du target)
5. ⚠️  MODIFIÉ: Lag features avec horizon de prévision approprié

Performances attendues post-correction:
- R²: 0.80 - 0.92 (réaliste)
- RMSE: 0.15 - 0.40 kWh
- MAPE: 8% - 15%

Auteur: Dhouha Meliane (corrigé)
Date: Décembre 2024
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SafeEnergyFeatureEngineering:
    """
    Feature engineering SANS data leakage pour prédiction énergétique.
    
    Principes anti-leakage:
    1. Exclure toutes mesures électriques corrélées au target
    2. Utiliser SEULEMENT features temporelles + lag target
    3. Lag > forecast_horizon (éviter look-ahead bias)
    """
    
    # VARIABLES INTERDITES (causent leakage)
    FORBIDDEN_VARS = [
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',  # Composantes target
        'Global_intensity',  # P = V × I / 1000
        'Voltage',  # Idem
        'Global_reactive_power',  # Dérivée target
    ]
    
    def __init__(self, df: pd.DataFrame, 
                 timestamp_col: str = 'timestamp',
                 forecast_horizon: int = 1):
        """
        Args:
            df: DataFrame données brutes
            timestamp_col: Colonne timestamp
            forecast_horizon: Horizon prédiction (en pas de temps)
                              Exemple: 1 = prédire 1 minute en avance
                                      60 = prédire 1 heure en avance
        """
        self.df = df.copy()
        self.timestamp_col = timestamp_col
        self.forecast_horizon = forecast_horizon
        self.feature_report = {}
        
        # Parse et tri chronologique
        if not pd.api.types.is_datetime64_any_dtype(self.df[timestamp_col]):
            self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        
        self.df = self.df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Retirer variables interdites
        self._remove_forbidden_vars()
        
        logger.info(f"[INIT] Feature Engineering Safe")
        logger.info(f"  Records: {len(self.df):,}")
        logger.info(f"  Forecast horizon: {forecast_horizon} pas")
        
    def _remove_forbidden_vars(self):
        """Retire variables causant leakage."""
        removed = []
        for var in self.FORBIDDEN_VARS:
            if var in self.df.columns:
                self.df = self.df.drop(columns=[var])
                removed.append(var)
        
        if removed:
            logger.info(f"[PROTECTION] Variables retirées: {removed}")
    
    def create_temporal_features(self) -> pd.DataFrame:
        """
        Features temporelles (TOUJOURS safe).
        """
        logger.info("[1/3] Features temporelles")
        
        df = self.df.copy()
        ts = df[self.timestamp_col]
        
        # Basiques
        df['year'] = ts.dt.year
        df['month'] = ts.dt.month
        df['day'] = ts.dt.day
        df['hour'] = ts.dt.hour
        df['minute'] = ts.dt.minute
        df['dayofweek'] = ts.dt.dayofweek
        df['dayofyear'] = ts.dt.dayofyear
        df['weekofyear'] = ts.dt.isocalendar().week
        df['quarter'] = ts.dt.quarter
        
        # Calendrier
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_workday'] = (df['dayofweek'] < 5).astype(int)
        
        # Saison
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,  # Hiver
            3: 1, 4: 1, 5: 1,   # Printemps
            6: 2, 7: 2, 8: 2,   # Été
            9: 3, 10: 3, 11: 3  # Automne
        })
        
        # Période journée
        df['time_of_day'] = pd.cut(
            df['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype(int)
        
        # Peak hours (consommation élevée)
        df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
        
        # Cycliques (sin/cos)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        
        temporal_count = len([c for c in df.columns if c not in self.df.columns])
        logger.info(f"  ✓ {temporal_count} features temporelles créées")
        
        self.df = df
        return self.df
    
    def create_safe_lag_features(self, 
                                 target_col: str = 'Global_active_power',
                                 max_lag: int = 168) -> pd.DataFrame:
        """
        Lag features du TARGET avec protection anti-leakage.
        
        RÈGLE CRITIQUE (Sertis 2024):
        "Lag period DOIT être >= forecast_horizon"
        
        Exemple:
        - Si forecast_horizon = 1 (prédire 1 min en avance)
          → lag_1 OK (on connaît t-1)
        - Si forecast_horizon = 60 (prédire 1h en avance)
          → lag_1 à lag_59 INTERDITS (futur proche inconnu)
          → lag_60+ OK
        
        Args:
            target_col: Variable cible
            max_lag: Lag maximum (défaut 168 = 1 semaine si minute-level)
        """
        logger.info(f"[2/3] Lag features (horizon={self.forecast_horizon})")
        
        df = self.df.copy()
        
        if target_col not in df.columns:
            logger.warning(f"Target '{target_col}' non trouvé")
            return self.df
        
        # Lags stratégiques (adaptés au forecast horizon)
        min_lag = max(1, self.forecast_horizon)
        
        strategic_lags = []
        
        # Lag immédiat (si horizon=1)
        if self.forecast_horizon == 1:
            strategic_lags.extend([1, 2, 3, 5, 10])
        
        # Lags pattern journalier (60 min/heure, 1440 min/jour)
        strategic_lags.extend([
            60,    # 1 heure avant
            120,   # 2 heures
            360,   # 6 heures
            720,   # 12 heures
            1440,  # 1 jour (même heure hier)
        ])
        
        # Pattern hebdomadaire
        strategic_lags.append(10080)  # 1 semaine (7*24*60)
        
        # Filtrer lags >= forecast_horizon
        valid_lags = [lag for lag in strategic_lags if lag >= min_lag]
        
        logger.info(f"  Lags utilisés: {valid_lags}")
        
        for lag in valid_lags:
            if lag <= len(df):
                feature_name = f'target_lag_{lag}'
                df[feature_name] = df[target_col].shift(lag)
        
        lag_count = len([c for c in df.columns if 'lag' in c])
        logger.info(f"  ✓ {lag_count} lag features créées")
        
        self.df = df
        return self.df
    
    def create_safe_rolling_features(self,
                                    target_col: str = 'Global_active_power',
                                    windows: List[int] = None) -> pd.DataFrame:
        """
        Rolling statistics du TARGET (avec shift pour éviter leakage).
        
        IMPORTANT: shift(1) AVANT rolling pour éviter look-ahead.
        """
        logger.info("[3/3] Rolling features")
        
        df = self.df.copy()
        
        if target_col not in df.columns:
            logger.warning(f"Target '{target_col}' non trouvé")
            return self.df
        
        if windows is None:
            windows = [60, 360, 1440]  # 1h, 6h, 24h (si minute-level)
        
        # Filtrer windows >= forecast_horizon
        valid_windows = [w for w in windows if w >= self.forecast_horizon]
        
        for window in valid_windows:
            # CRITIQUE: shift(1) AVANT rolling
            shifted = df[target_col].shift(1)
            
            # Rolling mean
            df[f'target_rolling_mean_{window}'] = shifted.rolling(
                window=window, min_periods=1
            ).mean()
            
            # Rolling std
            df[f'target_rolling_std_{window}'] = shifted.rolling(
                window=window, min_periods=1
            ).std()
        
        rolling_count = len([c for c in df.columns if 'rolling' in c])
        logger.info(f"  ✓ {rolling_count} rolling features créées")
        
        self.df = df
        return self.df
    
    def create_all_safe_features(self, 
                                target_col: str = 'Global_active_power') -> pd.DataFrame:
        """
        Pipeline complet SANS leakage.
        """
        logger.info("=" * 80)
        logger.info("FEATURE ENGINEERING SAFE (NO LEAKAGE)")
        logger.info("=" * 80)
        
        initial_shape = self.df.shape
        
        # Pipeline
        self.create_temporal_features()
        self.create_safe_lag_features(target_col=target_col)
        self.create_safe_rolling_features(target_col=target_col)
        
        # Supprimer NaN
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        dropped = initial_rows - len(self.df)
        
        logger.info("=" * 80)
        logger.info(f"TERMINÉ: {initial_shape} → {self.df.shape}")
        logger.info(f"Features ajoutées: {self.df.shape[1] - initial_shape[1]}")
        logger.info(f"Lignes NaN drop: {dropped:,}")
        logger.info("=" * 80)
        
        # Validation anti-leakage
        self._validate_no_leakage(target_col)
        
        return self.df
    
    def _validate_no_leakage(self, target_col: str):
        """
        Validation post-création: aucune feature avec r > 0.95.
        """
        logger.info("\n[VALIDATION ANTI-LEAKAGE]")
        
        exclude = [self.timestamp_col, 'Date', 'Time', target_col]
        feature_cols = [
            c for c in self.df.columns 
            if c not in exclude and self.df[c].dtype in ['int64', 'float64']
        ]
        
        target = self.df[target_col]
        high_corr = []
        
        for col in feature_cols:
            corr = self.df[col].corr(target)
            if abs(corr) > 0.95:
                high_corr.append((col, corr))
        
        if high_corr:
            logger.warning(f"  ⚠️  {len(high_corr)} features avec r > 0.95:")
            for feat, corr in high_corr:
                logger.warning(f"    - {feat}: r = {corr:.6f}")
            logger.warning("  → Investiguer ces features!")
        else:
            logger.info("  ✓ Aucune corrélation anormale détectée")
            logger.info("  ✓ Dataset validé sans leakage apparent")


# ========================================================================
# SCRIPT PRINCIPAL
# ========================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 80)
    print("RECRÉATION FEATURES SANS DATA LEAKAGE")
    print("=" * 80 + "\n")
    
    # ÉTAPE 1: Charger données BRUTES (pas processed!)
    print("[1/4] Chargement données brutes...")
    
    try:
        # Option A: Données brutes originales
        df_raw = pd.read_csv(
            './data/raw/household_power_consumption.txt',
            sep=';',
            na_values=['?'],
            parse_dates={'timestamp': ['Date', 'Time']}
        )
        print(f"  ✓ Chargé: {df_raw.shape}")
        
    except FileNotFoundError:
        print("  ❌ Fichier brut non trouvé")
        print("  Utilisation données cleaned (si disponible)...")
        
        df_raw = pd.read_csv('./data/processed/data_cleaned.csv')
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        print(f"  ✓ Chargé (cleaned): {df_raw.shape}")
    
    # Nettoyage NaN
    df_raw = df_raw.dropna()
    print(f"  Après NaN drop: {df_raw.shape}")
    
    # ÉTAPE 2: Feature engineering SAFE
    print("\n[2/4] Feature engineering sans leakage...")
    
    fe = SafeEnergyFeatureEngineering(
        df=df_raw,
        timestamp_col='timestamp',
        forecast_horizon=1  # Prédire 1 minute en avance
    )
    
    df_safe = fe.create_all_safe_features(target_col='Global_active_power')
    
    # ÉTAPE 3: Sauvegarde
    print("\n[3/4] Sauvegarde dataset safe...")
    
    output_path = './data/processed/data_with_safe_features.csv'
    df_safe.to_csv(output_path, index=False)
    print(f"  ✓ Sauvegardé: {output_path}")
    print(f"  Shape: {df_safe.shape}")
    
    # ÉTAPE 4: Rapport
    print("\n[4/4] Résumé features créées...")
    
    feature_cols = [c for c in df_safe.columns if c not in ['timestamp', 'Date', 'Time', 'Global_active_power']]
    
    print(f"\n  Total features: {len(feature_cols)}")
    print(f"  Samples: {len(df_safe):,}")
    
    print("\n  Categories:")
    temporal = [c for c in feature_cols if any(x in c for x in ['hour', 'day', 'month', 'year', 'sin', 'cos', 'weekend', 'season', 'peak'])]
    lag = [c for c in feature_cols if 'lag' in c]
    rolling = [c for c in feature_cols if 'rolling' in c]
    
    print(f"    - Temporelles: {len(temporal)}")
    print(f"    - Lag: {len(lag)}")
    print(f"    - Rolling: {len(rolling)}")
    
    print("\n" + "=" * 80)
    print("✓ DATASET SAFE PRÊT POUR ENTRAÎNEMENT")
    print("=" * 80)
    print("\nÉTAPES SUIVANTES:")
    print("1. Ré-entraîner modèles avec: data_with_safe_features.csv")
    print("2. Attendre performances réalistes:")
    print("   - R²: 0.75 - 0.92")
    print("   - RMSE: 0.15 - 0.40 kWh")
    print("   - MAPE: 8% - 15%")
    print("\n3. Si R² > 0.95 persiste → Partager résultats pour analyse")