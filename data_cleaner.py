import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnergyDataCleaner:
    """
    Nettoyage et préparation des données énergétiques.
    
    Techniques basées sur:
    [6] Zhang & Chen (2022). Data Quality Assessment for 
        Smart Home Energy Analytics.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_report = {}
        self.initial_shape = df.shape
    
    def handle_missing_values(self, 
                              strategy: str = 'interpolate') -> pd.DataFrame:
        """Gestion des valeurs manquantes."""
        logger.info(f"Traitement des valeurs manquantes (strategie: {strategy})")
        initial_missing = self.df.isnull().sum().sum()
        
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'interpolate':
            # Sauvegarder timestamp si présent
            timestamp_col = None
            if 'timestamp' in self.df.columns:
                timestamp_col = self.df['timestamp'].copy()
                self.df = self.df.set_index('timestamp')
            
            # Interpolation linéaire pour séries temporelles
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            if self.df.index.dtype == 'datetime64[ns]':
                # Interpolation temporelle si index datetime
                self.df[numeric_cols] = self.df[numeric_cols].interpolate(
                    method='time',
                    limit_direction='both'
                )
            else:
                # Interpolation linéaire standard sinon
                self.df[numeric_cols] = self.df[numeric_cols].interpolate(
                    method='linear',
                    limit_direction='both'
                )
            
            # Restaurer timestamp comme colonne
            if timestamp_col is not None:
                self.df = self.df.reset_index()
        elif strategy == 'forward_fill':
            self.df = self.df.fillna(method='ffill', limit=60)
        
        final_missing = self.df.isnull().sum().sum()
        
        self.cleaning_report['missing_values'] = {
            'initial': initial_missing,
            'final': final_missing,
            'removed': initial_missing - final_missing,
            'percentage_cleaned': (1 - final_missing/max(initial_missing, 1)) * 100
        }
        
        logger.info(f">> Valeurs manquantes reduites: {initial_missing} -> {final_missing}")
        
        return self.df
    
    def remove_outliers(self, 
                        columns: List[str], 
                        method: str = 'iqr') -> pd.DataFrame:
        """Détection et suppression des outliers."""
        logger.info(f"Détection outliers (méthode: {method})")
        initial_rows = len(self.df)
        
        for col in columns:
            if col not in self.df.columns:
                logger.warning(f"Colonne {col} non trouvée, ignorée")
                continue
            
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (
                    (self.df[col] >= lower_bound) & 
                    (self.df[col] <= upper_bound)
                )
                self.df = self.df[outliers_mask]
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                self.df = self.df[z_scores < 3]
        
        final_rows = len(self.df)
        
        self.cleaning_report['outliers'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'removed': initial_rows - final_rows,
            'percentage_removed': ((initial_rows - final_rows) / initial_rows) * 100
        }
        
        logger.info(f">> Outliers supprimes: {initial_rows - final_rows} lignes ({self.cleaning_report['outliers']['percentage_removed']:.2f}%)")
        
        return self.df
    
    def validate_physical_constraints(self) -> pd.DataFrame:
        """Validation des contraintes physiques."""
        logger.info("Validation des contraintes physiques")
        constraints = {
            'Global_active_power': (0, 15),
            'Voltage': (220, 260),
            'Global_intensity': (0, 60)
        }
        initial = len(self.df)
        
        for col, (min_val, max_val) in constraints.items():
            if col in self.df.columns:
                mask = (self.df[col] >= min_val) & (self.df[col] <= max_val)
                invalid_count = (~mask).sum()
                self.df = self.df[mask]
                logger.info(f"  {col}: {invalid_count} valeurs invalides supprimées")
        
        final = len(self.df)
        
        self.cleaning_report['physical_validation'] = {
            'initial': initial,
            'final': final,
            'invalid_removed': initial - final
        }
        
        logger.info(f">> Validation physique: {initial - final} enregistrements invalides supprimes")
        
        return self.df
    
    def remove_duplicates(self) -> pd.DataFrame:
        """Suppression des doublons sur timestamp."""
        logger.info("Suppression des doublons")
        initial = len(self.df)
        self.df = self.df.drop_duplicates(subset=['timestamp'], keep='first')
        final = len(self.df)
        
        self.cleaning_report['duplicates'] = {
            'removed': initial - final
        }
        
        logger.info(f">> Doublons supprimes: {initial - final}")
        
        return self.df
    
    def clean_all(self, 
                  remove_outliers_flag: bool = True,
                  fill_gaps: bool = False) -> pd.DataFrame:
        """Pipeline complet de nettoyage."""
        logger.info("=" * 70)
        logger.info("DÉBUT DU NETTOYAGE DES DONNÉES")
        logger.info("=" * 70)
        
        self.handle_missing_values(strategy='interpolate')
        self.remove_duplicates()
        self.validate_physical_constraints()
        
        if remove_outliers_flag:
            outlier_cols = ['Global_active_power', 'Voltage', 'Global_intensity']
            self.remove_outliers(columns=outlier_cols, method='iqr')
        
        logger.info("=" * 70)
        logger.info("NETTOYAGE TERMINÉ")
        logger.info(f"Forme initiale: {self.initial_shape}")
        logger.info(f"Forme finale:   {self.df.shape}")
        logger.info(f"Réduction:      {self.initial_shape[0] - self.df.shape[0]:,} lignes ({((self.initial_shape[0] - self.df.shape[0])/self.initial_shape[0]*100):.2f}%)")
        logger.info("=" * 70)
        
        return self.df
    
    def get_cleaning_report(self) -> Dict:
        """Retourne le rapport de nettoyage."""
        return self.cleaning_report
    
    def print_summary_before_after(self, df_original: pd.DataFrame):
        """Affiche un resume detaille avant/apres le nettoyage."""
        print("\n" + "=" * 80)
        print("ANALYSE QUALITE DES DONNEES - SMART HOME ENERGY MONITORING")
        print("=" * 80)
        
        # Resume AVANT nettoyage
        print("\n[*] ETAT INITIAL DU DATASET")
        print("-" * 80)
        print(f"  Nombre total d'enregistrements : {len(df_original):,}")
        print(f"  Nombre de variables            : {df_original.shape[1]}")
        
        if 'timestamp' in df_original.columns:
            print(f"  Periode couverte               : {df_original['timestamp'].min()} -> {df_original['timestamp'].max()}")
        
        memory_mb = df_original.memory_usage(deep=True).sum() / (1024**2)
        print(f"  Utilisation memoire            : {memory_mb:.2f} MB")
        
        missing = df_original.isnull().sum().sum()
        print(f"  Valeurs manquantes totales     : {missing:,}")
        
        if 'timestamp' in df_original.columns:
            duplicates = df_original['timestamp'].duplicated().sum()
            print(f"  Doublons detectes              : {duplicates:,}")
        
        # Statistiques par colonne
        print("\n[*] STATISTIQUES DESCRIPTIVES (AVANT)")
        print("-" * 80)
        numeric_cols = df_original.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Limiter a 5 colonnes
            if col in df_original.columns:
                print(f"\n  {col}:")
                print(f"    - Moyenne  : {df_original[col].mean():.4f}")
                print(f"    - Std      : {df_original[col].std():.4f}")
                print(f"    - Min      : {df_original[col].min():.4f}")
                print(f"    - Max      : {df_original[col].max():.4f}")
                print(f"    - Manquant : {df_original[col].isnull().sum()}")
        
        # Resume APRES nettoyage
        print("\n" + "=" * 80)
        print("[*] ETAT APRES NETTOYAGE")
        print("-" * 80)
        print(f"  Nombre total d'enregistrements : {len(self.df):,}")
        print(f"  Nombre de variables            : {self.df.shape[1]}")
        
        if 'timestamp' in self.df.columns:
            print(f"  Periode couverte               : {self.df['timestamp'].min()} -> {self.df['timestamp'].max()}")
        
        memory_mb_clean = self.df.memory_usage(deep=True).sum() / (1024**2)
        print(f"  Utilisation memoire            : {memory_mb_clean:.2f} MB")
        
        missing_clean = self.df.isnull().sum().sum()
        print(f"  Valeurs manquantes totales     : {missing_clean:,}")
        
        # Rapport detaille du nettoyage
        print("\n" + "=" * 80)
        print("[*] RAPPORT DETAILLE DU NETTOYAGE")
        print("=" * 80)
        
        for step, details in self.cleaning_report.items():
            print(f"\n>> {step.upper().replace('_', ' ')}:")
            for key, value in details.items():
                if isinstance(value, float):
                    print(f"   - {key}: {value:.2f}")
                else:
                    print(f"   - {key}: {value:,}" if isinstance(value, int) else f"   - {key}: {value}")
        
        # Impact global
        print("\n" + "=" * 80)
        print("[*] IMPACT GLOBAL DU NETTOYAGE")
        print("=" * 80)
        rows_removed = len(df_original) - len(self.df)
        pct_removed = (rows_removed / len(df_original)) * 100
        print(f"  Enregistrements supprimes      : {rows_removed:,} ({pct_removed:.2f}%)")
        print(f"  Enregistrements conserves      : {len(self.df):,} ({100-pct_removed:.2f}%)")
        
        quality_improvement = ((missing - missing_clean) / max(missing, 1)) * 100
        print(f"  Amelioration qualite donnees   : {quality_improvement:.2f}%")
        
        print("=" * 80 + "\n")


# ========================================================================
# SCRIPT DE TEST PRINCIPAL
# ========================================================================

def test_energy_cleaner(filepath: str):
    """
    Fonction de test principale pour le EnergyDataCleaner.
    
    Args:
        filepath: Chemin vers le fichier household_power_consumption.txt
    """
    
    print("\n" + "-" * 80)
    print("TEST DU MODULE DE NETTOYAGE DES DONNEES ENERGETIQUES")
    print("-" * 80 + "\n")
    
    # Chargement des donnees
    print("[INFO] Chargement des donnees...")
    try:
        df = pd.read_csv(
            filepath,
            sep=';',
            parse_dates={'timestamp': ['Date', 'Time']},
            na_values=['?', ''],
            low_memory=False
        )
        print(f"[OK] Donnees chargees: {df.shape[0]:,} lignes, {df.shape[1]} colonnes\n")
    except Exception as e:
        print(f"[ERREUR] Erreur lors du chargement: {e}")
        return
    
    # Conversion des types
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Creer une copie pour comparaison
    df_original = df.copy()
    
    # Initialisation du cleaner
    cleaner = EnergyDataCleaner(df)
    
    # Execution du nettoyage
    print("\n[INFO] Lancement du pipeline de nettoyage...\n")
    df_clean = cleaner.clean_all(remove_outliers_flag=True, fill_gaps=False)
    
    # Affichage du rapport complet
    cleaner.print_summary_before_after(df_original)
    
    # Sauvegarde optionnelle
    output_path = './data/processed/data_cleaned.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"[OK] Donnees nettoyees sauvegardees: {output_path}\n")
    
    return df_clean, cleaner


# ========================================================================
# EXECUTION DU TEST
# ========================================================================

if __name__ == "__main__":
    # IMPORTANT: Remplacez ce chemin par celui de votre fichier
    FILEPATH = "./data/raw/household_power_consumption.txt"
    
    # Vous pouvez aussi tester avec un extrait limite:
    # Pour charger seulement les 10000 premieres lignes:
    # df = pd.read_csv(FILEPATH, sep=';', nrows=10000, ...)
    
    print("\n[CONFIG] Configuration du test:")
    print(f"   Fichier: {FILEPATH}")
    print(f"   Mode: Analyse complete\n")
    
    df_clean, cleaner = test_energy_cleaner(FILEPATH)
    
    print("\n[SUCCESS] Test termine avec succes!")
    print(f"   Dataset nettoye disponible: df_clean ({len(df_clean):,} lignes)")
    print(f"   Rapport accessible via: cleaner.get_cleaning_report()\n")