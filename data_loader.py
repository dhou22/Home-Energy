import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnergyDataLoader:
    """
    Classe pour charger et parser les données énergétiques UCI.
    
    Référence:
    [4] Hebrail, G. & Berard, A. (2012). Individual household 
        electric power consumption. UCI ML Repository.
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Charge les données avec parsing optimisé.
        
        Returns:
            pd.DataFrame: Données brutes chargées
        """
        logger.info(f"Chargement des données depuis {self.filepath}")
        
        # Colonnes du dataset UCI
        columns = [
            'Date', 'Time', 'Global_active_power',
            'Global_reactive_power', 'Voltage',
            'Global_intensity', 'Sub_metering_1',
            'Sub_metering_2', 'Sub_metering_3'
        ]
        
        # Chargement avec types optimisés
        self.df = pd.read_csv(
            self.filepath,
            sep=';',
            names=columns,
            skiprows=1,
            na_values='?',
            low_memory=False,
            parse_dates={'timestamp': ['Date', 'Time']},
            date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M:%S')
        )
        
        logger.info(f"Données chargées: {self.df.shape[0]:,} lignes, "
                   f"{self.df.shape[1]} colonnes")
        
        return self.df
    
    def get_data_info(self) -> Dict:
        """Statistiques descriptives du dataset."""
        if self.df is None:
            raise ValueError("Données non chargées. Appelez load_data() d'abord.")
        
        info = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'date_range': (self.df['timestamp'].min(), 
                          self.df['timestamp'].max()),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'dtypes': self.df.dtypes.to_dict()
        }
        
        return info
    
    def print_summary(self):
        """Affiche un résumé du dataset."""
        info = self.get_data_info()
        
        print("=" * 70)
        print("RÉSUMÉ DU DATASET - SMART HOME ENERGY DATA")
        print("=" * 70)
        print(f" Nombre total d'enregistrements : {info['total_rows']:,}")
        print("-" * 70)
        print(f" Nombre de variables           : {info['total_columns']}")
        print("-" * 70)
        print(f" Période couverte              : {info['date_range'][0]} → {info['date_range'][1]}")
        print("-" * 70)
        print(f" Utilisation mémoire           : {info['memory_usage_mb']:.2f} MB")
        print("-" * 70)
        print(f" Valeurs manquantes totales    : {sum(info['missing_values'].values()):,}")
        print("-" * 70)
        print(f" Doublons détectés             : {info['duplicates']:,}")
        print("=" * 70)


# Charger les données
loader = EnergyDataLoader('./data/raw/household_power_consumption.txt')
df_raw = loader.load_data()
loader.print_summary()

# Afficher les premières lignes
print("\n📋 Aperçu des données:")
print(df_raw.head())