import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class EnergyEDA:
    """
    Analyse exploratoire des données énergétiques résidentielles.
    
    Méthodologie inspirée de:
    [4] UK-DALE Dataset - Unified Kingdom Domestic Appliance-Level Electricity
    [5] REFIT Dataset - Residential Energy Footprint in Time
    
    Analyses effectuées:
    - Distribution des variables énergétiques
    - Patterns temporels (journalier, hebdomadaire, saisonnier)
    - Corrélations entre variables
    - Détection d'anomalies
    - Profils de consommation
    """
    
    def __init__(self, df: pd.DataFrame, timestamp_col: str = 'timestamp'):
        self.df = df.copy()
        self.timestamp_col = timestamp_col
        self.results = {}
        
        # Vérifier timestamp
        if not pd.api.types.is_datetime64_any_dtype(self.df[timestamp_col]):
            self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        
        # Trier par timestamp
        self.df = self.df.sort_values(timestamp_col).reset_index(drop=True)
        
        logger.info(f"EDA initialise: {len(self.df):,} observations")
    
    def statistical_summary(self) -> pd.DataFrame:
        """
        Statistiques descriptives complètes.
        
        Returns:
            DataFrame avec statistiques détaillées
        """
        logger.info("Calcul des statistiques descriptives...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        summary = pd.DataFrame({
            'count': self.df[numeric_cols].count(),
            'mean': self.df[numeric_cols].mean(),
            'std': self.df[numeric_cols].std(),
            'min': self.df[numeric_cols].min(),
            '25%': self.df[numeric_cols].quantile(0.25),
            '50%': self.df[numeric_cols].quantile(0.50),
            '75%': self.df[numeric_cols].quantile(0.75),
            'max': self.df[numeric_cols].max(),
            'skewness': self.df[numeric_cols].skew(),
            'kurtosis': self.df[numeric_cols].kurtosis()
        })
        
        self.results['statistical_summary'] = summary
        
        return summary
    
    def distribution_analysis(self, 
                            columns: List[str] = None,
                            save_path: str = None):
        """
        Analyse de la distribution des variables.
        
        Selon [6] Zhang & Chen (2022), l'analyse de distribution permet 
        d'identifier les transformations nécessaires pour la modélisation. 
        Les données énergétiques suivent souvent des distributions 
        log-normales ou gamma.
        
        Args:
            columns: Variables à analyser
            save_path: Chemin pour sauvegarder les graphiques
        """
        logger.info("Analyse de distribution des variables...")
        
        if columns is None:
            columns = ['Global_active_power', 'Voltage', 'Global_intensity']
        
        fig, axes = plt.subplots(len(columns), 2, figsize=(14, 4*len(columns)))
        
        if len(columns) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(columns):
            if col not in self.df.columns:
                logger.warning(f"Colonne {col} non trouvee, ignoree")
                continue
            
            data = self.df[col].dropna()
            
            # Histogramme
            axes[idx, 0].hist(data, bins=50, edgecolor='black', alpha=0.7)
            axes[idx, 0].set_xlabel(col)
            axes[idx, 0].set_ylabel('Frequence')
            axes[idx, 0].set_title(f'Distribution de {col}')
            axes[idx, 0].axvline(data.mean(), color='red', linestyle='--', label=f'Moyenne: {data.mean():.2f}')
            axes[idx, 0].axvline(data.median(), color='green', linestyle='--', label=f'Mediane: {data.median():.2f}')
            axes[idx, 0].legend()
            
            # Q-Q plot pour normalité
            stats.probplot(data, dist="norm", plot=axes[idx, 1])
            axes[idx, 1].set_title(f'Q-Q Plot - {col}')
            
            # Test de normalité Shapiro-Wilk (sur échantillon)
            sample = data.sample(min(5000, len(data)))
            stat, p_value = stats.shapiro(sample)
            
            axes[idx, 1].text(
                0.05, 0.95, 
                f'Shapiro-Wilk p-value: {p_value:.4f}\nNormalite: {"Non" if p_value < 0.05 else "Oui"}',
                transform=axes[idx, 1].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique sauvegarde: {save_path}")
        
        plt.show()
    
    def temporal_patterns_daily(self, save_path: str = None):
        """
        Analyse des patterns journaliers de consommation.
        
        Selon [2] Wang et al. (2023), les patterns journaliers montrent 
        typiquement 3 pics de consommation:
        - Matin (7-9h): Préparation petit-déjeuner, douches
        - Midi (12-14h): Préparation déjeuner
        - Soir (18-22h): Retour domicile, préparation dîner, loisirs
        """
        logger.info("Analyse des patterns journaliers...")
        
        # Extraire l'heure
        self.df['hour'] = self.df[self.timestamp_col].dt.hour
        
        # Agrégation par heure
        hourly_pattern = self.df.groupby('hour')['Global_active_power'].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Pattern moyen avec bande de confiance
        axes[0].plot(hourly_pattern['hour'], hourly_pattern['mean'], 
                    linewidth=2, label='Moyenne')
        axes[0].fill_between(
            hourly_pattern['hour'],
            hourly_pattern['mean'] - hourly_pattern['std'],
            hourly_pattern['mean'] + hourly_pattern['std'],
            alpha=0.3, label='Ecart-type'
        )
        axes[0].set_xlabel('Heure de la journee')
        axes[0].set_ylabel('Puissance active (kW)')
        axes[0].set_title('Pattern Journalier Moyen de Consommation Energetique')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(range(0, 24))
        
        # Identifier les heures de pointe
        peak_hour = hourly_pattern.loc[hourly_pattern['mean'].idxmax(), 'hour']
        low_hour = hourly_pattern.loc[hourly_pattern['mean'].idxmin(), 'hour']
        axes[0].axvline(peak_hour, color='red', linestyle='--', 
                       label=f'Pic: {peak_hour}h ({hourly_pattern["mean"].max():.2f} kW)')
        axes[0].axvline(low_hour, color='green', linestyle='--',
                       label=f'Creux: {low_hour}h ({hourly_pattern["mean"].min():.2f} kW)')
        axes[0].legend()
        
        # Boxplot par heure
        self.df.boxplot(column='Global_active_power', by='hour', ax=axes[1])
        axes[1].set_xlabel('Heure de la journee')
        axes[1].set_ylabel('Puissance active (kW)')
        axes[1].set_title('Distribution de Consommation par Heure')
        axes[1].get_figure().suptitle('')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique sauvegarde: {save_path}")
        
        plt.show()
        
        # Stocker résultats
        self.results['hourly_pattern'] = hourly_pattern
        self.results['peak_hour'] = int(peak_hour)
        self.results['low_hour'] = int(low_hour)
        
        logger.info(f">> Heure de pic: {peak_hour}h")
        logger.info(f">> Heure creuse: {low_hour}h")
    
    def temporal_patterns_weekly(self, save_path: str = None):
        """
        Analyse des patterns hebdomadaires.
        
        Hypothèse selon [4] UK-DALE: Consommation différente entre 
        jours de semaine (routine travail) et weekend (présence accrue).
        """
        logger.info("Analyse des patterns hebdomadaires...")
        
        # Extraire jour de la semaine
        self.df['day_of_week'] = self.df[self.timestamp_col].dt.dayofweek
        day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        
        # Agrégation par jour
        weekly_pattern = self.df.groupby('day_of_week')['Global_active_power'].agg([
            'mean', 'std'
        ]).reset_index()
        weekly_pattern['day_name'] = [day_names[i] for i in weekly_pattern['day_of_week']]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(weekly_pattern['day_name'], weekly_pattern['mean'], 
               yerr=weekly_pattern['std'], capsize=5, alpha=0.7)
        ax.set_xlabel('Jour de la semaine')
        ax.set_ylabel('Puissance active moyenne (kW)')
        ax.set_title('Pattern Hebdomadaire de Consommation Energetique')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Moyenne semaine vs weekend
        weekday_mean = weekly_pattern[weekly_pattern['day_of_week'] < 5]['mean'].mean()
        weekend_mean = weekly_pattern[weekly_pattern['day_of_week'] >= 5]['mean'].mean()
        
        ax.axhline(weekday_mean, color='blue', linestyle='--', 
                  label=f'Moyenne semaine: {weekday_mean:.2f} kW')
        ax.axhline(weekend_mean, color='red', linestyle='--',
                  label=f'Moyenne weekend: {weekend_mean:.2f} kW')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique sauvegarde: {save_path}")
        
        plt.show()
        
        # Test statistique semaine vs weekend
        weekday_data = self.df[self.df['day_of_week'] < 5]['Global_active_power'].dropna()
        weekend_data = self.df[self.df['day_of_week'] >= 5]['Global_active_power'].dropna()
        
        t_stat, p_value = stats.ttest_ind(weekday_data, weekend_data)
        
        self.results['weekly_pattern'] = weekly_pattern
        self.results['weekday_vs_weekend'] = {
            'weekday_mean': weekday_mean,
            'weekend_mean': weekend_mean,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_difference': p_value < 0.05
        }
        
        logger.info(f">> Moyenne semaine: {weekday_mean:.2f} kW")
        logger.info(f">> Moyenne weekend: {weekend_mean:.2f} kW")
        logger.info(f">> Difference significative: {'Oui' if p_value < 0.05 else 'Non'} (p={p_value:.4f})")
    
    def correlation_analysis(self, save_path: str = None):
        """
        Analyse de corrélation entre variables.
        
        Selon [3] Mocanu et al. (2023), l'analyse de corrélation permet 
        d'identifier les relations linéaires entre variables et de 
        détecter la multicolinéarité pour la sélection de features.
        """
        logger.info("Analyse de correlation...")
        
        # Sélectionner colonnes numériques pertinentes
        energy_cols = [
            'Global_active_power', 'Global_reactive_power', 
            'Voltage', 'Global_intensity',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
        ]
        
        available_cols = [col for col in energy_cols if col in self.df.columns]
        
        # Matrice de corrélation
        corr_matrix = self.df[available_cols].corr()
        
        # Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        ax.set_title('Matrice de Correlation des Variables Energetiques')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique sauvegarde: {save_path}")
        
        plt.show()
        
        # Identifier corrélations fortes
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        self.results['correlation_matrix'] = corr_matrix
        self.results['high_correlations'] = high_corr
        
        logger.info(f">> Correlations fortes detectees: {len(high_corr)}")
        for corr in high_corr:
            logger.info(f"   {corr['var1']} <-> {corr['var2']}: {corr['correlation']:.3f}")
    
    def submetering_analysis(self, save_path: str = None):
        """
        Analyse de la consommation par sous-compteur.
        
        Les sous-compteurs permettent de décomposer la consommation totale:
        - Sub_metering_1: Cuisine (four, micro-ondes, lave-vaisselle)
        - Sub_metering_2: Buanderie (lave-linge, sèche-linge)
        - Sub_metering_3: Chauffage électrique et climatisation
        - Reste: Autres appareils (éclairage, TV, ordinateurs, etc.)
        """
        logger.info("Analyse des sous-compteurs...")
        
        submeter_cols = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        
        if not all(col in self.df.columns for col in submeter_cols):
            logger.warning("Colonnes de sous-compteurs manquantes")
            return
        
        # Consommation totale mesurée vs sous-compteurs
        self.df['total_submeters'] = self.df[submeter_cols].sum(axis=1) / 1000  # Wh -> kWh
        self.df['other_consumption'] = self.df['Global_active_power'] - self.df['total_submeters']
        self.df['other_consumption'] = self.df['other_consumption'].clip(lower=0)
        
        # Statistiques
        submeter_stats = {
            'Cuisine (Sub 1)': self.df['Sub_metering_1'].sum() / 1000,
            'Buanderie (Sub 2)': self.df['Sub_metering_2'].sum() / 1000,
            'Chauffage/Clim (Sub 3)': self.df['Sub_metering_3'].sum() / 1000,
            'Autres': self.df['other_consumption'].sum()
        }
        
        # Graphiques
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        axes[0].pie(
            submeter_stats.values(), 
            labels=submeter_stats.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        axes[0].set_title('Repartition de la Consommation par Categorie')
        
        # Bar chart
        axes[1].bar(submeter_stats.keys(), submeter_stats.values(), color=colors)
        axes[1].set_ylabel('Consommation totale (kWh)')
        axes[1].set_title('Consommation par Sous-Compteur')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique sauvegarde: {save_path}")
        
        plt.show()
        
        # Calculer pourcentages
        total_consumption = sum(submeter_stats.values())
        submeter_percentages = {
            k: (v / total_consumption * 100) for k, v in submeter_stats.items()
        }
        
        self.results['submeter_consumption'] = submeter_stats
        self.results['submeter_percentages'] = submeter_percentages
        
        logger.info(">> Repartition de la consommation:")
        for name, pct in submeter_percentages.items():
            logger.info(f"   {name}: {pct:.1f}%")
    
    def generate_full_report(self, output_dir: str = './results/eda'):
        """
        Génère un rapport EDA complet avec tous les graphiques.
        
        Args:
            output_dir: Répertoire de sauvegarde
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("GENERATION DU RAPPORT EDA COMPLET")
        logger.info("=" * 70)
        
        # Statistiques descriptives
        summary = self.statistical_summary()
        summary.to_csv(f'{output_dir}/statistical_summary.csv')
        logger.info(f">> Statistiques sauvegardees")
        
        # Distributions
        self.distribution_analysis(
            columns=['Global_active_power', 'Voltage', 'Global_intensity'],
            save_path=f'{output_dir}/distributions.png'
        )
        
        # Patterns temporels
        self.temporal_patterns_daily(save_path=f'{output_dir}/daily_pattern.png')
        self.temporal_patterns_weekly(save_path=f'{output_dir}/weekly_pattern.png')
        
        # Corrélations
        self.correlation_analysis(save_path=f'{output_dir}/correlations.png')
        
        # Sous-compteurs
        self.submetering_analysis(save_path=f'{output_dir}/submetering.png')
        
        logger.info("=" * 70)
        logger.info("RAPPORT EDA COMPLETE")
        logger.info(f"Fichiers sauvegardes dans: {output_dir}")
        logger.info("=" * 70)
        
        return self.results


# ========================================================================
# SCRIPT DE TEST
# ========================================================================

if __name__ == "__main__":
    # Charger données
    print("[INFO] Chargement des donnees...")
    df = pd.read_csv('./data/processed/data_cleaned.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Initialiser EDA
    eda = EnergyEDA(df, timestamp_col='timestamp')
    
    # Générer rapport complet
    results = eda.generate_full_report(output_dir='./results/eda')
    
    print("\n[SUCCESS] Analyse exploratoire terminee!")
    print(f"Resultats disponibles dans: ./results/eda/")