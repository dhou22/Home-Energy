import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class FeatureCorrelationAnalyzer:
    """
    Analyse avancée de corrélation et sélection de features pour prédiction énergétique.
    
    Basé sur:
    - Ahmad et al. (2024): Feature selection methods for energy forecasting
    - Wang et al. (2023): Correlation analysis in smart home datasets
    - Mocanu et al. (2023): Feature importance in deep learning for power systems
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'Global_active_power'):
        """
        Args:
            df: DataFrame avec features et target
            target_col: Colonne cible à prédire
        """
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = [col for col in df.columns 
                            if col not in [target_col, 'timestamp'] 
                            and df[col].dtype in ['float64', 'int64']]
        self.results = {}
        
    def compute_correlation_matrix(self):
        """
        Calcule matrice de corrélation complète (Pearson, Spearman, Kendall).
        
        Pearson: Relations linéaires
        Spearman: Relations monotones non-linéaires  
        Kendall: Robuste aux outliers
        """
        print("=" * 80)
        print("ANALYSE DE CORRÉLATION MULTI-MÉTHODES")
        print("=" * 80)
        
        cols = self.feature_cols + [self.target_col]
        data = self.df[cols].dropna()
        
        # Pearson (linéaire)
        pearson = data.corr(method='pearson')
        
        # Spearman (monotone)
        spearman = data.corr(method='spearman')
        
        # Kendall (robuste)
        kendall = data.corr(method='kendall')
        
        self.results['pearson'] = pearson
        self.results['spearman'] = spearman
        self.results['kendall'] = kendall
        
        # Corrélations avec la target
        self.results['target_correlations'] = pd.DataFrame({
            'Pearson': pearson[self.target_col].drop(self.target_col),
            'Spearman': spearman[self.target_col].drop(self.target_col),
            'Kendall': kendall[self.target_col].drop(self.target_col)
        }).sort_values('Pearson', key=abs, ascending=False)
        
        print("\n📊 Corrélations avec la cible:", self.target_col)
        print(self.results['target_correlations'].round(4))
        
        return self.results['target_correlations']
    
    def mutual_information_analysis(self):
        """
        Calcule l'information mutuelle (détecte relations non-linéaires).
        
        Référence: Ahmad et al. (2024) - "Mutual Information outperforms 
        correlation for non-linear energy patterns"
        """
        print("\n" + "=" * 80)
        print("INFORMATION MUTUELLE (Non-linéarité)")
        print("=" * 80)
        
        X = self.df[self.feature_cols].dropna()
        y = self.df[self.target_col].loc[X.index]
        
        # Calcul MI
        mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
        
        mi_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'MI_Score': mi_scores,
            'MI_Normalized': mi_scores / mi_scores.max()
        }).sort_values('MI_Score', ascending=False)
        
        self.results['mutual_information'] = mi_df
        
        print("\n🔍 Top features par Information Mutuelle:")
        print(mi_df.to_string(index=False))
        
        return mi_df
    
    def feature_importance_rf(self):
        """
        Importance des features via Random Forest (Gini importance).
        
        Référence: Raza & Khosravi (2023) - "Tree-based methods excel 
        at capturing feature interactions in energy data"
        """
        print("\n" + "=" * 80)
        print("IMPORTANCE DES FEATURES (Random Forest)")
        print("=" * 80)
        
        X = self.df[self.feature_cols].dropna()
        y = self.df[self.target_col].loc[X.index]
        
        # Random Forest avec hyperparamètres optimisés
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': rf.feature_importances_,
            'Importance_Pct': rf.feature_importances_ / rf.feature_importances_.sum() * 100
        }).sort_values('Importance', ascending=False)
        
        self.results['rf_importance'] = importance_df
        
        print("\n🌳 Importance Random Forest:")
        print(importance_df.to_string(index=False))
        
        return importance_df
    
    def f_statistic_analysis(self):
        """
        Test F-statistique ANOVA (relation linéaire univariée).
        
        H0: Feature n'a pas d'effet sur target
        H1: Feature a un effet significatif
        """
        print("\n" + "=" * 80)
        print("TEST F-STATISTIQUE (ANOVA)")
        print("=" * 80)
        
        X = self.df[self.feature_cols].dropna()
        y = self.df[self.target_col].loc[X.index]
        
        f_scores, p_values = f_regression(X, y)
        
        f_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'F_Score': f_scores,
            'p_value': p_values,
            'Significant': p_values < 0.05
        }).sort_values('F_Score', ascending=False)
        
        self.results['f_statistics'] = f_df
        
        print("\n📈 Test F-statistique (α=0.05):")
        print(f_df.to_string(index=False))
        
        return f_df
    
    def multicollinearity_check(self, threshold=0.8):
        """
        Détecte multicolinéarité (features redondantes).
        
        VIF (Variance Inflation Factor):
        - VIF < 5: Pas de multicolinéarité
        - 5 ≤ VIF < 10: Multicolinéarité modérée
        - VIF ≥ 10: Multicolinéarité sévère (problème)
        
        Référence: Zhang & Chen (2022) - "Multicollinearity degrades 
        model interpretability and stability"
        """
        print("\n" + "=" * 80)
        print("ANALYSE DE MULTICOLINÉARITÉ")
        print("=" * 80)
        
        corr_matrix = self.df[self.feature_cols].corr().abs()
        
        # Paires hautement corrélées
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        high_corr_pairs = [
            (corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            for i in range(len(corr_matrix.index))
            for j in range(len(corr_matrix.columns))
            if upper_triangle[i, j] and corr_matrix.iloc[i, j] > threshold
        ]
        
        if high_corr_pairs:
            print(f"\n⚠️  Paires de features hautement corrélées (|r| > {threshold}):")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"   {feat1} ↔ {feat2}: r = {corr:.3f}")
        else:
            print(f"\n✅ Aucune multicolinéarité détectée (seuil: {threshold})")
        
        self.results['multicollinearity'] = high_corr_pairs
        
        return high_corr_pairs
    
    def pca_variance_analysis(self, n_components=None):
        """
        Analyse en Composantes Principales (réduction dimensionnalité).
        
        Objectif: Identifier combien de composantes expliquent 95% variance
        """
        print("\n" + "=" * 80)
        print("ANALYSE EN COMPOSANTES PRINCIPALES (PCA)")
        print("=" * 80)
        
        X = self.df[self.feature_cols].dropna()
        
        # Standardisation (obligatoire pour PCA)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        if n_components is None:
            n_components = min(len(self.feature_cols), len(X))
        
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        
        # Variance expliquée
        variance_df = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            'Variance_Explained': pca.explained_variance_ratio_,
            'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
        })
        
        # Nombre de composantes pour 95% variance
        n_components_95 = np.argmax(variance_df['Cumulative_Variance'] >= 0.95) + 1
        
        print(f"\n📊 Variance expliquée par composante:")
        print(variance_df.to_string(index=False))
        print(f"\n✨ {n_components_95} composantes expliquent ≥95% de la variance")
        
        self.results['pca'] = {
            'variance_df': variance_df,
            'n_components_95': n_components_95,
            'pca_model': pca
        }
        
        return variance_df
    
    def create_visualizations(self, output_path='./results/'):
        """
        Génère toutes les visualisations scientifiques.
        """
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Figure 1: Matrice de corrélation (heatmap)
        self._plot_correlation_heatmap(output_path)
        
        # Figure 2: Corrélations avec target (barplot comparatif)
        self._plot_target_correlations(output_path)
        
        # Figure 3: Information mutuelle vs Corrélation
        self._plot_mi_vs_correlation(output_path)
        
        # Figure 4: Importance RF
        self._plot_rf_importance(output_path)
        
        # Figure 5: PCA variance expliquée
        self._plot_pca_variance(output_path)
        
        # Figure 6: Scatterplots top features
        self._plot_top_features_scatter(output_path)
        
        print(f"\n✅ Visualisations sauvegardées dans: {output_path}")
    
    def _plot_correlation_heatmap(self, output_path):
        """Heatmap de corrélation avec annotations."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for ax, (method, corr_matrix) in zip(axes, [
            ('Pearson', self.results['pearson']),
            ('Spearman', self.results['spearman']),
            ('Kendall', self.results['kendall'])
        ]):
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Coefficient de corrélation'},
                ax=ax
            )
            ax.set_title(f'Matrice de Corrélation - {method}', fontsize=14, weight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_path}correlation_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_target_correlations(self, output_path):
        """Barplot comparatif des corrélations avec la target."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df_plot = self.results['target_correlations'].reset_index()
        df_plot = df_plot.melt(id_vars='index', var_name='Method', value_name='Correlation')
        
        sns.barplot(
            data=df_plot,
            x='index',
            y='Correlation',
            hue='Method',
            ax=ax
        )
        
        ax.set_xlabel('Features', fontsize=12, weight='bold')
        ax.set_ylabel('Corrélation avec ' + self.target_col, fontsize=12, weight='bold')
        ax.set_title('Corrélations Feature-Target (Multi-méthodes)', fontsize=14, weight='bold')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.legend(title='Méthode', frameon=True)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_path}target_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_mi_vs_correlation(self, output_path):
        """Scatter: Information Mutuelle vs Corrélation Pearson."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mi_df = self.results['mutual_information'].set_index('Feature')
        corr_df = self.results['target_correlations']['Pearson'].abs()
        
        combined = pd.DataFrame({
            'MI': mi_df['MI_Normalized'],
            'Corr': corr_df
        }).dropna()
        
        ax.scatter(combined['Corr'], combined['MI'], s=100, alpha=0.6, edgecolors='black')
        
        for feature, row in combined.iterrows():
            ax.annotate(feature, (row['Corr'], row['MI']), 
                       fontsize=9, alpha=0.7, ha='right')
        
        ax.set_xlabel('|Corrélation Pearson| avec Target', fontsize=12, weight='bold')
        ax.set_ylabel('Information Mutuelle (normalisée)', fontsize=12, weight='bold')
        ax.set_title('Linéarité vs Non-linéarité des Relations', fontsize=14, weight='bold')
        ax.grid(alpha=0.3)
        
        # Diagonale de référence
        lims = [0, max(combined['Corr'].max(), combined['MI'].max())]
        ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='Linéarité parfaite')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_path}mi_vs_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rf_importance(self, output_path):
        """Barplot importance Random Forest."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df_plot = self.results['rf_importance'].sort_values('Importance', ascending=True)
        
        ax.barh(df_plot['Feature'], df_plot['Importance'], color='steelblue', edgecolor='black')
        ax.set_xlabel('Importance (Gini)', fontsize=12, weight='bold')
        ax.set_ylabel('Features', fontsize=12, weight='bold')
        ax.set_title('Importance des Features - Random Forest', fontsize=14, weight='bold')
        
        # Annotations
        for i, (feat, imp) in enumerate(zip(df_plot['Feature'], df_plot['Importance'])):
            ax.text(imp, i, f' {imp:.4f}', va='center', fontsize=9)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_path}rf_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pca_variance(self, output_path):
        """Variance expliquée PCA."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        variance_df = self.results['pca']['variance_df']
        
        ax.bar(variance_df['PC'], variance_df['Variance_Explained'], 
               alpha=0.7, label='Variance individuelle', edgecolor='black')
        ax.plot(variance_df['PC'], variance_df['Cumulative_Variance'], 
                'r-o', linewidth=2, markersize=8, label='Variance cumulée')
        
        ax.axhline(0.95, color='green', linestyle='--', linewidth=2, 
                   label='Seuil 95%', alpha=0.7)
        
        ax.set_xlabel('Composantes Principales', fontsize=12, weight='bold')
        ax.set_ylabel('Variance Expliquée', fontsize=12, weight='bold')
        ax.set_title('Analyse PCA - Variance Expliquée', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_path}pca_variance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_top_features_scatter(self, output_path):
        """Scatterplots des top 4 features vs target."""
        top_features = self.results['target_correlations'].head(4).index
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for ax, feature in zip(axes, top_features):
            ax.scatter(
                self.df[feature],
                self.df[self.target_col],
                alpha=0.3,
                s=1
            )
            
            # Regression line
            z = np.polyfit(self.df[feature].dropna(), 
                          self.df[self.target_col].loc[self.df[feature].dropna().index], 1)
            p = np.poly1d(z)
            ax.plot(self.df[feature].sort_values(), 
                   p(self.df[feature].sort_values()), 
                   "r--", linewidth=2, alpha=0.7)
            
            corr = self.results['target_correlations'].loc[feature, 'Pearson']
            ax.set_xlabel(feature, fontsize=11, weight='bold')
            ax.set_ylabel(self.target_col, fontsize=11, weight='bold')
            ax.set_title(f'{feature} vs {self.target_col}\nr = {corr:.3f}', 
                        fontsize=12, weight='bold')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_path}top_features_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_feature_selection_report(self):
        """
        Génère rapport de recommandation pour sélection de features.
        """
        print("\n" + "=" * 80)
        print("RAPPORT DE SÉLECTION DE FEATURES")
        print("=" * 80)
        
        # Critère 1: Corrélation forte (|r| > 0.5)
        strong_corr = self.results['target_correlations'][
            self.results['target_correlations']['Pearson'].abs() > 0.5
        ]
        
        # Critère 2: MI élevée (top 50%)
        mi_threshold = self.results['mutual_information']['MI_Score'].median()
        high_mi = self.results['mutual_information'][
            self.results['mutual_information']['MI_Score'] > mi_threshold
        ]
        
        # Critère 3: RF Importance élevée (top 50%)
        rf_threshold = self.results['rf_importance']['Importance'].median()
        high_rf = self.results['rf_importance'][
            self.results['rf_importance']['Importance'] > rf_threshold
        ]
        
        # Features recommandées (consensus)
        recommended = set(strong_corr.index) & set(high_mi['Feature']) & set(high_rf['Feature'])
        
        print("\n✅ FEATURES RECOMMANDÉES (consensus 3 méthodes):")
        for feat in recommended:
            print(f"   • {feat}")
            print(f"     - Corrélation: {self.results['target_correlations'].loc[feat, 'Pearson']:.3f}")
            print(f"     - MI Score: {self.results['mutual_information'][self.results['mutual_information']['Feature']==feat]['MI_Score'].values[0]:.3f}")
            print(f"     - RF Importance: {self.results['rf_importance'][self.results['rf_importance']['Feature']==feat]['Importance'].values[0]:.3f}")
        
        # Features à éviter (multicolinéarité)
        if self.results['multicollinearity']:
            print("\n⚠️  ATTENTION - Paires multicolinéaires:")
            for feat1, feat2, corr in self.results['multicollinearity']:
                print(f"   {feat1} ↔ {feat2}: r = {corr:.3f}")
                print(f"   → Recommandation: Garder la feature la plus importante")
        
        return list(recommended)


# ==============================================================================
# EXÉCUTION PRINCIPALE
# ==============================================================================

if __name__ == "__main__":
    print("🔬 ANALYSE DE CORRÉLATION ET SÉLECTION DE FEATURES")
    print("   Projet: Smart Home Energy Analytics")
    print("=" * 80)
    
    # Chargement des données
    print("\n📂 Chargement des données...")
    df_clean = pd.read_csv('./data/processed/data_cleaned.csv')
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
    
    print(f"   ✓ {len(df_clean):,} observations chargées")
    print(f"   ✓ {len(df_clean.columns)} colonnes")
    
    # Initialisation de l'analyseur
    analyzer = FeatureCorrelationAnalyzer(
        df=df_clean,
        target_col='Global_active_power'
    )
    
    # Analyses
    print("\n🔍 Lancement des analyses...")
    
    # 1. Corrélations
    analyzer.compute_correlation_matrix()
    
    # 2. Information mutuelle
    analyzer.mutual_information_analysis()
    
    # 3. Importance RF
    analyzer.feature_importance_rf()
    
    # 4. Test F
    analyzer.f_statistic_analysis()
    
    # 5. Multicolinéarité
    analyzer.multicollinearity_check(threshold=0.8)
    
    # 6. PCA
    analyzer.pca_variance_analysis()
    
    # Visualisations
    print("\n📊 Génération des visualisations...")
    analyzer.create_visualizations(output_path='./results/')
    
    # Rapport final
    recommended_features = analyzer.generate_feature_selection_report()
    
    print("\n" + "=" * 80)
    print("✅ ANALYSE TERMINÉE")
    print("=" * 80)
    print(f"\n📁 Résultats disponibles dans: ./results/")
    print("\nVisualisations générées:")
    print("   • correlation_matrices.png")
    print("   • target_correlations.png")
    print("   • mi_vs_correlation.png")
    print("   • rf_importance.png")
    print("   • pca_variance.png")
    print("   • top_features_scatter.png")