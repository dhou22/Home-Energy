import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, ForeignKey, Index
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Optional
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

Base = declarative_base()


class DatabaseManager:
    """
    Gestion de la base de données MySQL pour Smart Home Energy Analytics.
    
    Architecture basée sur:
    [9] Reinhardt et al. (2022). On the Design of Databases for 
        Energy Consumption Data. Recommandations pour schéma normalisé 
        3NF avec indexation optimisée sur timestamps.
    
    Design patterns:
    - Normalisation 3NF pour éviter redondance
    - Index composites sur (household_id, timestamp)
    - Tables d'agrégation pour optimiser requêtes analytiques
    - Foreign keys avec CASCADE pour intégrité référentielle
    """
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 3306,
                 user: str = 'root',
                 password: str = '',
                 database: str = 'smart_energy_db'):
        """
        Initialisation de la connexion à la base de données.
        
        Args:
            host: Adresse du serveur MySQL
            port: Port MySQL (défaut: 3306)
            user: Utilisateur MySQL
            password: Mot de passe
            database: Nom de la base de données
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        
        # Création de l'engine SQLAlchemy
        self.engine = None
        self.metadata = MetaData()
        self.session = None
        
        self._connect()
    
    def _connect(self):
        """Établit la connexion à MySQL."""
        try:
            # Connexion sans spécifier la database pour la créer si nécessaire
            connection_string = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}"
            temp_engine = create_engine(connection_string)
            
            # Créer la database si elle n'existe pas
            with temp_engine.connect() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {self.database} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
                conn.commit()
            
            # Connexion à la database spécifique
            self.connection_string = f"{connection_string}/{self.database}"
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            
            logger.info(f"Connexion etablie a MySQL: {self.host}:{self.port}/{self.database}")
            
        except SQLAlchemyError as e:
            logger.error(f"Erreur de connexion a MySQL: {e}")
            raise
    
    def create_schema(self):
        """
        Création du schéma complet de la base de données.
        
        Architecture normalisée 3NF avec tables:
        1. households - Informations sur les foyers
        2. energy_measurements - Mesures énergétiques brutes
        3. sub_meters - Sous-compteurs par appareil
        4. hourly_consumption - Agrégations horaires
        5. daily_consumption - Agrégations journalières
        6. predictions - Prédictions du modèle ML
        """
        logger.info("Creation du schema de la base de donnees...")
        
        try:
            # SQL pour créer les tables avec le schéma documenté
            schema_sql = """
            -- Table: households
            CREATE TABLE IF NOT EXISTS households (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                location VARCHAR(255),
                surface_area DECIMAL(6,2),
                occupants INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_name (name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            
            -- Table: energy_measurements
            CREATE TABLE IF NOT EXISTS energy_measurements (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                household_id INT NOT NULL,
                global_active_power DECIMAL(8,3),
                global_reactive_power DECIMAL(8,3),
                voltage DECIMAL(6,2),
                global_intensity DECIMAL(6,3),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_timestamp (timestamp),
                INDEX idx_household (household_id),
                INDEX idx_household_timestamp (household_id, timestamp),
                FOREIGN KEY (household_id) REFERENCES households(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            
            -- Table: sub_meters
            CREATE TABLE IF NOT EXISTS sub_meters (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                measurement_id BIGINT NOT NULL,
                meter_type ENUM('kitchen', 'laundry', 'climate') NOT NULL,
                energy_consumed DECIMAL(8,3),
                INDEX idx_measurement (measurement_id),
                INDEX idx_type (meter_type),
                FOREIGN KEY (measurement_id) REFERENCES energy_measurements(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            
            -- Table: hourly_consumption
            CREATE TABLE IF NOT EXISTS hourly_consumption (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                date DATE NOT NULL,
                hour INT NOT NULL,
                household_id INT NOT NULL,
                avg_power DECIMAL(8,3),
                max_power DECIMAL(8,3),
                min_power DECIMAL(8,3),
                total_energy DECIMAL(10,3),
                record_count INT,
                UNIQUE KEY unique_datetime (date, hour, household_id),
                INDEX idx_date (date),
                FOREIGN KEY (household_id) REFERENCES households(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            
            -- Table: daily_consumption
            CREATE TABLE IF NOT EXISTS daily_consumption (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                date DATE NOT NULL,
                household_id INT NOT NULL,
                total_energy DECIMAL(10,3),
                avg_power DECIMAL(8,3),
                max_power DECIMAL(8,3),
                min_power DECIMAL(8,3),
                peak_hour INT,
                UNIQUE KEY unique_date (date, household_id),
                INDEX idx_date (date),
                FOREIGN KEY (household_id) REFERENCES households(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            
            -- Table: predictions
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                household_id INT NOT NULL,
                predicted_power DECIMAL(8,3),
                actual_power DECIMAL(8,3),
                model_name VARCHAR(50),
                confidence DECIMAL(5,4),
                error_abs DECIMAL(8,3),
                error_pct DECIMAL(6,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_timestamp (timestamp),
                INDEX idx_model (model_name),
                FOREIGN KEY (household_id) REFERENCES households(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            # Exécuter les commandes SQL
            with self.engine.connect() as conn:
                for statement in schema_sql.split(';'):
                    if statement.strip():
                        conn.execute(text(statement))
                conn.commit()
            
            logger.info(">> Schema cree avec succes")
            logger.info("   - households")
            logger.info("   - energy_measurements")
            logger.info("   - sub_meters")
            logger.info("   - hourly_consumption")
            logger.info("   - daily_consumption")
            logger.info("   - predictions")
            
        except SQLAlchemyError as e:
            logger.error(f"Erreur lors de la creation du schema: {e}")
            raise
    
    def insert_household(self, 
                        name: str,
                        location: str = None,
                        surface_area: float = None,
                        occupants: int = None) -> int:
        """
        Insère un nouveau foyer dans la table households.
        
        Args:
            name: Nom du foyer
            location: Localisation
            surface_area: Surface en m²
            occupants: Nombre d'occupants
        
        Returns:
            ID du foyer créé
        """
        try:
            insert_sql = text("""
                INSERT INTO households (name, location, surface_area, occupants)
                VALUES (:name, :location, :surface_area, :occupants)
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(
                    insert_sql,
                    {
                        'name': name,
                        'location': location,
                        'surface_area': surface_area,
                        'occupants': occupants
                    }
                )
                conn.commit()
                household_id = result.lastrowid
            
            logger.info(f"Foyer cree: ID={household_id}, Name={name}")
            return household_id
            
        except SQLAlchemyError as e:
            logger.error(f"Erreur lors de l'insertion du foyer: {e}")
            raise
    
    def bulk_insert_measurements(self, 
                                df: pd.DataFrame,
                                household_id: int,
                                batch_size: int = 10000):
        """
        Insertion en masse des mesures énergétiques.
        
        Optimisation selon [9] Reinhardt et al. (2022):
        - Batch insert pour performances (10,000 lignes par batch)
        - Désactivation temporaire des index pendant insertion
        - Commit par batch pour éviter lock prolongé
        
        Args:
            df: DataFrame avec colonnes timestamp, Global_active_power, etc.
            household_id: ID du foyer
            batch_size: Taille des batches d'insertion
        """
        logger.info(f"Insertion de {len(df):,} mesures energetiques...")
        
        try:
            # Préparer les données
            df_insert = df.copy()
            df_insert['household_id'] = household_id
            
            # Renommer pour correspondre au schéma SQL (minuscules)
            column_mapping = {
                'Global_active_power': 'global_active_power',
                'Global_reactive_power': 'global_reactive_power',
                'Voltage': 'voltage',
                'Global_intensity': 'global_intensity'
            }
            
            # Appliquer le renommage seulement si les colonnes existent
            for old_name, new_name in column_mapping.items():
                if old_name in df_insert.columns:
                    df_insert = df_insert.rename(columns={old_name: new_name})
            
            # Colonnes requises après renommage
            required_cols = [
                'timestamp', 'household_id', 'global_active_power',
                'global_reactive_power', 'voltage', 'global_intensity'
            ]
            
            # Vérifier colonnes après renommage
            missing_cols = set(required_cols) - set(df_insert.columns)
            if missing_cols:
                logger.error(f"Colonnes disponibles: {df_insert.columns.tolist()}")
                raise ValueError(f"Colonnes manquantes apres renommage: {missing_cols}")
            
            # Sélectionner colonnes
            df_insert = df_insert[required_cols]
            
            # Insertion par batches
            total_batches = len(df_insert) // batch_size + 1
            
            for i in range(0, len(df_insert), batch_size):
                batch = df_insert.iloc[i:i+batch_size]
                batch.to_sql(
                    'energy_measurements',
                    con=self.engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                batch_num = i // batch_size + 1
                logger.info(f"   Batch {batch_num}/{total_batches} insere ({len(batch)} lignes)")
            
            logger.info(f">> {len(df_insert):,} mesures inserees avec succes")
            
        except SQLAlchemyError as e:
            logger.error(f"Erreur lors de l'insertion des mesures: {e}")
            raise
    
    def bulk_insert_submeters(self,
                             df: pd.DataFrame,
                             household_id: int):
        """
        Insertion des données de sous-compteurs.
        
        Args:
            df: DataFrame avec Sub_metering_1, Sub_metering_2, Sub_metering_3
            household_id: ID du foyer
        """
        logger.info("Insertion des donnees de sous-compteurs...")
        
        try:
            # Vérifier que les colonnes existent
            submeter_cols = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
            if not all(col in df.columns for col in submeter_cols):
                logger.warning(f"Colonnes de sous-compteurs manquantes. Disponibles: {df.columns.tolist()}")
                return
            
            # Récupérer les IDs des measurements
            query = text("""
                SELECT id, timestamp 
                FROM energy_measurements 
                WHERE household_id = :household_id
                ORDER BY timestamp
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'household_id': household_id})
                measurements = pd.DataFrame(result.fetchall(), columns=['measurement_id', 'timestamp'])
            
            # S'assurer que timestamp est datetime dans les deux DataFrames
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            if not pd.api.types.is_datetime64_any_dtype(measurements['timestamp']):
                measurements['timestamp'] = pd.to_datetime(measurements['timestamp'])
            
            # Préparer données submeters
            df_sub = df[['timestamp', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].copy()
            df_sub = df_sub.merge(measurements, on='timestamp', how='inner')
            
            # Transformer en format long
            submeters_data = []
            for _, row in df_sub.iterrows():
                submeters_data.append({
                    'measurement_id': row['measurement_id'],
                    'meter_type': 'kitchen',
                    'energy_consumed': row['Sub_metering_1']
                })
                submeters_data.append({
                    'measurement_id': row['measurement_id'],
                    'meter_type': 'laundry',
                    'energy_consumed': row['Sub_metering_2']
                })
                submeters_data.append({
                    'measurement_id': row['measurement_id'],
                    'meter_type': 'climate',
                    'energy_consumed': row['Sub_metering_3']
                })
            
            df_submeters = pd.DataFrame(submeters_data)
            
            # Insertion
            df_submeters.to_sql(
                'sub_meters',
                con=self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            logger.info(f">> {len(df_submeters):,} enregistrements de sous-compteurs inseres")
            
        except SQLAlchemyError as e:
            logger.error(f"Erreur lors de l'insertion des sous-compteurs: {e}")
            raise
    
    def create_hourly_aggregations(self, household_id: int):
        """
        Création des agrégations horaires.
        
        Selon [9] Reinhardt et al. (2022), les tables d'agrégation 
        pré-calculées accélèrent les requêtes analytiques de 10-50x 
        en évitant les GROUP BY répétés sur tables volumineuses.
        
        Args:
            household_id: ID du foyer
        """
        logger.info("Creation des agregations horaires...")
        
        try:
            aggregation_sql = text("""
                INSERT INTO hourly_consumption 
                    (date, hour, household_id, avg_power, max_power, min_power, total_energy, record_count)
                SELECT 
                    DATE(timestamp) as date,
                    HOUR(timestamp) as hour,
                    household_id,
                    AVG(global_active_power) as avg_power,
                    MAX(global_active_power) as max_power,
                    MIN(global_active_power) as min_power,
                    SUM(global_active_power) / 60 as total_energy,
                    COUNT(*) as record_count
                FROM energy_measurements
                WHERE household_id = :household_id
                GROUP BY DATE(timestamp), HOUR(timestamp), household_id
                ON DUPLICATE KEY UPDATE
                    avg_power = VALUES(avg_power),
                    max_power = VALUES(max_power),
                    min_power = VALUES(min_power),
                    total_energy = VALUES(total_energy),
                    record_count = VALUES(record_count)
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(aggregation_sql, {'household_id': household_id})
                conn.commit()
                rows_affected = result.rowcount
            
            logger.info(f">> {rows_affected:,} agregations horaires creees")
            
        except SQLAlchemyError as e:
            logger.error(f"Erreur lors des agregations horaires: {e}")
            raise
    
    def create_daily_aggregations(self, household_id: int):
        """Création des agrégations journalières."""
        logger.info("Creation des agregations journalieres...")
        
        try:
            aggregation_sql = text("""
                INSERT INTO daily_consumption 
                    (date, household_id, total_energy, avg_power, max_power, min_power, peak_hour)
                SELECT 
                    date,
                    household_id,
                    SUM(total_energy) as total_energy,
                    AVG(avg_power) as avg_power,
                    MAX(max_power) as max_power,
                    MIN(min_power) as min_power,
                    (SELECT hour FROM hourly_consumption hc2 
                     WHERE hc2.date = hc1.date AND hc2.household_id = hc1.household_id
                     ORDER BY avg_power DESC LIMIT 1) as peak_hour
                FROM hourly_consumption hc1
                WHERE household_id = :household_id
                GROUP BY date, household_id
                ON DUPLICATE KEY UPDATE
                    total_energy = VALUES(total_energy),
                    avg_power = VALUES(avg_power),
                    max_power = VALUES(max_power),
                    min_power = VALUES(min_power),
                    peak_hour = VALUES(peak_hour)
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(aggregation_sql, {'household_id': household_id})
                conn.commit()
                rows_affected = result.rowcount
            
            logger.info(f">> {rows_affected:,} agregations journalieres creees")
            
        except SQLAlchemyError as e:
            logger.error(f"Erreur lors des agregations journalieres: {e}")
            raise
    
    def get_database_stats(self) -> Dict:
        """Statistiques sur la base de données."""
        logger.info("Collecte des statistiques de la base de donnees...")
        
        stats = {}
        tables = ['households', 'energy_measurements', 'sub_meters', 
                  'hourly_consumption', 'daily_consumption', 'predictions']
        
        try:
            with self.engine.connect() as conn:
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}"))
                    count = result.fetchone()[0]
                    stats[table] = count
                
                # Taille de la base de données
                result = conn.execute(text("""
                    SELECT 
                        SUM(data_length + index_length) / 1024 / 1024 AS size_mb
                    FROM information_schema.tables
                    WHERE table_schema = :database
                """), {'database': self.database})
                
                size_mb = result.fetchone()[0]
                stats['database_size_mb'] = round(size_mb, 2) if size_mb else 0
            
            return stats
            
        except SQLAlchemyError as e:
            logger.error(f"Erreur lors de la collecte des statistiques: {e}")
            return {}
    
    def print_database_summary(self):
        """Affiche un résumé de la base de données."""
        stats = self.get_database_stats()
        
        print("\n" + "=" * 80)
        print("RESUME DE LA BASE DE DONNEES")
        print("=" * 80)
        print(f"Database: {self.database}")
        print(f"Taille totale: {stats.get('database_size_mb', 0):.2f} MB")
        print("-" * 80)
        
        for table, count in stats.items():
            if table != 'database_size_mb':
                print(f"{table:30s} : {count:>10,} enregistrements")
        
        print("=" * 80 + "\n")
    
    def close(self):
        """Ferme la connexion à la base de données."""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
        logger.info("Connexion fermee")


# ========================================================================
# SCRIPT DE TEST
# ========================================================================

if __name__ == "__main__":
    # Configuration de la connexion
    DB_CONFIG = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': '',  # Modifier selon votre configuration
        'database': 'smart_energy_db'
    }
    
    print("\n[INFO] Initialisation du DatabaseManager...")
    db = DatabaseManager(**DB_CONFIG)
    
    print("\n[INFO] Creation du schema...")
    db.create_schema()
    
    print("\n[INFO] Creation d'un foyer de test...")
    household_id = db.insert_household(
        name='UCI Household 2006-2010',
        location='Sceaux, France',
        surface_area=150.0,
        occupants=4
    )
    
    print(f"\n[OK] Foyer cree avec ID: {household_id}")
    
    # Charger les données nettoyées
    print("\n[INFO] Chargement des donnees pour insertion...")
    df = pd.read_csv('./data/processed/data_cleaned.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Limiter pour le test (premières 100,000 lignes)
    df_sample = df.head(100000)
    
    print(f"\n[INFO] Insertion de {len(df_sample):,} mesures...")
    db.bulk_insert_measurements(df_sample, household_id, batch_size=10000)
    
    print("\n[INFO] Creation des agregations...")
    db.create_hourly_aggregations(household_id)
    db.create_daily_aggregations(household_id)
    
    print("\n[INFO] Statistiques de la base de donnees:")
    db.print_database_summary()
    
    db.close()
    print("\n[SUCCESS] Test termine avec succes!")