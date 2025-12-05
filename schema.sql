-- Table principale: Mesures énergétiques
CREATE TABLE energy_measurements (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    household_id INT NOT NULL,
    global_active_power DECIMAL(8,3),
    global_reactive_power DECIMAL(8,3),
    voltage DECIMAL(6,2),
    global_intensity DECIMAL(6,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_timestamp (timestamp),
    INDEX idx_household (household_id),
    FOREIGN KEY (household_id) REFERENCES households(id)
);

-- Table: Foyers
CREATE TABLE households (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    location VARCHAR(255),
    surface_area DECIMAL(6,2),
    occupants INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: Sous-compteurs (appareils)
CREATE TABLE sub_meters (
    id INT AUTO_INCREMENT PRIMARY KEY,
    measurement_id INT NOT NULL,
    meter_type ENUM('kitchen', 'laundry', 'climate') NOT NULL,
    energy_consumed DECIMAL(8,3),
    FOREIGN KEY (measurement_id) REFERENCES energy_measurements(id),
    INDEX idx_measurement (measurement_id),
    INDEX idx_type (meter_type)
);

-- Table: Agrégations horaires (optimisation requêtes)
CREATE TABLE hourly_consumption (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    hour INT NOT NULL,
    household_id INT NOT NULL,
    avg_power DECIMAL(8,3),
    max_power DECIMAL(8,3),
    total_energy DECIMAL(10,3),
    UNIQUE KEY unique_datetime (date, hour, household_id),
    FOREIGN KEY (household_id) REFERENCES households(id)
);

-- Table: Agrégations journalières
CREATE TABLE daily_consumption (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    household_id INT NOT NULL,
    total_energy DECIMAL(10,3),
    avg_power DECIMAL(8,3),
    max_power DECIMAL(8,3),
    min_power DECIMAL(8,3),
    peak_hour INT,
    UNIQUE KEY unique_date (date, household_id),
    FOREIGN KEY (household_id) REFERENCES households(id)
);

-- Table: Prédictions
CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    household_id INT NOT NULL,
    predicted_power DECIMAL(8,3),
    actual_power DECIMAL(8,3),
    model_name VARCHAR(50),
    confidence DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (household_id) REFERENCES households(id)
);
