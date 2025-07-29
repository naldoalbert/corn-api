CREATE DATABASE if not exists cornPredict;

USE cornPredict;

CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    label VARCHAR(50),
    confidence FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);