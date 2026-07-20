-- SCHEMA: predictions
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,  -- Auto-incremented ID
    timestamp TIMESTAMP NOT NULL,  -- Timestamp when prediction was made
    topic VARCHAR(255) NOT NULL,   -- Kafka topic (ecg_topic)
    filename VARCHAR(255) NOT NULL,  -- Filename of the ECG image
    true_class_folder VARCHAR(255),  -- True class folder name
    pred_idx INT NOT NULL,  -- Predicted class index (integer value)
    pred_label VARCHAR(255) NOT NULL,  -- Predicted class label (e.g., "Normal", "MI", etc.)
    confidence FLOAT,  -- Confidence score (float value between 0 and 1)
    latency_ms INT,  -- Time taken to process the message, in milliseconds
    status VARCHAR(255) -- Status (e.g., "ok", "bad_payload", etc.)
);
-- DROP SCHEMA IF EXISTS predictions ;

CREATE SCHEMA IF NOT EXISTS predictions
    AUTHORIZATION postgres;