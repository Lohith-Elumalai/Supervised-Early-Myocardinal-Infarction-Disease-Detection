CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,  
    timestamp TIMESTAMP NOT NULL, 
    topic VARCHAR(255) NOT NULL,  
    filename VARCHAR(255) NOT NULL,  
    true_class_folder VARCHAR(255),  
    pred_idx INT,
    pred_label VARCHAR(255) NOT NULL, 
    confidence FLOAT,  
    latency_ms INT,
    status VARCHAR(255) 
);