# Supervised Early Myocardial Infarction Disease Detection

An end-to-end ECG image classification project that combines transfer learning,
Kafka-based streaming inference, PostgreSQL prediction storage, and a Streamlit
dashboard.

> **Medical notice:** This project is for research and educational use only. It
> is not a medical device and must not be used to diagnose, triage, or treat a
> patient. Predictions require validation by qualified medical professionals.

## What the project does

The project classifies ECG images into four dataset classes:

- `Abnormal`
- `History_MI`
- `MI`
- `Normal`

The model is a frozen ImageNet-pretrained MobileNetV2 feature extractor with a
small trainable classification head. Images are resized to `224 x 224` and
scaled to the `[0, 1]` range.

The application has two inference paths:

1. **Interactive inference:** upload an image in the Streamlit dashboard.
2. **Streaming inference:** the Kafka producer publishes ECG images, the Kafka
   consumer predicts each message, appends the result to
   `stream_predictions.csv`, and writes the result to PostgreSQL.

## Architecture

```text
ECG image dataset
        |
        v
image_preprocessing.py ---> X.npy, y.npy
        |
        v
training.py ---> best_model.h5, best_weights.weights.h5, mlruns/
        |
        +----------------------+
        |                      |
        v                      v
dashboard.py             producer.py ---> Kafka topic: ecg_topic
                               |
                               v
                         consumer.py
                               |
                    +----------+----------+
                    |                     |
                    v                     v
             stream_predictions.csv   PostgreSQL
                                             |
                                             v
                                  Superset container/dashboard
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `mlproject/image_preprocessing.py` | Loads the class-folder dataset, resizes images, normalizes pixels, and creates `X.npy` and `y.npy`. |
| `mlproject/training.py` | Trains and evaluates the MobileNetV2 classifier and records metrics/models in MLflow. |
| `mlproject/dashboard.py` | Streamlit UI for image upload, prediction confidence, class probabilities, and recent streaming results. |
| `mlproject/producer.py` | Selects random dataset images and publishes base64-encoded messages to Kafka. |
| `mlproject/consumer.py` | Consumes Kafka messages, runs inference, logs results to CSV, and inserts rows into PostgreSQL. |
| `mlproject/to/gradcam.py` | Grad-CAM helper for visualizing image regions used by the classifier. |
| `mlproject/schema.sql` | PostgreSQL table definition for prediction records. |
| `mlproject/PG Local.session.sql` | SQL session/export containing the prediction table and a PostgreSQL schema declaration. |
| `mlproject/Dockerfile` | Builds an Apache Superset image with the PostgreSQL driver. |
| `mlproject/docker-compose.yml` | Starts Superset on port `8088`. |
| `mlproject/superset_config.py` | Superset secret key and PostgreSQL SQLAlchemy connection configuration. |
| `mlproject/dataset/ECG_IMAGES_DATASET/` | Class-organized ECG image dataset. |

## Prerequisites

- Python 3.9 or newer
- A working Kafka broker available at `127.0.0.1:9092`
- PostgreSQL available at `localhost:5432`
- Docker Desktop and Docker Compose (only required for Superset)
- A Python environment with the packages used by the scripts:

```powershell
python -m pip install numpy opencv-python tensorflow scikit-learn mlflow `
  confluent-kafka psycopg2-binary streamlit matplotlib pandas
```

The repository does not currently include a `requirements.txt`; the command
above reflects the imports used by the application.

## Quick start

Run the following commands from the `mlproject` directory. The scripts use
relative paths for the dataset, generated arrays, model files, logs, and MLflow
artifacts.

```powershell
cd mlproject
```

### 1. Prepare the training arrays

```powershell
python image_preprocessing.py
```

This creates:

- `X.npy`: normalized image data with shape `(N, 224, 224, 3)`
- `y.npy`: integer class labels

The generated files are ignored by Git.

### 2. Train the model

```powershell
python training.py
```

Training uses an 80/20 stratified train/test split, a 20% validation split
inside the training set, a batch size of `32`, and up to `10` epochs with early
stopping on validation loss.

The command produces:

- `best_model.h5`: complete Keras model
- `best_weights.weights.h5`: model weights
- `mlruns/`: local MLflow tracking data

### 3. Run the Streamlit dashboard

```powershell
streamlit run dashboard.py
```

Open the URL printed by Streamlit, usually
`http://localhost:8501`. Upload a supported image (`png`, `jpg`, `jpeg`, `bmp`,
or `webp`) to view the predicted class, confidence, and class probabilities.
The dashboard also displays the latest rows from `stream_predictions.csv` when
that file exists.

### 4. Start Kafka streaming

Start Kafka separately and verify that it is listening on
`127.0.0.1:9092`. Then open two terminals in `mlproject`.

Terminal 1:

```powershell
python consumer.py
```

Terminal 2:

```powershell
python producer.py
```

The producer creates the `ecg_topic` topic if necessary, selects a random ECG
image approximately every three seconds, and publishes its base64-encoded
contents. The consumer loads `best_model.h5` when available, writes results to
`stream_predictions.csv`, and attempts to insert them into PostgreSQL.

## PostgreSQL setup

Create the database and prediction table before running the consumer:

```powershell
psql -U postgres -c "CREATE DATABASE ecg_db;"
psql -U postgres -d ecg_db -f schema.sql
```

Configure the PostgreSQL connection in `consumer.py` for the local database
credentials. The table stores:

| Column | Description |
| --- | --- |
| `timestamp` | Time the message was processed |
| `topic` | Kafka topic |
| `filename` | Source image filename |
| `true_class_folder` | Dataset folder supplied by the producer |
| `pred_idx` | Numeric predicted class index |
| `pred_label` | Predicted class name |
| `confidence` | Highest softmax probability |
| `latency_ms` | Inference and message-processing latency |
| `status` | Processing state, such as `ok` or `bad_payload` |

## Superset

The Compose file starts Apache Superset and exposes it at
`http://localhost:8088`:

```powershell
cd mlproject
docker compose up --build
```

The Compose configuration mounts `superset_config.py` and persists Superset
data in `superset_home/`. The Superset image installs `psycopg2-binary` so it
can connect to PostgreSQL.

When Superset runs in Docker on Windows, the configured database host is
`host.docker.internal`. Ensure PostgreSQL accepts that connection and that the
database, username, password, and port in `superset_config.py` are correct.

The current Compose and Superset configuration contains development secret and
connection settings. Replace them with environment variables or deployment
secrets before sharing the project or using it outside a local environment.
Never commit real database passwords or production secret keys.

Stop Superset with:

```powershell
docker compose down
```

## Generated files and Git-ignored artifacts

The following files are generated during preprocessing, training, streaming,
or Superset operation and are intentionally ignored:

```text
mlproject/X.npy
mlproject/y.npy
mlproject/best_model.h5
mlproject/best_model.keras
mlproject/best_weights.weights.h5
mlproject/mlruns/
mlproject/stream_predictions.csv
mlproject/superset_home/
```

Keep trained models and MLflow runs outside source control unless the project
needs a versioned model-release process.

## Data format

The preprocessing and producer scripts expect this directory structure:

```text
mlproject/
└── dataset/
    └── ECG_IMAGES_DATASET/
        ├── Abnormal/
        ├── History_MI/
        ├── MI/
        └── Normal/
```

Each class directory should contain readable image files. The current dataset
contains 928 images in total:

| Class | Images |
| --- | ---: |
| `Abnormal` | 233 |
| `History_MI` | 172 |
| `MI` | 239 |
| `Normal` | 284 |

## Limitations

- The project has no automated test suite or pinned dependency lock file.
- Kafka and PostgreSQL are external dependencies; `docker-compose.yml` starts
  Superset only.
- Training and inference depend on the class ordering used by the dataset and
  the `CLASS_NAMES` lists in the inference scripts. Keep these synchronized
  when changing dataset classes.
- The model is evaluated on a held-out split from the supplied dataset. No
  clinical validation, calibration, bias analysis, or external test-set
  evaluation is included.
- Generated model files must exist in the working directory before starting
  dashboard or consumer inference.

## License and dataset

No license or dataset provenance information is currently included in this
repository. Add the applicable software and dataset licenses before
redistributing the code or ECG images.
