import os
import json
import csv
import time
import base64
import datetime as dt
import numpy as np
from confluent_kafka import Consumer, KafkaError
import tensorflow as tf
import psycopg2
from psycopg2 import sql

BOOTSTRAP_SERVERS="127.0.0.1:9092"
INPUT_TOPIC="ecg_topic"
GROUP_ID="mlproject-consumer"

MODEL_CANDIDATES=[
    "best_model.keras",
    "best_model.h5",
]

IMG_SIZE=224
CLASS_NAMES=["Abnormal","History_MI","MI","Normal"]
LOG_FILE="stream_predictions.csv"

def _short_error(e):
    return str(e).splitlines()[0]


def _build_training_architecture(num_classes):
    base=tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    return tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes,activation="softmax"),
    ])


def _load_from_weights_h5(weights_path="best_model.h5"):
    if not os.path.exists(weights_path):
        return None
    if not os.path.exists("y.npy"):
        print("Missing y.npy")
        return None
    try:
        y=np.load("y.npy")
        num_classes=int(len(np.unique(y)))
        model=_build_training_architecture(num_classes)
        model.load_weights(weights_path)
        print("Loaded model from weights:",weights_path)
        print("Model input shape:",	model.input_shape)
        return model
    except Exception as e:
        print(f"Weight load failed for {weights_path}: {_short_error(e)}")
        return None


def load_model_if_available():
    for model_path in MODEL_CANDIDATES:
        if not os.path.exists(model_path):
            continue
        try:
            model=tf.keras.models.load_model(model_path,compile=False,safe_mode=False)
            print("Loaded model:", model_path)
            print("Model input shape:", model.input_shape)
            return model
        except Exception as e:
            print(f"Model load failed for {model_path}: {_short_error(e)}")
    return _load_from_weights_h5("best_model.h5")

def _ensure_log_file():
    if os.path.exists(LOG_FILE):
        return
    with open(LOG_FILE,"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp",
            "topic",
            "filename",
            "true_class_folder",
            "pred_idx",
            "pred_label",
            "confidence",
            "latency_ms",
            "status",
        ])
def _append_log_row(row):
    _ensure_log_file()
    with open(LOG_FILE,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def insert_into_postgresql(ts,topic,filename,true_folder,pred_idx,pred_label,conf,latency_ms,status):
    try:
        conn = psycopg2.connect(
            dbname="ecg_db",
            user="postgres",
            password="shiny",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()
        insert_query = """
            INSERT INTO predictions (
                timestamp,
                topic,
                filename,
                true_class_folder,
                pred_idx,
                pred_label,
                confidence,
                latency_ms,
                status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(insert_query, (
            ts,
            topic,
            filename,
            true_folder,
            int(pred_idx),
            pred_label,
            float(conf),
            int(latency_ms),
            status
        ))
        conn.commit()
    except Exception as e:
        print("Error inserting into PostgreSQL:", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def _decode_image_from_payload(payload):
    img_bytes= None
    if isinstance(payload,dict) and payload.get("image_path"):
        p=payload["image_path"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"image_path not found: {p}")
        img_bytes=tf.io.read_file(p).numpy()
    elif isinstance(payload, dict) and payload.get("image_b64"):
        img_bytes=base64.b64decode(payload["image_b64"])
    elif isinstance(payload, dict) and payload.get("image_bytes"):
        img_bytes=bytes(payload["image_bytes"])
    if img_bytes is None:
        return None

    img=tf.io.decode_image(img_bytes,channels=3,expand_animations=False)
    img=tf.image.resize(img,(IMG_SIZE,IMG_SIZE))
    img=tf.cast(img,tf.float32)/255.0
    img=tf.expand_dims(img,axis=0)
    return img.numpy()


def main():
    print("CONSUMER STARTED")
    _ensure_log_file()

    model=load_model_if_available()
    if model is None:
        print("No compatible model loaded. Consumer will run without inference")

    consumer=Consumer({
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([INPUT_TOPIC])
    warned_missing_topic=False

    try:
        while True:
            msg = consumer.poll(1)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    if not warned_missing_topic:
                        print(f"Topic '{INPUT_TOPIC}' is missing. Start producer to create it")
                        warned_missing_topic = True
                    continue
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print("Consumer error:", msg.error())
                continue

            warned_missing_topic=False
            t0=time.perf_counter()
            filename=""
            true_folder=""
            status="ok"
            try:
                payload=json.loads(msg.value().decode("utf-8"))
                filename=str(payload.get("filename", ""))
                true_folder= str(payload.get("true_class_folder", ""))
                print("Message received:",true_folder,"-",filename)
            except Exception:
                payload = None
                status = "bad_payload"
                print("Message received (unparseable payload)")

            pred_idx = ""
            pred_label = ""
            conf = ""
            try:
                if model is not None and payload is not None:
                    x=_decode_image_from_payload(payload)
                    if x is None:
                        status="no_image_in_payload"
                    else:
                        probs=model.predict(x,verbose=0)[0]
                        pred_idx=int(np.argmax(probs))
                        conf=float(probs[pred_idx])
                        pred_label=CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
                        print(f"Pred: {pred_label}  conf={conf:.3f}")
            except Exception as e:
                status=f"infer_error:{_short_error(e)}"
                print("Inference error:", _short_error(e))

            latency_ms=int((time.perf_counter()-t0)*1000)
            ts = dt.datetime.now().isoformat(timespec="seconds")
            _append_log_row([
                ts,
                INPUT_TOPIC,
                filename,
                true_folder,
                pred_idx,
                pred_label,
                conf,
                latency_ms,
                status,
            ])
            insert_into_postgresql(
                ts,
                INPUT_TOPIC,
                filename,
                true_folder,
                pred_idx,
                pred_label,
                conf,
                latency_ms,
                status
            )
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()