import os
import time
import json
import base64
import random
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

TOPIC="ecg_topic"
BOOTSTRAP="127.0.0.1:9092"
DATASET_DIR=r"dataset\ECG_IMAGES_DATASET"

#src of img for producer
def list_images(dataset_dir):
    classes=[d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,d))]
    all_images=[]
    for cls in classes:
        cls_path=os.path.join(dataset_dir,cls)
        for f in os.listdir(cls_path):
            if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp")):
                all_images.append((cls, os.path.join(cls_path, f)))
    return classes, all_images


def ensure_topic(topic_name):
    admin=AdminClient({"bootstrap.servers":BOOTSTRAP})
    metadata=admin.list_topics(timeout=10)
    if topic_name in metadata.topics:
        return
    futures=admin.create_topics([NewTopic(topic_name, num_partitions=1, replication_factor=1)])
    try:
        futures[topic_name].result(15)
        print(f"Created Kafka topic: {topic_name}",flush=True)
    except Exception as e:
        if "TOPIC_ALREADY_EXISTS" not in str(e):
            raise


def delivery_report(err,msg):
    if err is not None:
        print("Delivery failed:",err,flush=True)
    else:
        print("Delivered to",msg.topic(),"partition",msg.partition(),flush=True)


def main():
    print("PRODUCER STARTED",flush=True)
    print("Dataset path:",os.path.abspath(DATASET_DIR),flush=True)
    if not os.path.exists(DATASET_DIR):
        print("Dataset dir not found. Fix DATASET_DIR.", flush=True)
        return
    classes,all_images=list_images(DATASET_DIR)
    print("Classes:",classes,flush=True)
    print("Total images:",len(all_images),flush=True)
    if len(all_images) == 0:
        print("No images found", flush=True)
        return

    ensure_topic(TOPIC)
    producer=Producer({"bootstrap.servers": BOOTSTRAP})
    print("Connected to Kafka:",BOOTSTRAP,flush=True)

    while True:
        true_class,img_path=random.choice(all_images)
        with open(img_path,"rb") as f:
            img_b64=base64.b64encode(f.read()).decode("utf-8")
        payload={
            "true_class_folder": true_class,
            "filename": os.path.basename(img_path),
            "image_b64": img_b64,
            "timestamp": time.time(),
        }

        producer.produce(
            TOPIC,
            value=json.dumps(payload).encode("utf-8"),
            callback=delivery_report,
        )
        producer.poll(0)

        print("Sent:",true_class, "-",os.path.basename(img_path),flush=True)
        time.sleep(3)

if __name__ == "__main__":
    main()
