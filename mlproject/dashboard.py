import os
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

MODEL_CANDIDATES=[
    "best_model.keras",
    "best_model.h5",
]
IMG_SIZE=224
CLASS_NAMES=["Abnormal", "History_MI", "MI", "Normal"]

st.set_page_config(page_title="ECG Risk Classifier", layout="wide")
st.title("ECG Image Classification Dashboard")

@st.cache_resource
def load_model():
    last_err = None

    #try loading full model
    for p in MODEL_CANDIDATES:
        if not os.path.exists(p):
            continue
        try:
            m=tf.keras.models.load_model(p,compile=False,safe_mode=False)
            print("loaded model:", p)
            return m
        except Exception as e:
            last_err=e
            print("model load failed for", p, ":", str(e).splitlines()[0])

    #else try loading weight only case
    if os.path.exists("best_model.h5") and os.path.exists("y.npy"):
        y=np.load("y.npy")
        num_classes=int(len(np.unique(y)))
        base=tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE,IMG_SIZE,3),
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False
        m=tf.keras.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ])
        m.load_weights("best_model.h5")
        print("loaded weights from best_model.h5")
        return m

    raise RuntimeError(f"could not load any model. last error: {last_err}")

model=load_model()
col1,col2=st.columns(2)

with col1:
    st.subheader("Upload ECG Image")
    file = st.file_uploader("Choose ECG image", type=["png","jpg","jpeg","bmp","webp"])
    if file:
        file_bytes=np.asarray(bytearray(file.read()),dtype=np.uint8)
        img=cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
        if img is None:
            st.error("Could not decode image")
            st.stop()
        st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),caption="Uploaded ECG",use_container_width=True)
        img_resized=cv2.resize(img,(IMG_SIZE, IMG_SIZE)).astype("float32")/255.0
        x=np.expand_dims(img_resized,axis=0)
        probs=model.predict(x,verbose=0)[0]
        pred_idx=int(np.argmax(probs))
        conf=float(probs[pred_idx])
        pred_name=CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

        st.markdown(f"### Prediction: **{pred_name}**")
        st.markdown(f"### Confidence: **{conf:.3f}**")

with col2:
    st.subheader("Class Probabilities")
    if file:
        fig=plt.figure()
        plt.bar(CLASS_NAMES,probs)
        plt.xticks(rotation=20)
        plt.ylim(0,1)
        st.pyplot(fig)
    else:
        st.info("Upload ECG image to view probabilities")

st.divider()

st.subheader("Live Streaming Predictions")
st.caption("Displays latest real-time ECG predictions streamed from Kafka consumer")

LOG_FILE="stream_predictions.csv"
if os.path.exists(LOG_FILE):
    import pandas as pd
    df=pd.read_csv(LOG_FILE)
    st.write(df.tail(10))
    if "pred_label" in df.columns and len(df) > 0:
        st.subheader("prediction counts")
        counts=df["pred_label"].value_counts()
        st.bar_chart(counts)
    if "confidence" in df.columns and len(df) > 0:
        st.subheader("confidence trend")
        conf_df=df[["timestamp", "confidence"]].dropna()
        if len(conf_df) > 0:
            conf_df["confidence"] = pd.to_numeric(conf_df["confidence"], errors="coerce")
            st.line_chart(conf_df.set_index("timestamp")["confidence"])
else:
    st.info("No stream_predictions.csv found")