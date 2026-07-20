import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.tensorflow


def main():
    #mlflow setup
    project_dir=Path(__file__).resolve().parent
    mlruns_dir = project_dir/"mlruns"
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment("ECG_Classification_Local")

    #load dataset
    X = np.load("X.npy")
    y = np.load("y.npy")
    print("Data Loaded:", X.shape, y.shape)

    #to ensure data imported properly
    if len(X.shape)!=4 or X.shape[-1]!=3:
        raise ValueError(f"Expected X shape (N,224,224,3). Got {X.shape}")

    num_classes=len(np.unique(y))
    print("Classes:", num_classes, "Unique labels:", np.unique(y))

    #split data for training n testing
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    #model transfer learning using MobileNetV2
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    #build model
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    #callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    epochs=10
    batch_size=32

    #train model n log to mlflow
    with mlflow.start_run():
        mlflow.log_param("model", "MobileNetV2_no_top")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("img_size", 224)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
	
        history=model.fit(
            X_train,y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
	
	#eval model
        loss,acc=model.evaluate(X_test,y_test, verbose=1)
        print("Test Loss:", loss)
        print("Test Accuracy:", acc)
        mlflow.log_metric("test_loss", float(loss))
        mlflow.log_metric("test_accuracy", float(acc))

        #log last epoch vals
        if "accuracy" in history.history:
            mlflow.log_metric("train_accuracy_last", float(history.history["accuracy"][-1]))
        if "val_accuracy" in history.history:
            mlflow.log_metric("val_accuracy_last", float(history.history["val_accuracy"][-1]))
        if "loss" in history.history:
            mlflow.log_metric("train_loss_last", float(history.history["loss"][-1]))
        if "val_loss" in history.history:
            mlflow.log_metric("val_loss_last", float(history.history["val_loss"][-1]))

        #log model to mlflow artifacts
        mlflow.tensorflow.log_model(
            model,
            name="model",
            input_example=X_train[:1]
        )

    model.save_weights("best_weights.weights.h5")
    model.save("best_model.h5")
    print("Saved: best_model.h5")
    print("MLflow logs stored in: ./mlruns")

if __name__ == "__main__":
    main()
