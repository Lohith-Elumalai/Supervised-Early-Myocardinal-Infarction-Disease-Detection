import os
import numpy as np
import cv2
import tensorflow as tf

IMG_SIZE = 224
CLASS_NAMES = ["Abnormal", "History_MI", "MI", "Normal"]
MODEL_PATH = "best_model.h5"

def load_model_for_gradcam():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"missing model: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    _ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)), training=False)  # build
    print("loaded model:", MODEL_PATH)
    return model

def preprocess_image(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"could not read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    x = np.expand_dims(img_resized, axis=0)
    return img_rgb, x

def find_last_conv_layer(mobilenet_block):
    # mobilenet_block is the Functional layer mobilenetv2_1.00_224
    for layer in reversed(mobilenet_block.layers):
        out = getattr(layer, "output", None)
        if out is not None and len(out.shape) == 4:
            return layer.name
    raise ValueError("No 4D conv layer found inside MobileNetV2")

def make_gradcam_heatmap(model, x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # Build a clean functional wrapper around your loaded Sequential
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="gradcam_input")

    base = model.layers[0]  # mobilenetv2_1.00_224 (Functional)
    feat = base(inp, training=False)  # IMPORTANT: call it to get a tensor in THIS graph

    y = feat
    # run the remaining layers of your Sequential on top of feat
    for layer in model.layers[1:]:
        # Dropout/Dense accept training kw; safest is try/except
        try:
            y = layer(y, training=False)
        except TypeError:
            y = layer(y)

    grad_model = tf.keras.Model(inputs=inp, outputs=[feat, y])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]  # (7,7,1280)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # (7,7)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(class_index.numpy()), preds.numpy()[0]

def overlay_heatmap(orig_rgb, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (orig_rgb.shape[1], orig_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(orig_rgb, 1 - alpha, heatmap, alpha, 0)

if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\Shiny\Downloads\mlproject\mlproject\dataset\ECG_IMAGES_DATASET\Abnormal\HB(18).jpg"

    model = load_model_for_gradcam()
    orig_rgb, x = preprocess_image(IMAGE_PATH)

    heatmap, pred_idx, probs, used_layer = make_gradcam_heatmap(model, x)
    pred_name = CLASS_NAMES[pred_idx]
    conf = float(probs[pred_idx])

    out = overlay_heatmap(orig_rgb, heatmap, alpha=0.4)

    os.makedirs("gradcam_output", exist_ok=True)
    out_path = os.path.join("gradcam_output", f"gradcam_{pred_name}_{conf:.3f}.png")
    cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    print("gradcam layer used:", used_layer)
    print("prediction:", pred_name, "confidence:", round(conf, 3))
    print("saved:", out_path)