"""Hand-rolled Grad-CAM (Selvaraju et al., 2017)."""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src import config as C
from src.models import GRADCAM_LAYER


def make_heatmap(model, img_array, layer_name, class_idx=None):
    """(h, w) heatmap in [0, 1] for one preprocessed (1, H, W, 3) image."""
    grad_model = keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    x = tf.convert_to_tensor(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv, preds = grad_model(x, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    return cam / (cam.max() + 1e-8)


def overlay(img, heatmap, alpha=C.GRADCAM_ALPHA):
    """Blend a JET-coloured heatmap onto an RGB uint8 image of any size."""
    img = np.asarray(img).astype(np.uint8)
    h = cv2.resize(heatmap.astype(np.float32), (img.shape[1], img.shape[0]))
    colour = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(colour, alpha, img, 1 - alpha, 0)


def gradcam_overlay(model, name, img_array, raw_img, class_idx=None):
    """Overlay for backbone `name`: img_array is preprocessed (1,H,W,3), raw_img is [0,255] (H,W,3)."""
    return overlay(raw_img, make_heatmap(model, img_array, GRADCAM_LAYER[name], class_idx))
