"""Backbones: ImageNet base -> GAP -> Dropout -> Dense(4, softmax), built as one flat graph."""
from tensorflow import keras

from src import config as C

# name -> (keras.applications constructor, preprocessing module)
_APPS = {
    "mobilenetv2": ("MobileNetV2", "mobilenet_v2"),
    "efficientnetb0": ("EfficientNetB0", "efficientnet"),
    "densenet121": ("DenseNet121", "densenet"),
}

# Last 4-D (spatial) layer of each base, found by iterating model.layers
# (tests/test_models.py re-derives these with last_spatial_layer()).
GRADCAM_LAYER = {
    "mobilenetv2": "out_relu",
    "efficientnetb0": "top_activation",
    "densenet121": "relu",
}

HEAD_LAYERS = 3  # gap, dropout, predictions


def preprocess_fn(name):
    """Backbone-specific preprocessing for float images in [0, 255]."""
    return getattr(keras.applications, _APPS[name][1]).preprocess_input


def compile_model(model, lr, eager=False):
    """eager=True skips graph tracing, which dominates on tiny smoke runs."""
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"],
                  run_eagerly=eager, jit_compile=False)


def build_backbone(name, weights="imagenet", img_size=C.IMG_SIZE):
    """Compiled Keras model. Flat graph (no nested base) so Grad-CAM can reach inner layers."""
    base = getattr(keras.applications, _APPS[name][0])(
        include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    x = keras.layers.GlobalAveragePooling2D(name="head_gap")(base.output)
    x = keras.layers.Dropout(C.DROPOUT, name="head_dropout")(x)
    out = keras.layers.Dense(C.NUM_CLASSES, activation="softmax", name="head_predictions")(x)
    model = keras.Model(base.input, out, name=name)
    compile_model(model, C.LR_HEAD)
    return model


def set_phase(model, phase):
    """Phase 1: base frozen. Phase 2: top UNFREEZE_FRACTION of base trainable (BatchNorm stays frozen)."""
    base = model.layers[:-HEAD_LAYERS]
    cut = len(base) - int(len(base) * C.UNFREEZE_FRACTION) if phase == 2 else len(base)
    for i, layer in enumerate(base):
        layer.trainable = i >= cut and not isinstance(layer, keras.layers.BatchNormalization)


def last_spatial_layer(model):
    return [l.name for l in model.layers if len(l.output.shape) == 4][-1]
