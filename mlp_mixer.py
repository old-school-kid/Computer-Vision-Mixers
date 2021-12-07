import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import tensorflow_addons as tfa
import logging
tf.get_logger().setLevel(logging.ERROR)

tf.executing_eagerly()
# %matplotlib inline

SEEDS = 420

np.random.seed(SEEDS)
tf.random.set_seed(SEEDS)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

BATCH_SIZE = 128
BUFFER_SIZE= 420
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
NUM_CLASSES = 10
EPOCHS = 30

# model hyperparamters
IMG_SIZE = 72 # image_size % patch_size = 0 
PATCH_SIZE = 9
NUM_BLOCKS = 4
HIDDEN_DIM = 128
TOKENS_MLP_DIM = 64 
CHANNELS_MLP_DIM = 128

def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
    return image, label

data_augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip(
            "horizontal_and_vertical"
        ),
        layers.experimental.preprocessing.RandomRotation(factor=0.02),
        layers.experimental.preprocessing.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.map(preprocess_image, num_parallel_calls= tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.repeat(2)
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls= tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.cache()
test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

def MLP(x, mlp_dim):
    x = layers.Dense(mlp_dim)(x)
    x = tfa.layers.GELU()(x)
    return layers.Dense(x.shape[-1])(x)

def mixer_layer(x, tokens_mlp_dim, channels_mlp_dim):
    y = layers.LayerNormalization()(x)
    y = layers.Permute((2, 1))(y)
    
    token_mixing = MLP(y, tokens_mlp_dim)
    token_mixing = layers.Permute((2, 1))(token_mixing)
    x = x + token_mixing
    
    y = layers.LayerNormalization()(x)
    channel_mixing = MLP(y, channels_mlp_dim)
    output = x + channel_mixing
    return output

def mlp_mixer_model(num_blocks= NUM_BLOCKS,
              patch_size= PATCH_SIZE,
              hidden_dim= HIDDEN_DIM, 
              tokens_mlp_dim= TOKENS_MLP_DIM,
              channels_mlp_dim= CHANNELS_MLP_DIM,
              num_classes=10):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Conv2D(hidden_dim, kernel_size=patch_size,
                      strides=patch_size, padding="valid")(inputs)
    x = layers.Reshape((x.shape[1]*x.shape[2], x.shape[3]))(x)  # n, h, w, c -> n, (h, w), c

    for _ in range(num_blocks):
        x = mixer_layer(x, tokens_mlp_dim, channels_mlp_dim)
    
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs, name="MLP-Mixer")
    return model

keras.backend.clear_session()
model= mlp_mixer_model()
optimizer = tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = keras.losses.SparseCategoricalCrossentropy()
metric= [
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
        keras.metrics.TopKCategoricalAccuracy(3, name="top-3-accuracy")
    ]
model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metric
    )
model.summary()

filepath= 'mlp_mixer.h5'  
my_callbacks = [
  callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, verbose=1,
    mode='min', min_delta=0.0001, cooldown=0, min_lr=0 
  ), 
  callbacks.ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode= 'max')]

history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=test_dataset,
                    callbacks=my_callbacks)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

top3 = history.history['top-3-accuracy']
val_top3 = history.history['val_top-3-accuracy']

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

plt.plot(history.history["acc"], label="train_accuracy")
plt.plot(history.history["val_acc"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Accuracy Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

plt.plot(history.history["top-3-accuracy"], label="top-3_accuracy")
plt.plot(history.history["val_top-3-accuracy"], label="val_top-3_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation top-3 accuracy Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

