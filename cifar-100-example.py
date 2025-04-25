import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

# Data augmentation settings
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)

# Fit the data augmentation generator
datagen.fit(x_train)


# SE Block with SENet
def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.multiply([input_tensor, se])


def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Add SE module
    x = se_block(x)

    # Adjust the shape of the shortcut to match x
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)

    # Add shortcut
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


# Dense Block
def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        layer = layers.BatchNormalization()(x)
        layer = layers.ReLU()(layer)
        layer = layers.Conv2D(growth_rate, (3, 3), padding='same')(layer)
        x = layers.Concatenate()([x, layer])
    return x


# Transition Layer
def transition_layer(x, compression_factor=0.5):
    filters = int(x.shape[-1] * compression_factor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = layers.AveragePooling2D((2, 2), strides=2)(x)
    return x


# Build the model
inputs = layers.Input(shape=(32, 32, 3))
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)

# Add Residual Block
x = residual_block(x, filters=64)

# Add Dense Block and Transition Layer
x = dense_block(x, num_layers=6, growth_rate=32)
x = transition_layer(x)

x = residual_block(x, filters=128)

x = dense_block(x, num_layers=6, growth_rate=32)
x = transition_layer(x)

x = residual_block(x, filters=256)

x = dense_block(x, num_layers=6, growth_rate=32)
x = transition_layer(x)

x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(100, activation='softmax')(x)

model = models.Model(inputs, outputs)

# Compile the model, including top-5 accuracy
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)])

# Set callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

# Train the model using data augmentation
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          epochs=50,
          validation_data=(x_test, y_test),
          callbacks=callbacks)

# Evaluate the model
test_loss, test_acc, test_top5_acc = model.evaluate(x_test, y_test)
print(f"Test top-1 accuracy: {test_acc}")
print(f"Test top-5 accuracy: {test_top5_acc}")
