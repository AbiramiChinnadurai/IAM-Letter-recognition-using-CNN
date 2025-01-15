import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Paths
data_dir = "C:/Users/Abirami Chinnadurai/IAM dataset/iam-letter-recognition/data/processed"
model_path = "C:/Users/Abirami Chinnadurai/IAM dataset/iam-letter-recognition/models/letter_recognition_model.h5"

# Data generator with augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

# Training data generator
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Print the number of classes
num_classes = train_generator.num_classes
print(f'Number of classes: {num_classes}')

# Model definition with increased complexity
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Use num_classes for the output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Calculate steps_per_epoch and validation_steps
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = val_generator.samples // val_generator.batch_size

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Save the model in HDF5 format
model.save(model_path)
print(f"Model saved to {model_path}")

# Print validation accuracy and validation loss
val_loss, val_accuracy = model.evaluate(val_generator, steps=validation_steps)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")