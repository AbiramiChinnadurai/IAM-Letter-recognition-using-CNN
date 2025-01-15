import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path, test_data_dir, batch_size=32):
    # Load the model
    model = load_model(model_path)

    # Prepare the test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(28, 28),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Make predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Visualize predictions
    num_images = min(len(predicted_classes), 10)  # Adjust the number of images to display
    plt.figure(figsize=(15, 15))
    indices = np.random.choice(len(test_generator.filenames), num_images, replace=False)
    for i, idx in enumerate(indices):
        img, label = test_generator[idx // batch_size]
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img[idx % batch_size].reshape(28, 28), cmap='gray')
        plt.title(f'Predicted: {predicted_classes[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Visualize class distribution
    plt.figure(figsize=(10, 5))
    plt.hist(predicted_classes, bins=np.arange(test_generator.num_classes + 1) - 0.5, edgecolor='black')
    plt.xticks(range(test_generator.num_classes))
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Predicted Class Distribution')
    plt.show()

    return loss, accuracy

if __name__ == "__main__":
    model_path = 'C:/Users/Abirami Chinnadurai/IAM dataset/iam-letter-recognition/models/letter_recognition_model.h5'
    test_data_dir = 'C:/Users/Abirami Chinnadurai/IAM dataset/iam-letter-recognition/data/processed'
    
    # Evaluate the model
    loss, accuracy = evaluate_model(model_path, test_data_dir)