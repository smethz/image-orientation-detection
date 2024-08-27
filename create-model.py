import pandas as pd
import numpy as np
import tensorflow as tf
from keras import utils, models, preprocessing, layers
from sklearn.model_selection import train_test_split

def load_data(csv_file, img_size=(128, 128)):
    df = pd.read_csv(csv_file)
    X = []
    y = []
    for _, row in df.iterrows():
        img = preprocessing.image.load_img(row['path'], target_size=img_size)
        img_array = preprocessing.image.img_to_array(img) / 255.0  # Normalize pixel values
        X.append(img_array)
        y.append(row['degree'] // 30)  # Convert degree to class index
    return np.array(X), np.array(y)

# Create and compile the model
def create_model(input_shape, num_classes):
    model = models.Sequential([
       layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
       layers.MaxPooling2D((2, 2)),
       layers.Conv2D(64, (3, 3), activation='relu'),
       layers.MaxPooling2D((2, 2)),
       layers.Conv2D(64, (3, 3), activation='relu'),
       layers.Flatten(),
       layers.Dense(128, activation='relu'),
       layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        verbose=1)
    return history

# Predict function
def predict_degree(model, image_path, img_size=(128, 128)):
    img = preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_degree = predicted_class * 30
    return predicted_degree

if __name__ == "__main__":
    csv_file = 'data_rotation.csv'  # Replace with your CSV file path
    img_size = (128, 128)
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Load and preprocess data
    X, y = load_data(csv_file, img_size)
    num_classes = len(np.unique(y))
    y = utils.to_categorical(y, num_classes)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = create_model(input_shape=(*img_size, 3), num_classes=num_classes)
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Save the model
    model.save('rotation_detection_gen.h5')
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    
    # Example prediction
    test_image_path = './images/1.png'  # Replace with a test image path
    predicted_degree = predict_degree(model, test_image_path)
    print(f"Predicted rotation: {predicted_degree} degrees")
