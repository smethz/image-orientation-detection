from keras import  models, preprocessing 
import numpy as np

def predict_degree(model, image_path, img_size=(128, 128)):
    img = preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_degree = predicted_class * 30
    return predicted_degree

model = models.load_model('rotation_detection_gen.h5')

while True:
    test_path = input('image path ')
    test_image_path = './images/' + test_path
    print(f'image path: {test_image_path}')
    predicted_degree = predict_degree(model, test_image_path)
    print(f"Predicted rotation: {predicted_degree} degrees")
    