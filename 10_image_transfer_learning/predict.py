from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

#load model
model = load_model("model_3class.h5")

#class names
class_names = ["cat", "dog", "car"]

#load image
img = load_img("test.jpg", target_size=(224, 224))

#preprocess
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

#predict
pred = model.predict(img_array)
pred_class = np.argmax(pred)

print("Predicted class:", class_names[pred_class])
print("Raw output:", pred)