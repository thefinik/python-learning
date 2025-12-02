from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

#load data
train_dir = "dataset_3class/train"
test_dir = "dataset_3class/test"

#load pre-trained base model
base_model = MobileNetV2(
    weights="imagenet",      
    include_top=False,       
    input_shape=(224, 224, 3)
)

#freeze the base model
base_model.trainable = False

#build classifier head
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dense(3, activation="softmax")
])

#compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

#image generators
train_gen = ImageDataGenerator(rescale=1./255)
test_gen  = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

#train the model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=3
)

#save the model
model.save("model_3class.h5")
print("Training finished. Model saved as model_3class.h5")