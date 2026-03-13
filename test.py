import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model("model.h5")

datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    "dataset",
    target_size=(64, 64),
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)

loss, acc = model.evaluate(test_data)
print("Test Accuracy:", acc)