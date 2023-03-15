import tensorflow as tf
from tensorflow import keras
from resnet50 import resnet50

# Load the saved model
model = keras.models.load_model('/path/to/saved/model')

# Load the test dataset
test_dataset = keras.preprocessing.image_dataset_from_directory(
    '/path/to/test/dataset',
    image_size=(224, 224),
    batch_size=32)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_dataset)

# Print the test accuracy
print('Test accuracy:', accuracy)
