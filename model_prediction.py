import tensorflow as tf
from tensorflow import keras
from resnet50 import resnet50

# Load the saved model
model = keras.models.load_model('/path/to/saved/model')

# Load the new image to predict
image = keras.preprocessing.image.load_img('/path/to/new/image.jpg', target_size=(224, 224))
image = keras.preprocessing.image.img_to_array(image)
image = tf.expand_dims(image, 0)

# Predict the class of the new image
predictions = model.predict(image)
class_index = tf.argmax(predictions, axis=1)[0]
class_name = ['cat', 'dog'][class_index]

# Print the predicted class name
print('Predicted class:', class_name)
