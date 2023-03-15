import tensorflow as tf

def load_image(image_path):
    """Load an image from disk and preprocess it for the model."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

def predict(model, image_path, label_names):
    """Load an image, make a prediction with the model, and return the top predicted class and probability."""
    img = load_image(image_path)
    pred = model.predict(tf.expand_dims(img, axis=0))[0]
    top_class = label_names[pred.argmax()]
    top_prob = pred.max()
    return top_class, top_prob

def evaluate(model, dataset):
    """Evaluate the model on a dataset and return the loss and accuracy."""
    loss, accuracy = model.evaluate(dataset)
    return loss, accuracy

def save_model(model, filepath):
    """Save the model to disk."""
    model.save(filepath)

def load_model(filepath):
    """Load the model from disk."""
    model = tf.keras.models.load_model(filepath)
    return model
