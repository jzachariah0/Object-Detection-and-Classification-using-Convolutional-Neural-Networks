import os
import tensorflow as tf
from resnet50 import create_model
from data_preprocessing import create_dataset
from model_training import train_model
from utils import predict, evaluate, save_model

# Define constants
NUM_CLASSES = 10
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Define file paths
TRAIN_DIR = "train/"
VAL_DIR = "val/"
TEST_DIR = "test/"
MODEL_PATH = "model.h5"

# Create the model
model = create_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE)

# Create the datasets
train_ds = create_dataset(TRAIN_DIR, IMG_SIZE, BATCH_SIZE, shuffle=True)
val_ds = create_dataset(VAL_DIR, IMG_SIZE, BATCH_SIZE, shuffle=False)
test_ds = create_dataset(TEST_DIR, IMG_SIZE, BATCH_SIZE, shuffle=False)

# Train the model
train_model(model, train_ds, val_ds, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

# Evaluate the model on the test set
test_loss, test_accuracy = evaluate(model, test_ds)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# Save the model to disk
save_model(model, MODEL_PATH)

# Make a prediction with the model
label_names = sorted(os.listdir(TRAIN_DIR))
image_path = "test_image.jpg"
top_class, top_prob = predict(model, image_path, label_names)
print("Top class:", top_class)
print("Top probability:", top_prob)
