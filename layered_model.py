from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def define_dense_model_single_layer(input_size, activation_f='sigmoid', output_length=1):
    """
    Define a dense model with a single layer.

    Parameters:
    input_size (int): The size of the input data.
    activation_f (str): Activation function for the layer.
    output_length (int): The size of the output layer.

    Returns:
    model: A TensorFlow Keras Sequential model.
    """
    model = Sequential([
        Dense(output_length, activation=activation_f, input_shape=(input_size,))
    ])
    return model

def define_dense_model_with_hidden_layer(input_size, activation_func_array, hidden_layer_size, output_length=1):
    """
    Define a dense model with one hidden layer.

    Parameters:
    input_size (int): The size of the input data.
    activation_func_array (list): List of activation functions for each layer.
    hidden_layer_size (int): The size of the hidden layer.
    output_length (int): The size of the output layer.

    Returns:
    model: A TensorFlow Keras Sequential model.
    """
    model = Sequential([
        Dense(hidden_layer_size, activation=activation_func_array[0], input_shape=(input_size,)),
        Dense(output_length, activation=activation_func_array[1])
    ])
    return model

def get_mnist_data():
    """Get the MNIST data."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255 
    return (x_train, y_train), (x_test, y_test)

def binarize_labels(labels, target_digit=2):
    """Binarize the labels."""
    labels = 1*(labels==target_digit)
    return labels

def fit_mnist_model_single_digit(x_train, y_train, digit, model, epochs=5, batch_size=128):
    """
    Train the model for classifying a single digit from the MNIST dataset.

    Parameters:
    x_train (array): Training data.
    y_train (array): Labels for training data.
    digit (int): The digit to classify.
    model (keras.Model): The model to be trained.
    epochs (int): Number of epochs for training.
    batch_size (int): Batch size for training.

    Returns:
    model: The trained TensorFlow Keras model.
    """
    # Binarize labels for the specified digit
    y_train_binary = binarize_labels(y_train, digit)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(x_train, y_train_binary, epochs=epochs, batch_size=batch_size)
    return model

def evaluate_mnist_model_single_digit(x_test, y_test, digit, model):
    """
    Evaluate the performance of the trained model on test data.

    Parameters:
    x_test (array): Test data.
    y_test (array): Labels for test data.
    digit (int): The digit to classify.
    model (keras.Model): The trained model.

    Returns:
    loss (float): The loss value.
    accuracy (float): The accuracy value.
    """
    # Binarize labels for the specified digit
    y_test_binary = binarize_labels(y_test, digit)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test_binary)
    return loss, accuracy