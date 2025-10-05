import tensorflow as tf


def load_and_test_model(model_filename = "first_mnist_model.keras"):

    try:
        loaded_model = tf.keras.models.load_model(model_filename)

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        loss, acc = loaded_model.evaluate(x_test, y_test)
        print(f"\nLoaded Model Accuracy: {acc*100:.2f}% Loss:{loss:.4f}")

    except Exception as e:
        print(f"Error found: {e}")

if __name__ == "__main__":
    load_and_test_model()