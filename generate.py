import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_image(model_path, latent_dim=100):
    """
    Loads a trained GAN generator model and generates a single synthetic image.

    Parameters:
    - model_path (str): Path to the saved generator model (.keras or .h5).
    - latent_dim (int): Dimensionality of the latent space (default: 100).

    Returns:
    - None (displays the generated image).
    """
    # Load the trained generator model
    generator = tf.keras.models.load_model(model_path)

    # Generate a random noise vector (latent space input)
    random_latent_vector = np.random.normal(0, 1, (1, latent_dim))

    # Generate an image from the noise
    generated_image = generator.predict(random_latent_vector)

    # Rescale pixel values (assuming generator outputs [-1, 1] range)
    generated_image = (generated_image + 1) / 2.0

    # Display the generated image
    plt.imshow(generated_image[0], cmap="gray")  # Change cmap if using color images
    plt.axis("off")
    plt.show()
generate_image("generator_model.keras")  # or "generator_model.h5"
