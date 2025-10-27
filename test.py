#!/usr/bin/env python3

# Test script for loading and running the trained medikit model
# This script loads model_medikit.keras and tests the trained agent

import argparse
import itertools as it
import os
import numpy as np
import skimage.transform
import tensorflow as tf
from time import sleep

import vizdoom as vzd

# Model parameters
resolution = (30, 45)
frames_per_action = 12
episodes_to_test = 20

# Configuration file path - Health Gathering scenario
config_file_path = os.path.join(vzd.scenarios_path, "health_gathering.cfg")
model_path = "./model_medikit.keras"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the trained ViZDoom DQN agent - Health Gathering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py
  python test.py --episodes 5
  python test.py --episodes 10 --no-window
        
Health Gathering Scenario:
  - Tests the agent's ability to collect medikits
  - Rewards: +1 per tick alive, -100 for death
        """
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=episodes_to_test,
        help=f"Number of episodes to test (default: {episodes_to_test})"
    )
    
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without showing the game window (default: False)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=model_path,
        help=f"Path to the model file (default: {model_path})"
    )
    
    return parser.parse_args()


def preprocesar(img):
    """Preprocess game screen."""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=-1)
    return tf.stack(img)


def inicializar_juego(ventana_visible=True):
    """Initialize ViZDoom game."""
    print("Initializing doom...")
    juego = vzd.DoomGame()
    juego.load_config(config_file_path)
    juego.set_window_visible(ventana_visible)
    juego.set_mode(vzd.Mode.ASYNC_PLAYER)
    juego.set_screen_format(vzd.ScreenFormat.GRAY8)
    juego.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    juego.init()
    print("Doom initialized.")
    
    if ventana_visible:
        print("Game window is visible.")
    else:
        print("Running in headless mode (no window).")
    
    return juego


class DQN(tf.keras.Model):
    """Custom DQN model architecture - must match training script."""
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, kernel_size=6, strides=3, input_shape=(30, 45, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        
        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, kernel_size=3, strides=2, input_shape=(9, 14, 8)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        
        self.flatten = tf.keras.layers.Flatten()
        self.state_value = tf.keras.layers.Dense(1)
        self.advantage = tf.keras.layers.Dense(num_actions)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x1 = x[:, :96]
        x2 = x[:, 96:]
        x1 = self.state_value(x1)
        x2 = self.advantage(x2)
        
        x = x1 + (x2 - tf.reshape(tf.math.reduce_mean(x2, axis=1), shape=(-1, 1)))
        return x


class DQNAgent:
    """Simplified DQN Agent for testing (no training)."""
    def __init__(self, model_path, num_actions=8):
        self.num_actions = num_actions
        
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            # Load model with custom DQN class
            custom_objects = {'DQN': DQN}
            self.dqn = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def elegir_accion(self, estado):
        """Choose action using the trained model (no exploration)."""
        # Always use the model prediction (no epsilon exploration in testing)
        accion = int(tf.argmax(self.dqn(tf.reshape(estado, (1, 30, 45, 1))), axis=1))
        return accion


def test_agent(agente, juego, num_episodes=20):
    """Test the trained agent."""
    puntuaciones = []
    
    print(f"\nTesting agent over {num_episodes} episodes...")
    print("=" * 50)
    
    for episodio in range(num_episodes):
        juego.new_episode()
        
        while not juego.is_episode_finished():
            estado_juego = juego.get_state()
            
            if estado_juego is None:
                break
            
            estado = preprocesar(estado_juego.screen_buffer)
            indice_mejor_accion = agente.elegir_accion(estado)
            
            # Use smooth animation for better viewing
            juego.set_action(acciones[indice_mejor_accion])
            for _ in range(frames_per_action):
                juego.advance_action()
        
        # Get episode results
        puntuacion = juego.get_total_reward()
        puntuaciones.append(puntuacion)
        
        print(f"Episode {episodio + 1}/{num_episodes}: Score = {puntuacion}")
        
        # Sleep between episodes
        if episodio < num_episodes - 1:
            sleep(1.0)
    
    # Calculate and display statistics
    puntuaciones = np.array(puntuaciones)
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Episodes tested: {num_episodes}")
    print(f"Mean score: {puntuaciones.mean():.1f} ± {puntuaciones.std():.1f}")
    print(f"Min score: {puntuaciones.min():.1f}")
    print(f"Max score: {puntuaciones.max():.1f}")
    print(f"Median score: {np.median(puntuaciones):.1f}")
    print("=" * 50)
    
    return puntuaciones


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        print("Please make sure the model has been trained first.")
        exit(1)
    
    # Initialize game
    ventana_visible = not args.no_window
    juego = inicializar_juego(ventana_visible=ventana_visible)
    
    # Get number of available buttons and create action space
    n = juego.get_available_buttons_size()
    acciones = [list(a) for a in it.product([0, 1], repeat=n)]
    
    # Load agent
    agente = DQNAgent(model_path=args.model, num_actions=len(acciones))
    
    # Test the agent
    try:
        puntuaciones = test_agent(agente, juego, num_episodes=args.episodes)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    finally:
        juego.close()
        print("\nTest completed. Doom closed.")

