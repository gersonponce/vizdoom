#!/usr/bin/env python3

# M. Kempka, T.Sternal, M.Wydmuch, Z.Boztoprak
# Updated for Python 3.13 and TensorFlow 2.15+
# January 2021 - Updated 2024
#
# TRAINING FOR HEALTH GATHERING (MEDIKIT COLLECTING) SCENARIO
# - Train an agent to collect medikits for survival
# - Rewards: +1 for living, -100 for death
# - Actions: Turn left/right, Move forward
#
# NEW FEATURES:
# - Command line arguments for device selection:
#   Usage: python learning_tensorflow_medikit.py --device [auto|gpu|cpu]
#   * "auto": Automatically use GPU if available, otherwise CPU (default)
#   * "gpu": Force GPU usage (falls back to CPU if not available)
#   * "cpu": Force CPU usage
# - Training time tracking: The script now tracks and displays total training time

import argparse
import itertools as it
import os
from collections import deque
from random import sample
from time import sleep, time

import numpy as np
import skimage.color
import skimage.transform
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, ReLU
from tensorflow.keras.optimizers import SGD
from tqdm import trange

import vizdoom as vzd

# Eager execution is default in TensorFlow 2.x, no need to enable it
# tf.compat.v1.enable_eager_execution()  # REMOVED - Not needed in modern TensorFlow

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
replay_memory_size = 10000
num_train_epochs = 10  # Increased for health gathering
learning_steps_per_epoch = 2000
target_net_update_steps = 1000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frames_per_action = 12
resolution = (30, 45)
episodes_to_watch = 20

save_model = True
load = False
skip_learning = False
watch = True

# Configuration file path - Health Gathering scenario
config_file_path = os.path.join(vzd.scenarios_path, "health_gathering.cfg")
model_savefolder = "./model_medikit.keras"  # Different model for medikit scenario


def parse_arguments():
    """Parse command line arguments for device selection."""
    parser = argparse.ArgumentParser(
        description="ViZDoom Deep Q-Network Training with TensorFlow - Health Gathering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python learning_tensorflow_medikit.py --device gpu
  python learning_tensorflow_medikit.py --device cpu --watch-training
  python learning_tensorflow_medikit.py --device auto
        
Health Gathering Scenario:
  - Train an agent to collect medikits for survival
  - Rewards: +1 per tick alive, -100 for death
  - Actions: Turn left, Turn right, Move forward
        """
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Device to use for training: 'auto' (use GPU if available), 'gpu', or 'cpu' (default: auto)"
    )
    
    parser.add_argument(
        "--watch-training",
        action="store_true",
        help="Show the game window during training (default: False)"
    )
    
    return parser.parse_args()


def setup_device(device_choice):
    """Setup and return the TensorFlow device to use."""
    gpu_devices = tf.config.list_physical_devices("GPU")
    
    if device_choice.lower() == "gpu":
        if len(gpu_devices) > 0:
            print("Forcing GPU usage")
            return "/gpu:0"
        else:
            print("GPU not available, falling back to CPU")
            return "/cpu:0"
    elif device_choice.lower() == "cpu":
        print("Forcing CPU usage")
        return "/cpu:0"
    else:  # auto
        if len(gpu_devices) > 0:
            print("GPU available, using GPU")
            return "/gpu:0"
        else:
            print("No GPU available, using CPU")
            return "/cpu:0"


def preprocesar(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=-1)

    return tf.stack(img)


def inicializar_juego(ventana_visible=False):
    print("Initializing doom...")
    juego = vzd.DoomGame()
    juego.load_config(config_file_path)
    juego.set_window_visible(ventana_visible)
    juego.set_mode(vzd.Mode.PLAYER)
    juego.set_screen_format(vzd.ScreenFormat.GRAY8)
    juego.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    juego.init()
    print("Doom initialized.")
    if ventana_visible:
        print("Game window is visible during training.")

    return juego


class DQNAgent:
    def __init__(
        self, num_actions=8, epsilon=1, epsilon_min=0.1, epsilon_decay=0.9995, load=load
    ):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.num_actions = num_actions
        self.optimizer = SGD(learning_rate=learning_rate)

        if load:
            print("Loading model from: ", model_savefolder)
            self.dqn = tf.keras.models.load_model(model_savefolder)
        else:
            self.dqn = DQN(self.num_actions)
            self.red_objetivo = DQN(self.num_actions)

    def actualizar_red_objetivo(self):
        self.red_objetivo.set_weights(self.dqn.get_weights())

    def elegir_accion(self, estado):
        if self.epsilon < np.random.uniform(0, 1):
            accion = int(tf.argmax(self.dqn(tf.reshape(estado, (1, 30, 45, 1))), axis=1))
        else:
            accion = np.random.choice(range(self.num_actions), 1)[0]

        return accion

    def entrenar_dqn(self, muestras):
        bufer_pantalla, acciones, recompensas, bufer_pantalla_sig, terminados = dividir_tupla(muestras)

        ids_fila = list(range(bufer_pantalla.shape[0]))

        ids = extraer_digitos(ids_fila, acciones)
        ids_terminados = extraer_digitos(np.where(terminados)[0])

        with tf.GradientTape() as tape:
            # No need to watch variables explicitly in modern TensorFlow
            Q_prev = tf.gather_nd(self.dqn(bufer_pantalla), ids)

            Q_sig = self.red_objetivo(bufer_pantalla_sig)
            Q_sig = tf.gather_nd(
                Q_sig,
                extraer_digitos(ids_fila, tf.argmax(self.dqn(bufer_pantalla_sig), axis=1)),
            )

            objetivo_q = recompensas + self.discount_factor * Q_sig

            if len(ids_terminados) > 0:
                recompensas_terminados = tf.gather_nd(recompensas, ids_terminados)
                objetivo_q = tf.tensor_scatter_nd_update(
                    tensor=objetivo_q, indices=ids_terminados, updates=recompensas_terminados
                )

            # Use modern loss API
            #td_error = tf.reduce_mean(tf.keras.losses.mean_squared_error(q_target, Q_prev))
            error_td = tf.keras.losses.MeanSquaredError()(objetivo_q, Q_prev)
        gradientes = tape.gradient(error_td, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradientes, self.dqn.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


def dividir_tupla(muestras):
    muestras = np.array(muestras, dtype=object)
    bufer_pantalla = tf.stack(muestras[:, 0])
    acciones = muestras[:, 1]
    recompensas = tf.stack(muestras[:, 2])
    bufer_pantalla_sig = tf.stack(muestras[:, 3])
    terminados = tf.stack(muestras[:, 4])
    return bufer_pantalla, acciones, recompensas, bufer_pantalla_sig, terminados


def extraer_digitos(*argumentos):
    if len(argumentos) == 1:
        return list(map(lambda x: [x], argumentos[0]))

    return list(map(lambda x, y: [x, y], argumentos[0], argumentos[1]))


def obtener_muestras(memoria):
    if len(memoria) < batch_size:
        tamano_muestra = len(memoria)
    else:
        tamano_muestra = batch_size

    return sample(memoria, tamano_muestra)


def ejecutar(agente, juego, memoria_replay):
    tiempo_inicio = time()

    for episodio in range(num_train_epochs):
        puntuaciones_entrenamiento = []
        print("\nEpoch %d\n-------" % (episodio + 1))

        juego.new_episode()

        for i in trange(learning_steps_per_epoch, leave=False):
            estado = juego.get_state()
            bufer_pantalla = preprocesar(estado.screen_buffer)
            accion = agente.elegir_accion(bufer_pantalla)
            recompensa = juego.make_action(acciones[accion], frames_per_action)
            terminado = juego.is_episode_finished()

            if not terminado:
                bufer_pantalla_sig = preprocesar(juego.get_state().screen_buffer)
            else:
                bufer_pantalla_sig = tf.zeros(shape=bufer_pantalla.shape)

            if terminado:
                puntuaciones_entrenamiento.append(juego.get_total_reward())

                juego.new_episode()

            memoria_replay.append((bufer_pantalla, accion, recompensa, bufer_pantalla_sig, terminado))

            if i >= batch_size:
                agente.entrenar_dqn(obtener_muestras(memoria_replay))

            if (i % target_net_update_steps) == 0:
                agente.actualizar_red_objetivo()

        puntuaciones_entrenamiento = np.array(puntuaciones_entrenamiento)
        print(
            "Results: mean: {:.1f}±{:.1f},".format(
                puntuaciones_entrenamiento.mean(), puntuaciones_entrenamiento.std()
            ),
            "min: %.1f," % puntuaciones_entrenamiento.min(),
            "max: %.1f," % puntuaciones_entrenamiento.max(),
        )

        probar(test_episodes_per_epoch, juego, agente)
        tiempo_transcurrido = time() - tiempo_inicio
        print("Total elapsed time: %.2f minutes" % (tiempo_transcurrido / 60.0))
    
    # Return total training time
    return time() - tiempo_inicio


def probar(episodios_prueba, juego, agente):
    puntuaciones_prueba = []

    print("\nTesting...")
    for episodio_prueba in trange(episodios_prueba, leave=False):
        juego.new_episode()
        while not juego.is_episode_finished():
            estado = preprocesar(juego.get_state().screen_buffer)
            indice_mejor_accion = agente.elegir_accion(estado)
            juego.make_action(acciones[indice_mejor_accion], frames_per_action)

        puntuacion = juego.get_total_reward()
        puntuaciones_prueba.append(puntuacion)

    puntuaciones_prueba = np.array(puntuaciones_prueba)
    print(
        f"Results: mean: {puntuaciones_prueba.mean():.1f}±{puntuaciones_prueba.std():.1f},",
        "min: %.1f" % puntuaciones_prueba.min(),
        "max: %.1f" % puntuaciones_prueba.max(),
    )


class DQN(Model):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = Sequential(
            [
                Conv2D(8, kernel_size=6, strides=3, input_shape=(30, 45, 1)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv2 = Sequential(
            [
                Conv2D(8, kernel_size=3, strides=2, input_shape=(9, 14, 8)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.flatten = Flatten()

        self.state_value = Dense(1)
        self.advantage = Dense(num_actions)

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


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup device based on command line argument
    DEVICE = setup_device(args.device)
    
    agente = DQNAgent()
    juego = inicializar_juego(ventana_visible=args.watch_training)
    memoria_replay = deque(maxlen=replay_memory_size)

    n = juego.get_available_buttons_size()
    acciones = [list(a) for a in it.product([0, 1], repeat=n)]

    # Variable to store training time
    tiempo_entrenamiento = 0.0

    with tf.device(DEVICE):

        if not skip_learning:
            print("Starting the training for Health Gathering scenario!")
            print("Goal: Learn to collect medikits and survive!")

            tiempo_entrenamiento = ejecutar(agente, juego, memoria_replay)

            juego.close()
            print("======================================")
            print("Training is finished.")
            print(f"Total training time: {tiempo_entrenamiento:.2f} seconds ({tiempo_entrenamiento/60:.2f} minutes)")

            if save_model:
                agente.dqn.save(model_savefolder)

        juego.close()

        if watch:
            juego.set_window_visible(True)
            juego.set_mode(vzd.Mode.ASYNC_PLAYER)
            juego.init()

            print("\nWatching trained agent collect medikits...")
            for _ in range(episodes_to_watch):
                juego.new_episode()
                while not juego.is_episode_finished():
                    estado_juego = juego.get_state()
                    assert estado_juego is not None
                    estado = preprocesar(estado_juego.screen_buffer)
                    indice_mejor_accion = agente.elegir_accion(estado)

                    # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                    juego.set_action(acciones[indice_mejor_accion])
                    for _ in range(frames_per_action):
                        juego.advance_action()

                # Sleep between episodes
                sleep(1.0)
                puntuacion = juego.get_total_reward()
                print("Total score: ", puntuacion)

