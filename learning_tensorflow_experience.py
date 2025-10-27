#!/usr/bin/env python3

# M. Kempka, T.Sternal, M.Wydmuch, Z.Boztoprak
# Updated for Python 3.13 and TensorFlow 2.15+
# January 2021 - Updated 2024
#
# NEW FEATURES:
# - Prioritized Experience Replay [Schaul et al., 2016]
# - Command line arguments for device selection:
#   Usage: python learning_tensorflow_experience.py --device [auto|gpu|cpu]
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
num_train_epochs = 5
learning_steps_per_epoch = 2000
target_net_update_steps = 1000

# NN learning settings
batch_size = 64

# Prioritized Experience Replay parameters
priority_alpha = 0.6  # How much prioritization is used (0 = uniform, 1 = full priority)
priority_beta = 0.4   # Importance sampling exponent (linearly annealed to 1)
priority_beta_annealing = 1.0
priority_epsilon = 1e-6  # Small positive value to prevent zero priorities

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

# Configuration file path
config_file_path = os.path.join(vzd.scenarios_path, "simpler_basic.cfg")
model_savefolder = "./model_prioritized.keras"  # Use .keras extension for modern TensorFlow


def parse_arguments():
    """Parse command line arguments for device selection."""
    parser = argparse.ArgumentParser(
        description="ViZDoom Deep Q-Network Training with Prioritized Experience Replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python learning_tensorflow_experience.py --device gpu
  python learning_tensorflow_experience.py --device cpu
  python learning_tensorflow_experience.py --device auto
        """
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Device to use for training: 'auto' (use GPU if available), 'gpu', or 'cpu' (default: auto)"
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


def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=-1)

    return tf.stack(img)


def initialize_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")

    return game


class PriorityReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Implements proportional prioritization as described in [Schaul et al., 2016]
    """
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_annealing=1.0, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_annealing = beta_annealing  # How much to increase beta per step
        self.epsilon = epsilon  # Small constant to avoid zero priorities
        
        self.buffer = []
        self.priorities = []
        self.beta_step = 0.0  # Track beta annealing
        
    def add(self, experience, td_error=None):
        """Add experience to buffer with a priority."""
        if td_error is None:
            # If no TD error provided, use the maximum priority (for new experiences)
            priority = 1.0
        else:
            # Convert TD error to priority: p = |td_error|^alpha
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            
        self.buffer.append(experience)
        self.priorities.append(priority)
        
        # Remove oldest experiences if buffer is full
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
    
    def sample(self, batch_size):
        """Sample batch of experiences based on priority probabilities."""
        if len(self.buffer) < batch_size:
            sample_size = len(self.buffer)
        else:
            sample_size = batch_size
            
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), size=sample_size, p=probabilities)
        
        # Get experiences and their priorities
        experiences = [self.buffer[i] for i in indices]
        probs = probabilities[indices]
        
        # Compute importance sampling weights
        # w_i = (N * P(i))^(-beta)
        N = len(self.buffer)
        weights = (N * probs) ** (-self.beta)
        
        # Normalize weights by 1 / max(w_i)
        weights = weights / weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities for sampled experiences based on new TD errors."""
        for i, td_error in zip(indices, td_errors):
            # Convert TD error to priority
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[i] = priority
    
    def update_beta(self):
        """Anneal beta linearly towards priority_beta_annealing."""
        if self.beta < self.beta_annealing:
            self.beta = min(self.beta_annealing, self.beta + 0.00001)
        self.beta_step += 1


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
            self.target_net = DQN(self.num_actions)

    def update_target_net(self):
        self.target_net.set_weights(self.dqn.get_weights())

    def choose_action(self, state):
        if self.epsilon < np.random.uniform(0, 1):
            action = int(tf.argmax(self.dqn(tf.reshape(state, (1, 30, 45, 1))), axis=1))
        else:
            action = np.random.choice(range(self.num_actions), 1)[0]

        return action

    def train_dqn(self, samples, indices=None, weights=None):
        """Train DQN with optional importance sampling weights."""
        screen_buf, actions, rewards, next_screen_buf, dones = split_tuple(samples)

        row_ids = list(range(screen_buf.shape[0]))

        ids = extractDigits(row_ids, actions)
        done_ids = extractDigits(np.where(dones)[0])

        td_errors_list = []
        
        with tf.GradientTape() as tape:
            # No need to watch variables explicitly in modern TensorFlow
            Q_prev = tf.gather_nd(self.dqn(screen_buf), ids)

            Q_next = self.target_net(next_screen_buf)
            Q_next = tf.gather_nd(
                Q_next,
                extractDigits(row_ids, tf.argmax(self.dqn(next_screen_buf), axis=1)),
            )

            q_target = rewards + self.discount_factor * Q_next

            if len(done_ids) > 0:
                done_rewards = tf.gather_nd(rewards, done_ids)
                q_target = tf.tensor_scatter_nd_update(
                    tensor=q_target, indices=done_ids, updates=done_rewards
                )

            # Calculate TD errors for each sample
            td_errors = q_target - Q_prev
            
            # Store TD errors
            td_errors_list = td_errors.numpy()
            
            # Use importance sampling weights if provided
            if weights is not None:
                weights_tf = tf.constant(weights, dtype=tf.float32)
                weights_tf = tf.reshape(weights_tf, (-1, 1))
                
                # Weighted MSE loss
                per_sample_loss = tf.keras.losses.MeanSquaredError(reduction='none')(q_target, Q_prev)
                weighted_loss = tf.reduce_mean(per_sample_loss * weights_tf)
                td_error = weighted_loss
            else:
                # Use modern loss API
                td_error = tf.keras.losses.MeanSquaredError()(q_target, Q_prev)
                
        gradients = tape.gradient(td_error, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.dqn.trainable_variables))
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        
        # Return TD errors for priority updates
        return td_errors_list


def split_tuple(samples):
    samples = np.array(samples, dtype=object)
    screen_buf = tf.stack(samples[:, 0])
    actions = samples[:, 1]
    rewards = tf.stack(samples[:, 2])
    next_screen_buf = tf.stack(samples[:, 3])
    dones = tf.stack(samples[:, 4])
    return screen_buf, actions, rewards, next_screen_buf, dones


def extractDigits(*argv):
    if len(argv) == 1:
        return list(map(lambda x: [x], argv[0]))

    return list(map(lambda x, y: [x, y], argv[0], argv[1]))


def get_samples(memory):
    """Sample from replay buffer with priorities."""
    if len(memory.buffer) < batch_size:
        sample_size = len(memory.buffer)
    else:
        sample_size = batch_size

    samples, indices, weights = memory.sample(sample_size)
    return samples, indices, weights


def run(agent, game, replay_memory):
    time_start = time()

    for episode in range(num_train_epochs):
        train_scores = []
        print("\nEpoch %d\n-------" % (episode + 1))

        game.new_episode()

        for i in trange(learning_steps_per_epoch, leave=False):
            state = game.get_state()
            screen_buf = preprocess(state.screen_buffer)
            action = agent.choose_action(screen_buf)
            reward = game.make_action(actions[action], frames_per_action)
            done = game.is_episode_finished()

            if not done:
                next_screen_buf = preprocess(game.get_state().screen_buffer)
            else:
                next_screen_buf = tf.zeros(shape=screen_buf.shape)

            if done:
                train_scores.append(game.get_total_reward())

                game.new_episode()

            # Add experience with initial priority
            replay_memory.add((screen_buf, action, reward, next_screen_buf, done))

            if i >= batch_size:
                # Sample with priorities
                samples, indices, weights = get_samples(replay_memory)
                
                # Train and get TD errors
                td_errors = agent.train_dqn(samples, indices, weights)
                
                # Update priorities in replay buffer
                replay_memory.update_priorities(indices, td_errors)
                
                # Anneal beta
                replay_memory.update_beta()

            if (i % target_net_update_steps) == 0:
                agent.update_target_net()

        train_scores = np.array(train_scores)
        print(
            "Results: mean: {:.1f}±{:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        test(test_episodes_per_epoch, game, agent)
        elapsed_time = time() - time_start
        print("Total elapsed time: %.2f minutes" % (elapsed_time / 60.0))
    
    # Return total training time
    return time() - time_start


def test(test_episodes_per_epoch, game, agent):
    test_scores = []

    print("\nTesting...")
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.choose_action(state)
            game.make_action(actions[best_action_index], frames_per_action)

        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print(
        f"Results: mean: {test_scores.mean():.1f}±{test_scores.std():.1f},",
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
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
    
    agent = DQNAgent()
    game = initialize_game()
    
    # Create prioritized replay buffer instead of regular deque
    replay_memory = PriorityReplayBuffer(
        capacity=replay_memory_size,
        alpha=priority_alpha,
        beta=priority_beta,
        beta_annealing=priority_beta_annealing,
        epsilon=priority_epsilon
    )

    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Variable to store training time
    training_time = 0.0

    with tf.device(DEVICE):

        if not skip_learning:
            print("Starting the training with Prioritized Experience Replay!")
            print(f"Priority parameters: alpha={priority_alpha}, beta={priority_beta}")

            training_time = run(agent, game, replay_memory)

            game.close()
            print("======================================")
            print("Training is finished.")
            print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

            if save_model:
                agent.dqn.save(model_savefolder)
                print(f"Model saved to: {model_savefolder}")

        game.close()

        if watch:
            game.set_window_visible(True)
            game.set_mode(vzd.Mode.ASYNC_PLAYER)
            game.init()

            for _ in range(episodes_to_watch):
                game.new_episode()
                while not game.is_episode_finished():
                    game_state = game.get_state()
                    assert game_state is not None
                    state = preprocess(game_state.screen_buffer)
                    best_action_index = agent.choose_action(state)

                    # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                    game.set_action(actions[best_action_index])
                    for _ in range(frames_per_action):
                        game.advance_action()

                # Sleep between episodes
                sleep(1.0)
                score = game.get_total_reward()
                print("Total score: ", score)

