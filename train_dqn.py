import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation
import numpy as np
import torch

from agents.deep_q_agent import DeepQAgent

def main():
    ENV_NAME = "ALE/Pong-v5"
    NUM_EPISODES = 10_000
    RENDER = False  # rendering is slow; use human only for debugging
    MODEL_TYPE = "DQN"
    EPSILON = 1.0

    env = gym.make(ENV_NAME, render_mode="human" if RENDER else "rgb_array", frameskip=1)  # "human" for manual viewing
    # Preprocess to 84x84 grayscale, skip frames, then stack 4 frames
    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,
        noop_max=30,
        terminal_on_life_loss=False,
        scale_obs=False,   # keep 0..255, we’ll divide by 255. later
    )
    env = FrameStackObservation(env, stack_size=4)  # returns HWC: (84,84,4)
    env = TransformObservation(
        env,
        lambda o: np.asarray(o, dtype=np.float32) / 255.0,
        observation_space=gym.spaces.Box(
            low=0, high=1, shape=(4, 84, 84), dtype=np.float32
        ),
    )
    print("Obsv space is:", env.observation_space.shape)  # (210,160,3)

    agent = DeepQAgent(
        n_actions=env.action_space.n,
        input_dims=env.observation_space.shape,  # HWC; agent must permute to CHW before Conv2d
        env_name=ENV_NAME,
        epsilon=EPSILON,
    )

    scores, eps_history, steps_array = [], [], []
    n_steps, best_score = 0, -np.inf

    print("Commencing DQN training...")
    for i in range(NUM_EPISODES):
        state, info = env.reset()
        score, done = 0.0, False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            
            if RENDER:
                env.render()
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 1 else score

        print(f"episode: {i} | score: {score:.1f} | avg(100): {avg_score:.1f} "
              f"| best: {best_score:.1f} | epsilon: {agent.epsilon:.2f} | steps: {n_steps}")

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()  # if/when implemented

        eps_history.append(agent.epsilon)

if __name__ == "__main__":
    main()