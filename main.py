import itertools

import gym_kuhn_poker
import gymnasium as gym
import torch

import DQN
import plotting

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

game = "KuhnPoker-v0"

env = gym.make(game)

observation, info = env.reset()

agent = DQN.DQNAgent(len(observation), env.action_space.n, device, load_model=True)  # type: ignore
plotter = plotting.plot()  # Rename to avoid confusion with plt

max_net_reward = 0.0
for episode in itertools.count():
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # Add batch dimension
    print(f"Episode {episode + 1} : ", end='', flush=False)
    episode_over = False
    net_reward = 0.0
    while not episode_over:
        action = agent.select_action(env, state)
        observation, reward, terminated, truncated, info = env.step(action.item())

        if terminated:
            next_state = None
            reward = -1.0  # Penalty for termination
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) # Add batch dimension

        net_reward += reward  # type: ignore
        agent.model_update(state, action, next_state, reward)

        state = next_state

        episode_over = terminated or truncated
    print(f"net_reward: {net_reward}")

    plotter.append_reward(net_reward)
    cum_reward = plotter.update_plot()  # Update the plot with the current episode's net reward

    if episode % 100 == 0 and max_net_reward < cum_reward:
        max_net_reward = cum_reward
        agent.save_state()
        print(f"New best model saved with net reward: {cum_reward}")
    if cum_reward >= 500:
        print(f"Training complete after {episode + 1} episodes with max net reward: {cum_reward}")
        break
plotter.finish_plot()  # Finalize the plot after training
plotter.close()
env.close()
agent.save_state()
# play an example 
env = gym.make(game, render_mode="human")
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

episode_over = False
while not episode_over:
        action = agent.greedy_select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        state = next_state
        episode_over = terminated or truncated
        if truncated:
            print("model succesful.")

env.close()
