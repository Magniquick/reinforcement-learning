import numpy as np
from gym_uno import UnoEnv


def run_random_agent(players: int = 4, max_steps: int = 1000, render: bool = True):
    """
    Plays one full game with a random policy.

    Args:
        players: total number of players (including the random agent as player 0)
        max_steps: safety cap on steps to avoid infinite loops
        render: whether to print game state each turn

    Returns:
        total_steps: number of env steps taken
        final_reward: reward from the final step (+1 win, -1 loss)
    """
    env = UnoEnv(players=players)
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < max_steps:
        if render:
            env.render()

        # sample uniformly among legal actions
        legal = obs['legal_actions']
        valid_actions = np.nonzero(legal)[0]
        action = np.random.choice(valid_actions)
        print(env.id2card[action.item()] if render else "", end=' ')

        obs, reward, done, truncated, info = env.step(int(action))
        total_reward = reward  # reward only meaningful at end
        steps += 1

    if render:
        print(f"Finished in {steps} steps with reward {total_reward}\n")
    return steps, total_reward


if __name__ == "__main__":
    run_random_agent(players=4, render=True)
