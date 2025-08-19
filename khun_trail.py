# TODO: generalise to a multi-player setting
import gym_kuhn_poker
import torch
from DQN import Transition
import gymnasium as gym
import numpy as np
import copy
from gym_kuhn_poker.envs.kuhn_poker_env import ObsDict
from typing import Callable, Sequence

env = gym.make("KuhnPoker-v0")


def _get_action_fn_helper(obs: ObsDict, device: torch.device) -> torch.Tensor:
    """does preprocessing (sqeezing and other formatting to use the action fn)"""
    state = torch.tensor(gym.spaces.flatten(env.observation_space, obs), 
                        dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension
    return state


def get_trajectory(
    action_fn: Callable[[gym.Env, torch.Tensor], torch.Tensor], device: torch.device = torch.device("cpu")
) -> Sequence[list[Transition]]:
    """
    Get a full trajectory for each player for a given game.
    """
    obs, _ = env.reset()
    done = False
    truncated = False

    # Per-player pending (state, action) until their next turn
    pending: dict[int, tuple[ObsDict, int]] = {}  # pid -> {"state": obs_dict, "action": int}

    # Separate trajectories per player
    traj_p0: list[Transition] = []
    traj_p1: list[Transition] = []
    trajectories = (traj_p0, traj_p1)

    # Track last transition index per player (within that player's own trajectory list)
    last_idx_for_player = {}  # pid -> idx in trajectories[pid]

    print("\n=== Playing moves ===")

    current_obs = copy.deepcopy(obs)  # belongs to player_id who is about to act
    final_payouts = None

    while not (done or truncated):
        env.render()

        pid = int(current_obs["player_id"])  # 0 or 1

        # If this player had a previous pending action, finalize it now
        if pid in pending:
            pa = pending.pop(pid)
            # Append to THIS player's own trajectory
            my_traj = trajectories[pid]
            idx = len(my_traj)
            my_traj.append(
                Transition(
                    state=pa[0],
                    action=pa[1],
                    next_state=copy.deepcopy(current_obs),
                    reward=0.0,  # will be set on the player's last transition at terminal
                )
            )
            last_idx_for_player[pid] = idx

        # Choose action
        action = action_fn(env, _get_action_fn_helper(current_obs, device)).item()

        # Step env
        next_obs, reward_scalar, done, truncated, info = env.step(action)
        print(f"current observation (next to act): {next_obs}")
        print(f"acting player {pid} env-scalar reward (ignored until terminal): {reward_scalar}")

        if done or truncated:
            terminal_obs = copy.deepcopy(next_obs)
            final_payouts = info.get("rewards", None)

            # Finalize the CURRENT player's action into their own trajectory
            my_traj = trajectories[pid]
            idx = len(my_traj)
            my_traj.append(
                Transition(
                    state=copy.deepcopy(current_obs),
                    action=int(action),
                    next_state=terminal_obs,
                    reward=0.0,  # set later
                )
            )
            last_idx_for_player[pid] = idx

            # Finalize any other players still pending, into their own trajectories
            for rem_pid, pa in list(pending.items()):
                rem_pid = int(rem_pid)
                r_traj = trajectories[rem_pid]
                ridx = len(r_traj)
                r_traj.append(
                    Transition(
                        state=pa[0],
                        action=pa[1],
                        next_state=copy.deepcopy(terminal_obs),
                        reward=0.0,  # set later
                    )
                )
                last_idx_for_player[rem_pid] = ridx
            pending.clear()
            break

        # Not terminal: stash this player's (state, action) for next turn
        pending[pid] = (
            copy.deepcopy(current_obs),
            int(action),
        )

        # Advance loop
        current_obs = copy.deepcopy(next_obs)

    env.render()

    # --- Post-fill terminal rewards on each player's LAST transition ---
    assert final_payouts is not None, "Final payouts vector should not be None"
    for p in (0, 1):
        if p in last_idx_for_player and len(trajectories[p]) > 0:
            i = last_idx_for_player[p]
            t = trajectories[p][i]
            trajectories[p][i] = Transition(
                state=t.state,
                action=t.action,
                next_state=t.next_state,
                reward=float(final_payouts[p]),
            )

    print("\nGame over!")
    print(f"Final payouts (per player): {final_payouts}")
    winner = int(np.argmax(final_payouts))
    print(f"Winner is player {winner} (net {final_payouts[winner]} chips).")

    print(f"\nPlayer 0: {len(traj_p0)} transitions")
    for i, t in enumerate(traj_p0):
        pid_s = int(t.state["player_id"]) if t.state is not None else None
        pid_ns = int(t.next_state["player_id"]) if t.next_state is not None else None
        print(f"  P0[{i}]: pid={pid_s} -> pid_next={pid_ns}, action={t.action}, reward={t.reward}")
        print(t)

    print(f"\nPlayer 1: {len(traj_p1)} transitions")
    for i, t in enumerate(traj_p1):
        pid_s = int(t.state["player_id"]) if t.state is not None else None
        pid_ns = int(t.next_state["player_id"]) if t.next_state is not None else None
        print(f"  P1[{i}]: pid={pid_s} -> pid_next={pid_ns}, action={t.action}, reward={t.reward}")
        print(t)

    return trajectories
