import numpy as np
import gymnasium as gym
from gymnasium import spaces
from random import choice

from uno.game import UnoGame
from uno.const import COLORS, COLOR_CARD_TYPES, BLACK_CARD_TYPES


class UnoEnv(gym.Env):
    """
    Gymnasium environment wrapper for the UnoGame.
    The agent is always player 0. Other players are controlled by a simple rule-based policy.
    Observation:
        - hand: array of size max_hand containing encoded card IDs (padded)
        - hand_mask: binary mask indicating valid positions in `hand`
        - current_card: integer encoding of the top card
        - legal_actions: binary mask over actions (play positions + draw)
    Action:
        Discrete(max_hand + 1), where 0..max_hand-1 = play card at index, max_hand = draw
    Reward:
        +1 for winning, -1 for illegal play or losing, 0 otherwise
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, players: int = 2, max_hand: int = 108):
        super().__init__()
        if not 2 <= players <= 15:
            raise ValueError("Number of players must be between 2 and 15")
        self.num_players = players
        self.max_hand = max_hand
        self._build_card_mapping()

        # spaces
        n_cards = len(self.card2id)
        self.observation_space = spaces.Dict({
            "hand": spaces.MultiDiscrete([n_cards] * self.max_hand),
            "hand_mask": spaces.MultiBinary(self.max_hand),
            "current_card": spaces.Discrete(n_cards),
            "legal_actions": spaces.MultiBinary(self.max_hand + 1),
        })
        self.action_space = spaces.Discrete(self.max_hand + 1)

        self.reset()

    def _build_card_mapping(self):
        # create mapping from (color, type) -> unique ID
        all_cards = [(c, t) for c in COLORS for t in COLOR_CARD_TYPES] + [('black', t) for t in BLACK_CARD_TYPES]
        self.card2id = {card: idx for idx, card in enumerate(all_cards)}
        self.id2card = {idx: card for card, idx in self.card2id.items()}


    def reset(self, *, seed = None, options=None):
        super().reset(seed=seed)
        self.game = UnoGame(self.num_players)
        # ensure first card is not black without color
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        # wait until it's agent's turn
        while self.game.current_player.player_id != 0 and self.game.is_active:
            self._opponent_step()

        reward = 0
        done = False
        info = {}
        agent = self.game.players[0]

        # perform action
        if action < len(agent.hand):
            card = agent.hand[action]
            if self.game.current_card.playable(card):
                new_color = choice(COLORS) if card.color == 'black' else None
                self.game.play(0, action, new_color)
            else:
                reward = -1
                self.game.play(0, card=None)
        else:
            # draw
            self.game.play(0, card=None)

        # simulate opponents until agent's turn or end
        while self.game.current_player.player_id != 0 and self.game.is_active:
            self._opponent_step()

        done = not self.game.is_active
        if done:
            reward = 1 if self.game.winner.player_id == 0 else -1

        obs = self._get_obs()
        return obs, reward, done, False, info

    def _opponent_step(self):
        player = self.game.current_player
        if player.can_play(self.game.current_card):
            for idx, card in enumerate(player.hand):
                if self.game.current_card.playable(card):
                    new_color = choice(COLORS) if card.color == 'black' else None
                    self.game.play(player.player_id, idx, new_color)
                    return
        # otherwise draw
        self.game.play(player.player_id, card=None)

    def _get_obs(self):
        # encode hand
        hand = self.game.players[0].hand
        print(hand)
        ids = [self.card2id[(c.color, c.card_type)] for c in hand]
        hand_arr = np.zeros(self.max_hand, dtype=int)
        mask = np.zeros(self.max_hand, dtype=int)
        for i, cid in enumerate(ids):
            hand_arr[i] = cid
            mask[i] = 1

        # encode current card
        top = self.game.current_card
        cur_id = self.card2id[(top.color, top.card_type)]

        # legal actions mask
        legal = np.zeros(self.max_hand + 1, dtype=int)
        for i, c in enumerate(hand):
            if self.game.current_card.playable(c):
                legal[i] = 1
        legal[self.max_hand] = 1  # draw always allowed

        return {"hand": hand_arr, "hand_mask": mask, "current_card": cur_id, "legal_actions": legal}

    def render(self, mode='human'):
        print(f"Current card: {self.game.current_card}")
        print(f"Your hand: {', '.join(str(i) for i in self.game.players[0].hand)}")

    def close(self):
        pass
