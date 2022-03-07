import numpy as np
import gym.spaces
from rlgym.utils.gamestates import GameState
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils import common_values


class simple_action(ActionParser):
    def __init__(self):
        super().__init__()

    def get_action_space(self) -> gym.spaces.Space:
        min_values = np.array([-1,-1,-1,-1,-1,0,0,0])
        max_values = np.array([1,1,1,1,1,1,1,1])
        return gym.spaces.Box(min_values,max_values)

    def parse_actions(self, actions, state: GameState) -> np.ndarray:
        actions = actions.reshape((-1, 8))

        #clip the first 5 values to ensure they fall in the range -1 to 1
        actions[..., :5] = actions[..., :5].clip(-1, 1)

        #convert the last 3 elements to it's binary value.
        #the last 3 actions are jump, boost, and handbrake
        actions[...,-3:] = actions[...,3:].round()

        return actions