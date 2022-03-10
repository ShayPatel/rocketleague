from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
import rlgym.utils.common_values as common_values


class chase_ball_reward(RewardFunction):
    """
    This reward function incentivises the car to continually chase the ball.

    The reward is calculated in 2 components.
    The first is distance vector that is calculated as the normalized distance from the car to the ball.
    The distance reward is always negative to penalize the car for being further away from the ball.
    The second is the velocity reward, which is calculated as the dot product of the
    normalized velocity vector with the normalized distance vector.
    This reward incentivizes the car to move as fast as possible towards the ball.
    
    The reward is then velocity_reward + distance_penalty
    
    """
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        car_vector = player.car_data.position.copy()
        ball_vector = state.ball.position.copy()
        
        car_vector[0] = car_vector[0] /common_values.SIDE_WALL_X
        car_vector[1] = car_vector[1] /common_values.BACK_WALL_Y
        car_vector[2] = car_vector[2] /common_values.CEILING_Z
        ball_vector[0] = ball_vector[0] /common_values.SIDE_WALL_X
        ball_vector[1] = ball_vector[1] /common_values.BACK_WALL_Y
        ball_vector[2] = ball_vector[2] /common_values.CEILING_Z
        

        #this is not the same as the liu distance reward function
        distance_vector = car_vector - ball_vector
        distance = np.linalg.norm(distance_vector)
        #gives negative reward the further from the ball you are.
        distance_penalty = -distance



        vel = player.car_data.linear_velocity
        norm_pos_diff = distance_vector / distance
        vel /= common_values.CAR_MAX_SPEED
        velocity_reward = float(np.dot(norm_pos_diff, vel))

        total_reward = velocity_reward + distance_penalty
        return total_reward

class chase_ball_reward_corrected(RewardFunction):
    """
    This reward function incentivises the car to continually chase the ball.

    The reward is calculated in 2 components.
    The first is distance vector that is calculated as the normalized distance from the car to the ball.
    The distance reward is always negative to penalize the car for being further away from the ball.
    The second is the velocity reward, which is calculated as the dot product of the
    normalized velocity vector with the normalized distance vector.
    This reward incentivizes the car to move as fast as possible towards the ball.
    
    The reward is then velocity_reward + distance_penalty
    
    """
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        car_vector = player.car_data.position.copy()
        ball_vector = state.ball.position.copy()
        
        car_vector[0] = car_vector[0] /common_values.SIDE_WALL_X
        car_vector[1] = car_vector[1] /common_values.BACK_WALL_Y
        car_vector[2] = car_vector[2] /common_values.CEILING_Z
        ball_vector[0] = ball_vector[0] /common_values.SIDE_WALL_X
        ball_vector[1] = ball_vector[1] /common_values.BACK_WALL_Y
        ball_vector[2] = ball_vector[2] /common_values.CEILING_Z
        

        #this is not the same as the liu distance reward function
        distance_vector =  ball_vector - car_vector
        distance = np.linalg.norm(distance_vector)


        vel = player.car_data.linear_velocity
        norm_pos_diff = distance_vector / distance
        vel /= common_values.CAR_MAX_SPEED
        velocity_reward = float(np.dot(norm_pos_diff, vel))

        total_reward = velocity_reward - distance
        return total_reward