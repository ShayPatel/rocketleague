import numpy as np
import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.state_setters.random_state import RandomState
from rlgym.utils.state_setters.default_state import DefaultState
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.obs_builders.default_obs import DefaultObs
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward,LiuDistancePlayerToBallReward, FaceBallReward
from rlgym.utils.action_parsers.default_act import DefaultAction
from utils.action_parser import simple_action
from utils.rewards import chase_ball_reward, chase_ball_reward_corrected, chase_ball_and_score_reward

from stable_baselines3 import DDPG



def make_env(game_speed:int, action_parser=DefaultAction(), terminal_conditions=None, obs_builder=DefaultObs(), reward_function=LiuDistancePlayerToBallReward(), state_setter=DefaultState()):
    if not terminal_conditions:
        env = rlgym.make(
            game_speed=game_speed,
            action_parser=action_parser,
            terminal_conditions=[TimeoutCondition(400),GoalScoredCondition()],
            obs_builder=obs_builder,
            reward_fn=reward_function,
            state_setter=state_setter
        )
    else:
        env = rlgym.make(
            game_speed=game_speed,
            action_parser=action_parser,
            terminal_conditions=terminal_conditions,
            obs_builder=obs_builder,
            reward_fn=reward_function,
            state_setter=state_setter
        )

    return env


def eval_ddpg(model_name:str,env):

    model = DDPG.load(model_name)
    while True:
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)

            obs, reward, done, gameinfo = env.step(action)




if __name__ == "__main__":
    
    #name of the model. used to load or save
    #uncomment the model you want to use
    #model_name = "ddpg-player-to-ball-distance-final"
    #model_name = "ddpg-ball-to-goal-velocity"
    #model_name = "ddpg-chase-ball"
    #model_name = "ddpg-chase-ball-corrected"
    #model_name = "ddpg-chase-ball_and_score"
    model_name = "ddpg-chase-ball_and_score-weight-10"




    #game speed to train with
    #Set to 100 when the agent is fully programmed to train.
    #Set to 1 to see the game played in real time
    game_speed = 1
    #maximum number of steps to run 
    #max_steps = 300
    #randomly sets the position of the ball and environment on reset

    #the advanced observation builder has a state of 76 values.
    #this one includes the distance between the car and the ball.
    obs_builder = AdvancedObs()




    if model_name == "ddpg-player-to-ball-distance-final":
        max_steps = 300
        reward_function = LiuDistancePlayerToBallReward()
        state_setter = DefaultState()
        action_parser = DefaultAction()
    elif model_name == "ddpg-ball-to-goal-velocity":
        max_steps = 300
        reward_function = VelocityBallToGoalReward()
        state_setter = DefaultState()
        action_parser = simple_action()
    elif model_name == "ddpg-chase-ball-corrected":
        max_steps = 500
        reward_function = chase_ball_reward_corrected()
        state_setter = RandomState(ball_rand_speed=True, cars_rand_speed=True, cars_on_ground=True)
        action_parser = simple_action()
    elif model_name == "ddpg-chase-ball":
        max_steps = 500
        reward_function = chase_ball_reward()
        state_setter = RandomState(ball_rand_speed=True, cars_rand_speed=True, cars_on_ground=True)
        action_parser = simple_action()
    elif model_name == "ddpg-chase-ball_and_score":
        max_steps = 500
        reward_function = chase_ball_and_score_reward(weight=1)
        state_setter = RandomState(ball_rand_speed=True, cars_rand_speed=True, cars_on_ground=True)
        action_parser = simple_action()
    elif model_name == "ddpg-chase-ball_and_score-weight-10":
        max_steps = 500
        reward_function = chase_ball_and_score_reward(weight=10)
        state_setter = RandomState(ball_rand_speed=True, cars_rand_speed=True, cars_on_ground=True)
        action_parser = simple_action()


    #give the conditions to indicate if done
    terminal_conditions = [
        TimeoutCondition(max_steps=max_steps),
        GoalScoredCondition()
    ]

    env = make_env(
        game_speed=game_speed,
        action_parser=action_parser,
        terminal_conditions=terminal_conditions,
        obs_builder=obs_builder,
        reward_function=reward_function,
        state_setter=state_setter
    )
    
    
    eval_ddpg(model_name, env)