import numpy as np
import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.state_setters import random_state
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.obs_builders.default_obs import DefaultObs
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward,LiuDistancePlayerToBallReward, FaceBallReward

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import PPO


def make_env(game_speed:int, terminal_conditions=None, obs_builder=DefaultObs(), reward_function=LiuDistancePlayerToBallReward()):
    
    env = rlgym.make(
        game_speed=game_speed,
        terminal_conditions=terminal_conditions,
        obs_builder=obs_builder,
        reward_fn=reward_function
    )

    return env


def test():
    for i in range(1):
        obs = env.reset()
        print("reset")

        for i in range(1):
            action = env.action_space.sample()
            state, reward, done, gameinfo = env.step(action)

            print("action")
            print(action)

            print("state")
            print(state)
            print(len(state))

            print("reward")
            print(reward)

            print("gameinfo")
            print(gameinfo)


def ddpg(model_name:str, game_speed:int, terminal_conditions=None, obs_builder=DefaultObs(), reward_function=LiuDistancePlayerToBallReward()):
    env = make_env(
        game_speed=game_speed,
        terminal_conditions=terminal_conditions,
        obs_builder=obs_builder,
        reward_function=reward_function
    )

    action_means = np.zeros(8)
    #action_means[-3:] = 0.5
    sigma = 0.1 * np.ones(8)

    action_noise = NormalActionNoise(mean=action_means,sigma=sigma)

    model = DDPG(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0001,
        buffer_size=1000000,
        learning_starts=100000,
        batch_size=128,
        gamma=0.99,
        train_freq=(5, 'step'),
        action_noise=None,
        verbose=2
    )

    model.learn(
        total_timesteps=1000000,
        log_interval=100
    )
    model.save(model_name)
    env.close()

def eval_ddpg(model_name:str, game_speed:int, terminal_conditions=None, obs_builder=DefaultObs(), reward_function=LiuDistancePlayerToBallReward()):
    env = make_env(
        game_speed=game_speed,
        terminal_conditions=terminal_conditions,
        obs_builder=obs_builder,
        reward_function=reward_function
    )

    model = DDPG.load(model_name)
    while True:
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)

            obs, reward, done, gameinfo = env.step(action)


if __name__ == "__main__":
    #game speed to train with
    #Set to 100 when the agent is fully programmed to train.
    #Set to 1 to see the game played in real time
    game_speed = 1
    #maximum number of steps to run 
    max_steps = 300
    #randomly sets the position of the ball and environment on reset

    #the advanced observation builder has a state of 76 values.
    #this one includes the distance between the car and the ball.
    obs_builder = AdvancedObs()

    #select the reward function to use
    reward_function = LiuDistancePlayerToBallReward()

    #give the conditions to indicate if done
    terminal_conditions = [
        TimeoutCondition(max_steps=max_steps),
        GoalScoredCondition()
    ]

    
    #name of the model. used to load or save
    model_name = "ddpg-player-to-ball-distance-final"


    #main()
    #test()
    #ddpg(model_name, game_speed, max_steps, obs_builder, reward_function)
    eval_ddpg(model_name, game_speed, max_steps, obs_builder, reward_function)