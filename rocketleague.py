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
from utils.rewards import chase_ball_reward

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import PPO


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

def test(env, model_name:str=None):
    if model_name:
        model = DDPG.load(model_name)
    else:
        model = PPO("MlpPolicy", env=env, verbose=1)

    for i in range(20):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)

            obs, reward, done, gameinfo = env.step(action)
            #print("reward", reward)
            #player = gameinfo["state"].players[0]
            #print("position",player.car_data.position)
            #print("velocity",player.car_data.linear_velocity)
            #print("angular",player.car_data.angular_velocity)


    env.close()


def ddpg(model_name:str, env, training_timesteps:int):
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
        total_timesteps=training_timesteps,
        log_interval=100
    )
    #create a stable baselines model directory
    model.save(model_name)

    #close the game after training
    env.close()

def eval_ddpg(model_name:str,env):

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
    max_steps = 500
    #randomly sets the position of the ball and environment on reset

    #the advanced observation builder has a state of 76 values.
    #this one includes the distance between the car and the ball.
    obs_builder = AdvancedObs()

    #select the reward function to use
    #reward_function = LiuDistancePlayerToBallReward()
    #reward_function = VelocityBallToGoalReward()
    reward_function = chase_ball_reward()

    #give the conditions to indicate if done
    terminal_conditions = [
        TimeoutCondition(max_steps=max_steps),
        GoalScoredCondition()
    ]

    #select the action parser
    #action_parser = DefaultAction()
    action_parser = simple_action()

    #defines the state at each restart
    #state_setter = DefaultState()
    state_setter = RandomState(
        ball_rand_speed=True,
        cars_rand_speed=True,
        cars_on_ground=True
    )

    env = make_env(
        game_speed=game_speed,
        action_parser=action_parser,
        terminal_conditions=terminal_conditions,
        obs_builder=obs_builder,
        reward_function=reward_function,
        state_setter=state_setter
    )
    
    #name of the model. used to load or save
    #model_name = "ddpg-ball-to-goal-velocity"
    #model_name = "ddpg-player-to-ball-distance-final"
    model_name = "ddpg-chase-ball-simple-action"


    #main()
    #test(env,model_name)
    #ddpg(model_name, env, 4000000)
    eval_ddpg(model_name, env)