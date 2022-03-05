import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.state_setters import random_state
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs


#game speed to train with
#Set to 100 when the agent is fully programmed to train.
#Set to 1 to see the game played in real time
game_speed = 1
#maximum number of steps to run 
max_steps = 200
#randomly sets the position of the ball and environment on reset


env = rlgym.make(
    game_speed=game_speed,
    terminal_conditions=[
        TimeoutCondition(max_steps=max_steps),
        GoalScoredCondition()
    ],
    obs_builder=AdvancedObs()
)




def main():
    obs = env.reset()

    print("metadata")
    print(env.metadata)
    while True:
        #obs = env.reset()
        done = False
        while not done:
            #action = [0]*8
            #action[0] = 0.3
            #action[1] = 1
            action = env.action_space.sample()
            state, reward, done, gameinfo = env.step(action)

            print("action")
            print(action)

            print("state")
            print(state)

            print("reward")
            print(reward)

            print("gameinfo")
            print(gameinfo)



def test():
    for i in range(5):
        obs = env.reset()
        print("reset")

        for i in range(5):
            action = env.action_space.sample()
            state, reward, done, gameinfo = env.step(action)

            print("action")
            print(action)

            print("state")
            print(state)

            print("reward")
            print(reward)

            print("gameinfo")
            print(gameinfo)


if __name__ == "__main__":
    main()
    #test()