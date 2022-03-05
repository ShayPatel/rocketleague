


# Actions
The action is defined as a vector of length 8

0. throttle
    - [-1,1]
    - How fast the car should move forward or backward
1. steer
    - [-1,1]
    - left to right angle of the steering. -1 is all the way left. 1 is all the way right. 0 is stright.
2. yaw
    - [-1,1]
    - airborne yaw angle. -1 turn left fastest. 1 turn right fastest. 0 is straight.
3. pitch
    - [-1,1]
    - airborne pitch angle. -1 for backward. 1 for forward. 0 is straight.
4. roll
    - [-1,1]
    - airborne roll. -1 to roll left (counter-clockwise). 1 to roll right (clockwise). 0 for straight.
5. jump
    - 0 , 1
    - option to jump
6. boost
    - 0 , 1
    - option to use boost
7. handbrake
    - 0 , 1
    - option to use the handbrake



# Needed parameters
## Player object
The player is a PlayerData object.
- player.team_num
    - common_values.ORANGE_TEAM = 1
    - common_values.BLUE_TEAM = 0




# Flow

The environment is made of 3 classes
- Terminal condition
    - This class determines a condition to return the terminal state of the environment after an action has been taken. Multiple terminal conditions can be added to an environment
    - `reset(self, initial_state: GameState)`
        - Sets the initial attributes needed on the environment reset.
    - `is_terminal(self, current_state: GameState) -> bool`
        - read the current_state object and any class attributes to return if the environment is done. Returns the done variable in an env.step() call.
- Observation builder
- Action parser
    - use this class to edit the action space of the environment
    - `get_action_space(self) -> gym.spaces.Space`
        - use this function to creat the shape and type of the action space
    - `parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray`
        - use this function to change the how the environment takes in the action. it takes in a user defined action and converts it to the size 8 action vector expected by the environment.



# Troubleshoot
If the win32 dll fails to load, then install pywin32

```python
conda install pywin32
```

