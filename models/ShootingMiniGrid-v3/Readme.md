# ShootingMiniGrid-v3

| Model Name | Algorithm | Policy | Device | Training<br/>episodes | Training Time | Description                                                                                                    |
|------------|-----------|--------|--------|-----------------------|---------------|----------------------------------------------------------------------------------------------------------------|
| PPO        |PPO|MlpPolicy|cuda| 300000                | 20m 18s       | grid_size = 25, max_steps = 50, random positions, no walls                                                     |
| PPO_1      |PPO|MlpPolicy|cpu| 500000                | 16m 24s       | grid_size = 25, max_steps = 50, random positions, no walls                                                     |
| PPO_2      |PPO|MlpPolicy|cuda| 2000000               | 3h 46m 35s    | grid_size = 25, max_steps = 50, random positions, no walls                                                     |
| PPO_3      |PPO|MlpPolicy|cuda| 3000000               | 5h 12m 48s    | grid_size = 25, max_steps = 100, random positions, random_walls = True                                         |
| PPO_4      |PPO|MlpPolicy|cuda| 100000                | 7m 1s         | grid_size = 5, max_steps = 10, random positions, random_walls = False                                          |
| PPO_5      |PPO|MlpPolicy|cpu| 200000                | 6m 34s        | grid_size = 25, max_steps = 10, random positions for target only, agent_start_pos=(1, 1), random_walls = False |
| PPO_6      |PPO|MlpPolicy|cpu| 500000                | 15m 42s       | grid_size = 25, max_steps = 50, random positions for target only, agent_start_pos=(1, 1), random_walls = True  |
| PPO_7      |PPO|MlpPolicy|cuda| 500000                | 40m 15s       | grid_size = 25, max_steps = 10, random positions, random_walls = False                                         |
| PPO_10     |PPO|MlpPolicy|cuda| 1000000               | 1h 39m 21s    | grid_size = 25, max_steps = 5, random positions, random_walls = False                                          |
| PPO_11     |PPO|MlpPolicy|cuda| 1000000               | 1h 38m 22s    | grid_size = 25, max_steps = 50, random positions, random_walls = False                                         |
| PPO_12     |PPO|MlpPolicy|cuda| 2400000               | 2h 57m 59s    | grid_size = 25, max_steps = 50, random positions, random_walls = True                                          |
| PPO_13     |PPO|MlpPolicy|cuda| 3000000               | 3h 49m 38s    |grid_size = 25, max_steps = 50, random positions, random_walls = False|
| PPO_14     |PPO|MlpPolicy|cuda| 5000000               | 5h 52m 17s    |grid_size = 25, max_steps = 100, random positions, random_walls = False, static_walls = True|
| PPO_15     |PPO|MlpPolicy|cuda| 1000000               | 1h 29m 23s    |grid_size = 25, max_steps = 50, agent_position = (1, 12), target_position = (23, 23), random_walls = False, static_walls = True (only 3rd column)|
| PPO_16     |PPO|MlpPolicy|cuda| 3000000               | 3h 32m 47s    |grid_size = 25, max_steps = 50, agent_position = (1, 12), target_position = random, random_walls = False, static_walls = True (only 3rd column)|
| PPO_17     |PPO|MlpPolicy|cuda| 3000000               | 3h 29m 28s    |grid_size = 25, max_steps = 50, agent_position = (1, 12), target_position = random, random_walls = False, static_walls = True (all columns)|
| PPO_18     |PPO|MlpPolicy|cuda| 2200000               | 2h 32m 19s    |grid_size = 25, max_steps = 50, agent_position = (1, 12), target_position = random, random_walls = False, static_walls = True (all columns), rewarded by proximity of distance and angle|
|PPO_20|PPO|MlpPolicy|cuda|7000000|7h 53m 7s|grid_size = 25, max_steps = 50, agent_position = (1, 12), target_position = random, random_walls = False, static_walls = True (all columns)|
