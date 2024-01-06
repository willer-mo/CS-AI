# ShootingMiniGrid-v3

| Model Name | Algorithm | Policy | Device | Training<br/>episodes | Training Time | Description                                                                                                    |
|------------|-----------|--------|--------|------------------|------------|----------------------------------------------------------------------------------------------------------------|
| PPO        |PPO|MlpPolicy|cuda|300000| 20m 18s| grid_size = 25, max_steps = 50, random positions, no walls                                                     |
| PPO_1      |PPO|MlpPolicy|cpu|500000| 16m 24s| grid_size = 25, max_steps = 50, random positions, no walls                                                     |
|PPO_2|PPO|MlpPolicy|cuda|2000000|3h 46m 35s| grid_size = 25, max_steps = 50, random positions, no walls                                                     |
|PPO_3|PPO|MlpPolicy|cuda|3000000|5h 12m 48s| grid_size = 25, max_steps = 100, random positions, random_walls = True                                         |
|PPO_4|PPO|MlpPolicy|cuda|100000| 7m 1s| grid_size = 5, max_steps = 10, random positions, random_walls = False                                          |
|PPO_5|PPO|MlpPolicy|cpu|200000| 6m 34s| grid_size = 25, max_steps = 10, random positions for target only, agent_start_pos=(1, 1), random_walls = False |
|PPO_6|PPO|MlpPolicy|cpu|500000| 15m 42s|grid_size = 25, max_steps = 50, random positions for target only, agent_start_pos=(1, 1), random_walls = True|
