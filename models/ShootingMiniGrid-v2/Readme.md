# ShootingMiniGrid-v2

| Model Name | Algorithm | Policy | Device | Training<br/>episodes | Training Time | Description |
|----------|-----------|--------|--------|------------------|------------|-------------|
|PPO|PPO|CnnPolicy|cpu|300000| 12m 41s|grid_size = 5, static position for agent and target|
|PPO_1|PPO|CnnPolicy|cpu|300000| 15m 18s|grid_size = 5, static position for agent and target|
|PPO_2|PPO|CnnPolicy|cuda|100000| 9m 49s|grid_size = 5, static position for agent and target, max_steps = 50|
