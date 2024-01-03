from .shooting_minigrid_env_v1 import ShootingMiniGridEnvV1
from .shooting_minigrid_env_v2 import ShootingMiniGridEnvV2


def _check_valid_version(env_version):
    valid_versions = ["ShootingMiniGrid-v1", "ShootingMiniGrid-v2"]
    assert env_version in valid_versions, f"env_version must be one of {valid_versions}"


def make_shooting_minigrid_env(env_version=None, **kwargs):
    """
        Create a custom environment based on the specified version.

        Parameters:
        - env_version (str): The version of the environment to create.
        - kwargs: every other argument that the environment needs
        Returns:
        - object: The instantiated custom environment.
        """
    env_version = env_version or "ShootingMiniGrid-v1"
    _check_valid_version(env_version)
    if env_version == "ShootingMiniGrid-v1":
        return ShootingMiniGridEnvV1(**kwargs)
    elif env_version == "ShootingMiniGrid-v2":
        return ShootingMiniGridEnvV2(**kwargs)
    else:
        raise ValueError(f"Unknown environment version: {env_version}")

