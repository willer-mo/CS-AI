import os


def set_scaffolding(env_name=None, algorithm_name=None, policy=None, device=None):
    check_inputs(env_name=env_name, algorithm_name=algorithm_name, policy=policy, device=device)
    models_dir = f"../models/{env_name}"
    logdir = f"../logs/{env_name}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    model_names = os.listdir(models_dir)
    suffix = len(
        [
            model for model in model_names
            if os.path.isdir(os.path.join(models_dir, model)) and algorithm_name in model
        ]
    )
    model_name = f"{algorithm_name}{'_' + str(suffix) if suffix else ''}"
    models_dir = os.path.join(models_dir, model_name)
    os.makedirs(models_dir)
    return logdir, models_dir, model_name


def write_info_file(env_name=None, algorithm_name=None, model_name=None, policy=None, device=None, episodes=None, models_dir=None, description=None, time_elapsed=None):
    check_inputs(env_name=env_name, algorithm_name=algorithm_name, model_name=model_name, policy=policy, device=device, episodes=episodes, models_dir=models_dir, description=description, time_elapsed=time_elapsed)
    hours, remainder = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    readme_path = f"{models_dir}/../Readme.md"
    try:
        file_exists = os.path.exists(readme_path)
        with open(readme_path, 'a') as file:
            if not file_exists:
                title = (f"# {env_name}\n\n"
                         "| Model Name | Algorithm | Policy | Device | Training<br/>episodes | Training Time | Description |\n"
                         "|------------|-----------|--------|--------|------------------|------------|-------------|\n")
                file.write(title)
            text = f"|{model_name}|{algorithm_name}|{policy}|{device}|{episodes}|{str(int(hours)) + 'h' if hours else''} {str(int(minutes)) + 'm' if minutes else ''} {str(int(seconds)) + 's' if seconds else ''}|{description}|\n"
            file.write(text)
        print(f"Readme updated at '{readme_path}'.")
    except Exception as e:
        print(f"Error updating Readme file at '{readme_path}': {e}")
    return True


def check_inputs(**kwargs):
    for arg_name, arg_value in kwargs.items():
        if arg_value is None:
            raise ValueError(f"Parameter '{arg_name}' cannot be None.")
    