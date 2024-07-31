import yaml


def get_yaml(yaml_file):
    """
    For loading yaml files
    inputs
    yaml_file - string / path
    returns
    args - dictionary
    """
    if yaml_file == None:
        return None
    else:
        with open(yaml_file, 'r') as file:
            args = yaml.safe_load(file)
        return args


def get_vision_settings(args):
    yargs = get_yaml(args.yaml_path)
    for key, value in yargs['general_config'].items(): # copy yaml config into args
        setattr(args, key, value)
    for key, value in yargs['training_args'].items():
        setattr(args, key, value)
    return args, yargs, yargs['model_config']


if __name__ == '__main__':
    pass ### TODO