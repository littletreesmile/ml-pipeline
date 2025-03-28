import yaml


class Config:
    """
    Custom class to access dictionary-like data via dot notation.
    """

    def __init__(self, dictionary):
        # Convert dictionary keys to attributes with dot notation
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively convert sub-dictionaries
                value = Config(value)
            setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)


def load_config(path="config.yaml"):
    """
    Load YAML configuration file and return it as a Config object.
    """
    with open(path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)


# Load the config
config_path = "../config.yaml"
config = load_config(path=config_path)
