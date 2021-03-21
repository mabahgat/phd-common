import yaml
import pathlib


def __load_config():
    root_path = pathlib.Path(__file__).parent.absolute()
    with open("{}/../config.yaml".format(root_path), 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
        return Config(config_dict)


class Config:

    def __init__(self, config_dict):
        self._config_dict = config_dict
    
    def __getattribute__(self, key):
        if key == '_config_dict' or key not in self._config_dict:
            return super().__getattribute__(key)
        value = self._config_dict[key]
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def __getitem__(self, key):
        return self.__getattribute__(key)


global_config = __load_config()
