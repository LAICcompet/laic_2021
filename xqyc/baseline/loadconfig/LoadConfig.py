from utils.CommonTools import YamlLoad


class LoadConfig:
    filename = "./config/application.yml"
    yaml_load = YamlLoad(filename)
    config = yaml_load.get_config()
