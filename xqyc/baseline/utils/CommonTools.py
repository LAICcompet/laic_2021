import yaml
import os
import numpy as np


class YamlLoad:
    def __init__(self, filename):
        self.filename = filename

    def get_config(self):
        with open(self.filename, 'r', encoding="utf-8") as f:
            yml_data = f.read()
            # load方法转出为字典类型
            data = yaml.load(yml_data)
        return data
