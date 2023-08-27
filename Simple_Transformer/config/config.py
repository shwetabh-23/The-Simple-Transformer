import copy
import yaml
import os
class Config(dict):
    def __init__(self, d: dict) -> None:
        super().__init__(d)
        self.d = copy.deepcopy(d)
        for key, val in d.items():
            if isinstance(val, dict):
                val = Config(val)
            elif isinstance(val, list):
                val = [Config(x) if isinstance(x, dict) else x for x in val]
            setattr(self, key, val)
            self[key] = val
        

    def update(self, d: dict) -> dict:
            clone = copy.deepcopy(d)
            clone.update((d))
            return Config(clone)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok= True)
        with open(path, 'w') as f:
            yaml.dump(self.d, f, sort_keys= False)


def load_yaml(path: str, indent: int=0) -> str:
    root_dir = os.path.dirname(path)
    with open(path, 'r') as f:
        s = ''
        for line in f.readlines():
            if '!include' in line:
                include_pos = line.index('!include')
                pos = include_pos + 9
                path = os.path.join(root_dir, line[pos:].strip())
                s += '\n' + load_yaml(path, indent + include_pos)
            else:
                s += ' ' * indent + line
    return s
def load_config(path: str):
    yaml_file = load_yaml(path)
    return Config(yaml.safe_load(yaml_file))

