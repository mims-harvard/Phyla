import os
import sys
import yaml
import torch

def load_config(Config):
    #Only handles one nested level of config and assumes one nested level
    config = Config()

    try:
        config_filename = sys.argv[1:][0]
    except Exception:
        raise Exception("Must include a config file << python X config_file >>")

    try:
        args = yaml.safe_load(open(config_filename, "r"))
    except Exception:
        raise Exception(f"Config file {config_filename} not found")
    
    for i in args:
        if hasattr(config, i):
            attr = getattr(config, i)
            for j in args[i]:
                if hasattr(attr, j):
                    setattr(attr, j, args[i][j])
                else:
                    raise Exception(f"{j} not in {i} config")
        else:
            raise Exception(f"{i} not in config")
    
    return config
