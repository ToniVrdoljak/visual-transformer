from train import attributes_training
from config import get_train_yaml_config


if __name__ == '__main__':
    config = get_train_yaml_config('../configs/attributes_config.yaml')
    attributes_training(config)
