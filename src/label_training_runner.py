from train import label_training
from config import get_train_yaml_config


if __name__ == '__main__':
    config = get_train_yaml_config('../configs/label_config.yaml')
    label_training(config)
