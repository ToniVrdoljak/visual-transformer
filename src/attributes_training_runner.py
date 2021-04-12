from train import get_train_yaml_config, attributes_training

if __name__ == '__main__':
    config = get_train_yaml_config('../configs/attributes_config.yaml')
    attributes_training(config)
