from train import get_train_yaml_config, label_training

if __name__ == '__main__':
    config = get_train_yaml_config('../configs/label_config.yaml')
    label_training(config)
