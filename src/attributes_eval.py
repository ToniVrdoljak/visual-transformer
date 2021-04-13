from eval import evaluate
from config import get_train_yaml_config

if __name__ == '__main__':
    config = get_train_yaml_config('../configs/attributes_config.yaml')
    evaluate(config, eval_type='attributes')
