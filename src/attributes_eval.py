from eval import evaluate
from config import get_train_yaml_config
from utils import jaccard_index, f1


if __name__ == '__main__':
    metric_names = ['jaccard', 'F1']
    metric_functions = [jaccard_index, f1]
    config = get_train_yaml_config('../configs/attributes_config.yaml')
    evaluate(config, metric_names, metric_functions)
