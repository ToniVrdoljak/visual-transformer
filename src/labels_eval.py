from eval import evaluate
from config import get_train_yaml_config
from utils import accuracy, f1_macro_mc


if __name__ == '__main__':
    metric_names = ['acc@1', 'F1_macro']
    metric_functions = [lambda x, y: accuracy(x, y, topk=(1,))[0], f1_macro_mc]
    config = get_train_yaml_config('../configs/label_config.yaml')
    evaluate(config, metric_names, metric_functions)
