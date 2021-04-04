import torch
import torch.nn as nn

from config import get_train_yaml_config
from utils import setup_device, MetricTracker, TensorboardWriter, f1, jaccard_index
from model import VisionTransformer
from checkpoint import load_checkpoint
from data_loaders import *
from train import train_epoch, valid_epoch, save_model


def attributes_training(config):
    """Procedure for training attribute models (multi-label classification)"""

    # device
    device, device_ids = setup_device(config.n_gpu)

    # tensorboard
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    # metric tracker
    metric_names = ['loss', 'jaccard', 'F1']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    metric_functions = [jaccard_index, f1]

    # create model
    print("create model")
    model = VisionTransformer(
             image_size=(config.image_size, config.image_size),
             patch_size=(config.patch_size, config.patch_size),
             emb_dim=config.emb_dim,
             mlp_dim=config.mlp_dim,
             num_heads=config.num_heads,
             num_layers=config.num_layers,
             hidden_layers_dim=config.hidden_layers_dim,
             num_classes=config.num_classes,
             attn_dropout_rate=config.attn_dropout_rate,
             dropout_rate=config.dropout_rate,
             classifier_activation=eval('nn.' + config.classifier_activation),
             classifier_dropout_rate=config.classifier_dropout_rate)

    # load checkpoint
    if config.checkpoint_path:
        state_dict = load_checkpoint(config.checkpoint_path)
        if config.reset_classifier:
            c_keys = [key for key in state_dict.keys() if key.startswith('classifier')]
            for key in c_keys:
                del state_dict[key]
            print("re-initialize fc layer")
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
        print("Load pretrained weights from {}".format(config.checkpoint_path))

    # send model to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # create dataloader
    print("create dataloaders")
    train_dataloader = eval("{}DataLoader".format(config.dataset))(
                    data_dir=config.data_dir,
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='train')
    valid_dataloader = eval("{}DataLoader".format(config.dataset))(
                    data_dir=config.data_dir,
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='val')

    # training criterion
    print("create criterion and optimizer")
    pos_weight = torch.full((config.num_classes,), config.pos_weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # create optimizers and learning rate scheduler
    optimizer = eval('torch.optim.' + config.optimizer['type'])(
        params=model.parameters(),
        **config.optimizer['parameters'])

    lr_scheduler = eval('torch.optim.lr_scheduler.' + config.lr_scheduler['type'])(
        optimizer=optimizer,
        **config.lr_scheduler['parameters'])

    # start training
    print("start training")
    best_acc = 0.0
    epochs = config.train_steps // len(train_dataloader)
    for epoch in range(1, epochs + 1):
        log = {'epoch': epoch}

        # train the model
        model.train()
        result = train_epoch(epoch, model, train_dataloader, criterion, optimizer, lr_scheduler, train_metrics,
                             metric_names[1:], metric_functions, device, config.lr_scheduler['step_per_batch'])
        log.update(result)

        # validate the model
        model.eval()
        result = valid_epoch(epoch, model, valid_dataloader, criterion, valid_metrics, metric_names[1:],
                             metric_functions, device)

        log.update(**{'val_' + k: v for k, v in result.items()})

        # best acc
        best = False
        if log['val_jaccard'] > best_acc:
            best_acc = log['val_jaccard']
            best = True

        # save model
        save_model(config.checkpoint_dir, epoch, model, optimizer, lr_scheduler, device_ids, best)

        # print logged informations to the screen
        for key, value in log.items():
            print('    {:15s}: {}'.format(str(key), value))


if __name__ == '__main__':
    config = get_train_yaml_config('../configs/attributes_config.yaml')
    attributes_training(config)
