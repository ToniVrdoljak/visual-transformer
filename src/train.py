import os
import torch
import torch.nn as nn
import numpy as np
from model import VisionTransformer
from config import get_train_config, get_train_yaml_config
from checkpoint import load_checkpoint
from data_loaders import *
from utils import setup_device, accuracy, MetricTracker, TensorboardWriter, f1, jaccard_index, f1_macro_mc
from lad_datasets import get_image_labels


def train_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler, metrics, device=torch.device('cpu'),
                scheduler_step_per_batch=True, train_type='labels'):
    metrics.reset()

    # training loop
    for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        if scheduler_step_per_batch:
            lr_scheduler.step()

        if train_type == 'labels':
            (acc1,) = accuracy(batch_pred, batch_target, topk=(1,))
            f1_macro = f1_macro_mc(batch_pred, batch_target)

            metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
            metrics.update('loss', loss.item())
            metrics.update('acc1', acc1.item())
            metrics.update('F1_macro', f1_macro.item())

            if batch_idx % 100 == 0:
                print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} Acc@1: {:.2f}, F1_macro: {:.2f}"
                        .format(epoch, batch_idx, len(data_loader), loss.item(), acc1.item(), f1_macro.item()))

        elif train_type == 'attributes':
            jaccard = jaccard_index(batch_pred, batch_target)
            f1_score = f1(batch_pred, batch_target)

            metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
            metrics.update('loss', loss.item())
            metrics.update('jaccard', jaccard.item())
            metrics.update('F1', f1_score.item())

            if batch_idx % 100 == 0:
                print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} jaccard: {:.2f}, F1: {:.2f}"
                      .format(epoch, batch_idx, len(data_loader), loss.item(), jaccard.item(), f1_score.item()))

    if not scheduler_step_per_batch:
        lr_scheduler.step()

    return metrics.result()


def valid_epoch(epoch, model, data_loader, criterion, metrics, device=torch.device('cpu'), valid_type='labels'):
    metrics.reset()
    losses = []
    acc1s = []
    acc5s = []
    # validation loop
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)

            if valid_type == 'labels':
                (m1,) = accuracy(batch_pred, batch_target, topk=(1,))
                m2 = f1_macro_mc(batch_pred, batch_target)
            elif valid_type == 'attributes':
                m1 = jaccard_index(batch_pred, batch_target)
                m2 = f1(batch_pred, batch_target)

            losses.append(loss.item())
            acc1s.append(m1.item())
            acc5s.append(m2.item())

    loss = np.mean(losses)
    m1 = np.mean(acc1s)
    m2 = np.mean(acc5s)
    metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)

    if valid_type == 'labels':
        metrics.update('acc1', m1)
        metrics.update('F1_macro', m2)
    elif valid_type == 'attributes':
        metrics.update('jaccard', m1)
        metrics.update('F1', m2)

    return metrics.result()


def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str(save_dir + 'current.pth')
    torch.save(state, filename)

    if best:
        filename = str(save_dir + 'best.pth')
        torch.save(state, filename)


def label_training(config):
    """Procedure for training label models (standard multi-class classification)"""

    # device
    device, device_ids = setup_device(config.n_gpu)

    # tensorboard
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    # metric tracker
    metric_names = ['loss', 'acc1', 'F1_macro']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

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
                    split='train',
                    sample=config.sample)
    valid_dataloader = eval("{}DataLoader".format(config.dataset))(
                    data_dir=config.data_dir,
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='val')

    # training criterion
    print("create criterion and optimizer")

    if config.use_weights:
        image_labels = get_image_labels(os.path.join(config.data_dir, 'LAD_annotations'))
        inverse_weights = torch.tensor(list(image_labels.label_code.value_counts().sort_index()), dtype=torch.float)
        weights = 1 / inverse_weights
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

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
        result = train_epoch(epoch, model, train_dataloader, criterion, optimizer, lr_scheduler, train_metrics, device,
                             config.lr_scheduler['step_per_batch'])
        log.update(result)

        # validate the model
        model.eval()
        result = valid_epoch(epoch, model, valid_dataloader, criterion, valid_metrics, device)
        log.update(**{'val_' + k: v for k, v in result.items()})

        # best acc
        best = False
        if log['val_acc1'] > best_acc:
            best_acc = log['val_acc1']
            best = True

        # save model
        save_model(config.checkpoint_dir, epoch, model, optimizer, lr_scheduler, device_ids, best)

        # print logged informations to the screen
        for key, value in log.items():
            print('    {:15s}: {}'.format(str(key), value))


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
        result = train_epoch(epoch, model, train_dataloader, criterion, optimizer, lr_scheduler, train_metrics, device,
                             config.lr_scheduler['step_per_batch'], train_type='attributes')
        log.update(result)

        # validate the model
        model.eval()
        result = valid_epoch(epoch, model, valid_dataloader, criterion, valid_metrics, device, valid_type='attributes')
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
