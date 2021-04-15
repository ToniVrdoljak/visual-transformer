import os
import torch
import torch.nn as nn
import numpy as np
from model import VisionTransformer
from checkpoint import load_checkpoint
from data_loaders import *
from utils import setup_device, accuracy, MetricTracker, TensorboardWriter, f1, jaccard_index, f1_macro_mc
from lad_datasets import get_image_labels


def train_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler, metrics, metric_names, metric_fns,
                device=torch.device('cpu'), scheduler_step_per_batch=True):
    metrics.reset()

    m1_name, m2_name = metric_names
    m1_fn, m2_fn = metric_fns

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

        m1, m2 = m1_fn(batch_pred, batch_target), m2_fn(batch_pred, batch_target)

        metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())
        metrics.update(m1_name, m1.item())
        metrics.update(m2_name, m2.item())

        if batch_idx % 100 == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} {}: {:.2f}, {}: {:.2f}"
                  .format(epoch, batch_idx, len(data_loader), loss.item(), m1_name, m1.item(), m2_name, m2.item()))

    if not scheduler_step_per_batch:
        lr_scheduler.step()

    return metrics.result()


def valid_epoch(epoch, model, data_loader, criterion, metrics, metric_names, metric_fns, device=torch.device('cpu')):
    metrics.reset()
    losses = []
    m1s = []
    m2s = []

    m1_name, m2_name = metric_names
    m1_fn, m2_fn = metric_fns

    # validation loop
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)

            m1, m2 = m1_fn(batch_pred, batch_target), m2_fn(batch_pred, batch_target)

            losses.append(loss.item())
            m1s.append(m1.item())
            m2s.append(m2.item())

    loss = np.mean(losses)
    m1 = np.mean(m1s)
    m2 = np.mean(m2s)
    metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)

    metrics.update(m1_name, m1)
    metrics.update(m2_name, m2)

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
    metric_names = ['loss', 'acc@1', 'F1_macro']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    metric_functions = [lambda x, y: accuracy(x, y, topk=(1,))[0], f1_macro_mc]

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
        if log['val_acc@1'] > best_acc:
            best_acc = log['val_acc@1']
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
