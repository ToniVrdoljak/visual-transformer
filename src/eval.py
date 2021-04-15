import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model import VisionTransformer
from checkpoint import load_checkpoint
from data_loaders import *
from utils import setup_device


def evaluate(config, metric_names, metric_fns):
    # device
    device, device_ids = setup_device(config.n_gpu)

    # create model
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
        classifier_activation=eval('nn.' + config.classifier_activation))

    # load checkpoint
    if config.checkpoint_path:
        state_dict = load_checkpoint(config.checkpoint_path)
        model.load_state_dict(state_dict)
        print("Load pretrained weights from {}".format(config.checkpoint_path))

    # send model to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # create dataloader
    data_loader = eval("{}DataLoader".format(config.dataset))(
        data_dir=config.data_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        split='test')
    total_batch = len(data_loader)

    # starting evaluation
    print("Starting evaluation")

    m1_name, m2_name = metric_names
    m1_fn, m2_fn = metric_fns

    m1s = []
    m2s = []
    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=total_batch)
        for batch_idx, (data, target) in pbar:
            pbar.set_description("Batch {:05d}/{:05d}".format(batch_idx, total_batch))

            data = data.to(device)
            target = target.to(device)

            pred_logits = model(data)

            m1, m2 = m1_fn(pred_logits, target), m2_fn(pred_logits, target)

            m1s.append(m1.item())
            m2s.append(m2.item())

            pbar.set_postfix(**{m1_name: m1.item(), m2_name: m2.item()})

    print("Evaluation of model {:s} on dataset {:s}, {}: {:.4f}, {}: {:.4f}".format(config.model_arch,
                                                                                    config.dataset,
                                                                                    m1_name,
                                                                                    np.mean(m1s),
                                                                                    m2_name,
                                                                                    np.mean(m2s)))
