import torch
import numpy as np


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
