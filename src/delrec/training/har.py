"""Training and evaluation loop for HAR.

``init_optim_sche`` returns two optimizers and two schedulers, weights and
delays are optimized separately, with different learning rates, because a
delay moves in time steps and a weight does not. Per-epoch metrics go to the
optional ``logger`` callable and to the run's CSVs.
"""

import torch
import torch.nn.functional as F

from delrec.delay_layers import axonal_recdel
from delrec.networks import dcls_module, learned_delay_parameter, modified_batchnorm, spike_registrator
from delrec.utils import *

def get_dcls_sigma_for_epoch(config, epoch: int):

    if getattr(config, "DCLSversion", None) != "gauss":
        return 0.23

    total_epochs = max(1, config.epochs)

    decay_horizon = max(1, total_epochs // 2)

    sigma_min = 0.23
    if epoch >= decay_horizon:
        return sigma_min

    if config.siginit <= sigma_min:
        return sigma_min

    alpha = (sigma_min / float(config.siginit)) ** (1.0 / decay_horizon)
    sigma = float(config.siginit) * (alpha ** epoch)
    return max(sigma, sigma_min)


def get_spike_cost(model, normalize="NT"):
    """Eq. (20): mean squared firing rate over hidden layers, C = mean_l mean_{t,n} s^2.

    Added to the loss as lambda*C to trade accuracy against activity. Spikes come from
    the ``spike_registrator`` modules the network zoo inserts after each hidden layer.
    """
    costs = []
    for m in model.modules():
        if isinstance(m, spike_registrator):
            spk = getattr(m, "spikes", None)
            if spk is None or not torch.is_tensor(spk):
                continue
            spk = spk.float()  # (T, B, N)

            if spk.dim() != 3:
                costs.append(0.5 * spk.pow(2).mean())
                continue

            T, B, N = spk.shape
            if normalize == "NT":
                per_sample = 0.5 * spk.pow(2).sum(dim=(0, 2)) / (T * N)   # (B,)
                costs.append(per_sample.mean())
            else:
                costs.append(0.5 * spk.pow(2).mean())

    if not costs:
        return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
    return torch.stack(costs).mean()


def train(train_loader, model, optimizer, epoch, device, config, penalize_spikes=False, logger=no_log):
    train_loss = 0
    correct    = 0
    total      = 0

    model.train()

    # Set DCLS sigma for this epoch if applicable
    if config.DCLSversion == 'gauss':
        with torch.no_grad():
            sigma_now = get_dcls_sigma_for_epoch(config, epoch)
            for layer in model.layers:
                if isinstance(layer, dcls_module):
                    layer.SIG.fill_(sigma_now)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # inputs of shape (Batch, Time, Neurons)

        # if batch_idx > 3:  # for fast debug
        #     break

        inputs = inputs.permute(1,0,2).float().to(device)  #(time, batch, neurons)
        targets = targets.to(device)

        reset_states(model=model)
        outputs = model(inputs)
        loss = calc_loss_HAR(outputs, targets)

        if penalize_spikes:
            spike_cost = get_spike_cost(model)
            loss += config.spike_penalty * spike_cost

            logger({"spike_cost": spike_cost.item()})

        train_loss += loss.item()
        correct += calc_metric_HAR(outputs, targets)
        total += targets.size(0)

        for opt in optimizer: opt.zero_grad()
        loss.backward()
        for opt in optimizer: opt.step()

        progress_bar(
            batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%%'
            % (train_loss/(batch_idx+1), 100.*correct/total)
            )

    avg_loss = train_loss / len(train_loader)
    avg_acc = 100. * correct / total

    logger({
        'train/loss': avg_loss,
        'train/acc': avg_acc,
        'epoch': epoch,
    })

    logger(model.log_params())

    return avg_acc, avg_loss

def test(test_loader, model, epoch, device, config, penalize_spikes=False, logger=no_log):
    test_loss = 0
    correct = 0
    total = 0

    model.eval()

    if getattr(config, 'round_pos_each_epoch', False) and hasattr(model, 'round_pos'):
        model.round_pos()
    elif hasattr(model, 'clamp_delays'):
        model.clamp_delays()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            # inputs of shape (Batch, Time, Neurons)

            # if batch_idx > 3:  # for fast debug
            #     break

            inputs = inputs.permute(1,0,2).float().to(device)  #(time, batch, neurons)
            targets = targets.to(device)

            reset_states(model=model)
            outputs = model(inputs)
            loss = calc_loss_HAR(outputs, targets)

            if penalize_spikes:
                spike_cost = get_spike_cost(model)
                loss += config.spike_penalty * spike_cost

            test_loss += loss.item()
            correct += calc_metric_HAR(outputs, targets)
            total += targets.size(0)

            progress_bar(
                batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%%'
                % (test_loss/(batch_idx+1), 100.*correct/total)
            )

    avg_loss = test_loss / len(test_loader)
    avg_acc = 100. * correct / total

    logger({
        'test/loss': avg_loss,
        'test/acc': avg_acc,
        'epoch': epoch,
    })

    return avg_acc, avg_loss

def init_optim_sche(model, config):
    weights_norm = []
    weights = []
    positions = []

    for m in model.layers:
        if isinstance(m, torch.nn.Linear):
            weights.append(m.weight)
            if config.bias:
                weights.append(m.bias)

        elif isinstance(m, axonal_recdel):
            weights.append(m.recurrent_weights)
            if getattr(m, 'use_rec_bias', False):
                weights.append(m.recurrent_bias)
            positions.append(learned_delay_parameter(m, 'recurrent_delays'))

            if hasattr(m, 'p_spread'):
                positions.append(m.p_spread)

        elif isinstance(m, dcls_module):
            weights.append(m.weight)
            if m.bias is not None:
                weights.append(m.bias)
            positions.append(learned_delay_parameter(m, 'P'))
            # Gaussian widths are scheduled in train(), never optimized.
            if m.SIG is not None:
                m.SIG.requires_grad_(False)

        elif isinstance(m, modified_batchnorm):
            weights_norm.append(m.weight)
            if config.bias:
                weights_norm.append(m.bias)

    weights = [p for p in weights if p.requires_grad]
    weights_norm = [p for p in weights_norm if p.requires_grad]
    positions = [p for p in positions if p.requires_grad]

    optimizer = []
    scheduler = []

    wd_positions = getattr(config, 'weight_decay_positions', 0.0)

    if config.optim == 'adam':
        optimizer.append(torch.optim.Adam([{'params':weights, 'lr':config.lr_w, 'weight_decay':config.weight_decay},
                                           {'params':weights_norm, 'lr':config.lr_w, 'weight_decay':0},]))
        optimizer.append(torch.optim.Adam([{'params':positions, 'lr':config.lr_positions, 'weight_decay':wd_positions}]))
    elif config.optim == 'adamW':
        optimizer.append(torch.optim.AdamW([{'params':weights, 'lr':config.lr_w, 'weight_decay':config.weight_decay},
                                           {'params':weights_norm, 'lr':config.lr_w, 'weight_decay':0},]))
        optimizer.append(torch.optim.AdamW([{'params':positions, 'lr':config.lr_positions, 'weight_decay':wd_positions}]))
    else:
        raise NotImplementedError

    if config.scheduler_weights == 'cos':
        scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=config.epochs))
    elif config.scheduler_weights == 'onecycle':
        scheduler.append(torch.optim.lr_scheduler.OneCycleLR(optimizer[0], max_lr=config.lr_w, total_steps=config.epochs))
    elif config.scheduler_weights == 'step':
        scheduler.append(torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=config.step_size, gamma=config.step_gamma))
    else:
        raise NotImplementedError

    if config.scheduler_pos == 'cos':
        scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[1], T_max=config.epochs))
    elif config.scheduler_pos == 'onecycle':
        scheduler.append(torch.optim.lr_scheduler.OneCycleLR(optimizer[1], max_lr=config.lr_positions, total_steps=config.epochs))
    elif config.scheduler_pos == 'step':
        scheduler.append(torch.optim.lr_scheduler.StepLR(optimizer[1], step_size=config.step_size, gamma=config.step_gamma))
    else:
        raise NotImplementedError

    return optimizer, scheduler
