"""Training-set-only epochs, with the same temporal readout for loss/accuracy."""

import torch
from torch.nn import functional as F

from delrec.delay_layers import axonal_recdel, vanilla_recurrent
from delrec.networks import dcls_module, learned_delay_parameter
from delrec.training.al import get_dcls_sigma_for_epoch
from delrec.utils import reset_states


def make_optimizer(model, config):
    # Classify actual layers, including synaptic/hybrid subclasses and vanilla
    # recurrence. Combined recurrent + feedforward delay networks use rec rates.
    family = ('recurrent' if any(isinstance(m, (axonal_recdel, vanilla_recurrent))
                                for m in model.modules()) else 'feedforward')
    # Preserve effective values in the config saved alongside each checkpoint.
    config.lr_w = getattr(config, f'lr_w_{family}', config.lr_w)
    config.lr_positions = getattr(config, f'lr_positions_{family}', config.lr_positions)
    positions = []
    for module in model.modules():
        if isinstance(module, axonal_recdel):
            positions.append(learned_delay_parameter(module, 'recurrent_delays'))
            if hasattr(module, "p_spread"):
                positions.append(module.p_spread)
        elif isinstance(module, dcls_module):
            positions.append(learned_delay_parameter(module, 'P'))
            # Width is scheduled explicitly, rather than optimized.
            if config.DCLSversion == "gauss":
                module.SIG.requires_grad_(False)
    position_ids = {id(p) for p in positions}
    weights = [p for p in model.parameters() if p.requires_grad and id(p) not in position_ids]
    return torch.optim.AdamW([
        {"params": weights, "lr": config.lr_w, "weight_decay": config.weight_decay},
        {"params": [p for p in positions if p.requires_grad],
         "lr": config.lr_positions, "weight_decay": 0.0},
    ])


def set_epoch(model, config, epoch):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, axonal_recdel):
                module.update_sigma(epoch)
            elif isinstance(module, dcls_module) and config.DCLSversion == "gauss":
                module.SIG.fill_(get_dcls_sigma_for_epoch(config, epoch))


def run_epoch(loader, model, device, config, optimizer=None, diagnostics=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs = inputs.permute(1, 0, 2).contiguous().to(device)
            labels = labels.to(device)
            reset_states(model)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            if config.readout == "mean":
                logits = outputs.mean(0)
            elif config.readout == "sum":
                logits = outputs.sum(0)
            elif config.readout == "last":
                logits = outputs[-1]
            else:
                raise ValueError(f"Unknown readout: {config.readout}")
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite memorization loss")
            if training:
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                if diagnostics is not None:
                    diagnostics.before_step(optimizer)
                optimizer.step()
                if hasattr(model, "clamp_delays"):
                    model.clamp_delays()
                if diagnostics is not None:
                    diagnostics.after_step()
            total_loss += loss.item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.numel()
    reset_states(model)
    return {"loss": total_loss / count, "accuracy_percent": 100.0 * correct / count,
            "correct": correct, "num_samples": count}
