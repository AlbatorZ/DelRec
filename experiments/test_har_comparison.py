"""CPU regression checks: python -m unittest discover -s experiments -p test_har_comparison.py"""

from copy import deepcopy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from compare_har_snn_rsnn import Config, har, load_splits, make_loaders, parameter_counts, run_model, six_models
from delrec.delay_layers import axonal_recdel
from delrec.networks import dcls_module, learned_delay_parameter
from delrec.utils import reset_states


class HARComparisonTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = Config()
        self.config.hidden_layers = [8, 8, 8]
        self.config.time_window = 20
        self.config.epochs = 2
        self.config.batch_size = 4
        self.config.seed = 0
        self.config.recurrent_dropout_rate = 0.0

    def test_optimizers_gradients_offsets_and_checkpoint(self):
        generator = torch.Generator().manual_seed(52)
        x = torch.rand(20, 4, 3, generator=generator) * 2
        labels = torch.tensor([0, 1, 2, 3])
        for label, cfg, model in six_models(self.config, force_recurrence=False):
            with self.subTest(model=label):
                optimizers, _ = har.init_optim_sche(model, cfg)
                parameters = [p for opt in optimizers for group in opt.param_groups for p in group['params']]
                expected = {id(p) for p in model.parameters() if p.requires_grad}
                self.assertEqual(expected, {id(p) for p in parameters})
                self.assertEqual(len(parameters), len(expected))
                recurrent = [m for m in model.modules() if isinstance(m, axonal_recdel)]
                feedforward = [m for m in model.modules() if isinstance(m, dcls_module)]
                self.assertEqual(len(recurrent), 2 if label.startswith('Recurrent') else 0)
                self.assertEqual(bool(feedforward), label.startswith('Feedforward'))
                delays = [learned_delay_parameter(m, 'recurrent_delays') for m in recurrent]
                delays += [learned_delay_parameter(m, 'P') for m in feedforward]
                originals = [p.detach().clone() for p in delays]
                offsets = {n: b.clone() for n, b in model.named_buffers() if n.endswith('.offsets')}
                model.train()
                loss = har.calc_loss_HAR(model(x), labels)
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in delays))
                self.assertGreater(sum(p.grad.abs().sum().item() for p in delays), 0)
                for opt in optimizers:
                    opt.step()
                self.assertTrue(any(not torch.equal(before, after) for before, after in zip(originals, delays)))
                model.round_pos()
                self.assertTrue(all(torch.equal(b, dict(model.named_buffers())[n]) for n, b in offsets.items()))
                model.eval()
                reset_states(model)
                with torch.no_grad():
                    expected_output = model(x).clone()
                with tempfile.TemporaryDirectory() as temporary:
                    checkpoint = Path(temporary) / 'model.pth'
                    torch.save(model.state_dict(), checkpoint)
                    restored = type(model)(deepcopy(cfg))
                    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
                    restored.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(restored(x), expected_output)
                reset_states(model)

    def test_order_is_independent_of_model_random_draws(self):
        dataset = TensorDataset(torch.arange(12).reshape(12, 1, 1).float(), torch.arange(12))
        first = make_loaders((dataset,) * 2, self.config, torch.device('cpu'), False)[0]
        second = make_loaders((dataset,) * 2, self.config, torch.device('cpu'), False)[0]
        for _ in range(2):
            first_order = torch.cat([y for _, y in first])
            torch.rand(1000)
            self.assertTrue(torch.equal(first_order, torch.cat([y for _, y in second])))

    def test_full_training_partition_is_preserved(self):
        train = TensorDataset(torch.zeros(12, 20, 3), torch.zeros(12, dtype=torch.long))
        test = TensorDataset(torch.zeros(4, 20, 3), torch.zeros(4, dtype=torch.long))
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with patch('compare_har_snn_rsnn.load_dataset', return_value=(
                    SimpleNamespace(dataset=train), SimpleNamespace(dataset=test),
                    SimpleNamespace(dataset=test))):
                datasets = load_splits(self.config, out)
            self.assertEqual(len(datasets), 2)
            self.assertIs(datasets[0], train)
            self.assertIs(datasets[1], test)
            self.assertEqual(json.loads((out / 'split.json').read_text())['train_samples'], 12)
            self.assertFalse((out / 'split_indices.npz').exists())

    def test_reported_accuracy_and_checkpoint_use_best_test_epoch(self):
        self.config.epochs = 3
        # A middle-epoch peak must survive a later drop, including when the
        # last epoch has lower loss. Latest epoch wins equal-accuracy ties.
        for accuracies, expected_epoch in (([10.0, 80.0, 20.0], 2), ([80.0, 20.0, 80.0], 3)):
            with self.subTest(accuracies=accuracies), tempfile.TemporaryDirectory() as temporary:
                label, cfg, model = six_models(self.config, force_recurrence=False)[0]
                dataset = TensorDataset(torch.zeros(4, 20, 3), torch.zeros(4, dtype=torch.long))
                args = SimpleNamespace(smoke=False, kernel='v2', synaptic_kernel='v2')

                def train_epoch(loader, network, optimizers, epoch, device, config):
                    for opt in optimizers:
                        opt.step()
                    with torch.no_grad():
                        next(network.parameters()).fill_(epoch + 1)
                    return 25.0, 1.0

                directory = Path(temporary) / 'run'
                with patch.object(har, 'train', side_effect=train_epoch), patch.object(
                        har, 'test', side_effect=list(zip(accuracies, [3.0, 2.0, 1.0]))) as test:
                    history, result = run_model(label, cfg, model, (dataset, dataset),
                        torch.device('cpu'), args, directory)
                self.assertEqual(test.call_count, cfg.epochs)
                self.assertEqual(result['test_accuracy_percent'], max(accuracies))
                self.assertEqual(result['best_epoch'], expected_epoch)
                self.assertEqual(result['test_loss'], history[expected_epoch - 1]['test_loss'])
                best = torch.load(directory / 'best.pth', weights_only=False)
                last = torch.load(directory / 'last.pth', weights_only=False)
                self.assertEqual(best['epoch'], expected_epoch)
                self.assertEqual(best['test_accuracy_percent'], result['test_accuracy_percent'])
                self.assertTrue((next(iter(best['model'].values())) == expected_epoch).all())
                self.assertEqual(last['epoch'], cfg.epochs)

    def test_default_parameter_counts_and_mem_topology(self):
        expected = {'Recurrent Axonal': 105522, 'Recurrent Hybrid': 105522,
                    'Recurrent Synaptic': 152578, 'Feedforward Axonal': 58037,
                    'Feedforward Hybrid': 58037, 'Feedforward Synaptic': 114610}
        for label, cfg, model in six_models(Config(), force_recurrence=False):
            har.init_optim_sche(model, cfg)
            self.assertEqual(parameter_counts(model)['trainable_parameters'], expected[label])
        # Existing MEM callers retain their historical all-hidden recurrence.
        self.config.hidden_layers = [8]
        for label, _, model in six_models(self.config):
            if label.startswith('Recurrent'):
                self.assertEqual(sum(isinstance(m, axonal_recdel) for m in model.modules()), 1)


if __name__ == '__main__':
    unittest.main()
