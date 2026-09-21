"""CPU contracts for additive-eigen PPO KL, runnable in each host environment."""
import inspect
import math
import tempfile
from pathlib import Path

import numpy as np
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.distributions import MultivariateNormal, kl_divergence
from rl_games.algos_torch import a2c_continuous, models, model_builder, torch_ext
from rl_games.common import a2c_common, datasets, schedulers


def dense(mu, sigma, eigen_sigma, basis):
    factor = eigen_sigma.unsqueeze(-1) * basis
    return MultivariateNormal(mu, covariance_matrix=(
        torch.diag_embed(sigma.square()) + factor.transpose(-2, -1) @ factor))


class Actor(nn.Module):
    def __init__(self, eigen=True, grouped=False):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(7))
        self.logstd = nn.Parameter(torch.zeros(7))
        self.value = nn.Parameter(torch.zeros(1))
        self.grouped = grouped
        if grouped:
            self.sigma_ids = torch.tensor([50., 25., 0.])
            self.sigma_id_idx = 0
            self.logstd = nn.Parameter(torch.zeros(3, 7))
        if eigen:
            self.register_buffer('noise_eigadd_basis', torch.randn(3, 7) * 0.3)
            self.noise_eigadd_logsig = nn.Parameter(torch.zeros(3, 3) if grouped else torch.zeros(3))

    def forward(self, data):
        n = len(data['obs'])
        std = self.logstd
        if self.grouped:
            idx = (data['obs'][:, 0, None] == self.sigma_ids).float().argmax(1)
            std = std[idx]
        return (self.mu.expand(n, -1) + 0., std.expand(n, -1) + 0.,
                self.value.expand(n, -1) + 0., data.get('rnn_states'))

    def get_aux_loss(self):
        return None


def model(eigen=True, grouped=False):
    kwargs = dict(obs_shape=(2,), normalize_value=False,
                  normalize_input=False, value_size=1)
    if 'extra_info_start_idx' in inspect.signature(models.BaseModelNetwork).parameters:
        kwargs['extra_info_start_idx'] = None
    return models.ModelA2CContinuousLogStd.Network(Actor(eigen, grouped), **kwargs)


class EigenKLTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(723)
        torch.set_num_threads(2)

    def test_exact_full_gaussian_kl(self):
        for n, k, prefix in [(16, 10, 0), (29, 9, 7), (44, 20, 12)]:
            for dtype in (torch.float64, torch.float32):
                with self.subTest(n=n, dtype=dtype):
                    B = torch.randn(k, n, dtype=dtype) * 0.3
                    B[:, :prefix] = 0
                    mu0, mu1 = torch.randn(2, 13, n, dtype=dtype)
                    std0, std1 = (torch.randn(2, 13, n, dtype=dtype) * 0.4).exp()
                    s0, s1 = (torch.randn(2, 13, k, dtype=dtype) * 0.3).exp()
                    expected = kl_divergence(dense(mu0, std0, s0, B), dense(mu1, std1, s1, B))
                    got = torch_ext.policy_kl_eigadd(mu0, std0, s0, mu1, std1, s1, B, False)
                    tol = 2e-5 if dtype == torch.float32 else 1e-10
                    torch.testing.assert_close(got, expected, atol=tol, rtol=tol)
                    torch.testing.assert_close(
                        torch_ext.policy_kl_eigadd(mu0, std0, s0, mu1, std1, s1, B), got.mean())

    def test_eigen_only_change_reaches_scheduler(self):
        B = torch.randn(3, 7, dtype=torch.float64)
        mu, std = torch.zeros(4, 7, dtype=torch.float64), torch.ones(4, 7, dtype=torch.float64)
        old_s = torch.ones(3, dtype=torch.float64)
        new_s = old_s * 1.2
        got = torch_ext.policy_kl_eigadd(mu, std, new_s, mu, std, old_s, B)
        expected = kl_divergence(dense(mu, std, new_s, B), dense(mu, std, old_s, B)).mean()
        torch.testing.assert_close(got, expected)
        self.assertGreater(float(got), 0.032)
        lr, _ = schedulers.AdaptiveScheduler(0.016).update(1e-4, 0., 1, 0, float(got))
        self.assertLess(lr, 1e-4)

    def test_identity_and_diagonal_limit(self):
        mu = torch.randn(4, 7, dtype=torch.float64)
        std = torch.rand(4, 7, dtype=torch.float64) + 0.5
        s = torch.ones(4, 3, dtype=torch.float64)
        B = torch.randn(3, 7, dtype=torch.float64)
        self.assertEqual(float(torch_ext.policy_kl_eigadd(mu, std, s, mu, std, s, B)), 0.)
        zero = torch.zeros_like(s)
        got = torch_ext.policy_kl_eigadd(mu, std, zero, mu + .1, std * 1.1, zero, B, False)
        expected = kl_divergence(dense(mu, std, zero, B), dense(mu + .1, std * 1.1, zero, B))
        torch.testing.assert_close(got, expected)

    def test_autocast_and_half_inputs(self):
        B = torch.randn(3, 7)
        mu, std, s = torch.randn(4, 7).bfloat16(), torch.ones(4, 7), torch.ones(4, 3)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            got = torch_ext.policy_kl_eigadd(mu, std, s, mu + .25, std * 1.1, s * 1.2, B, False)
        expected = kl_divergence(dense(mu.float(), std, s, B), dense((mu + .25).float(), std * 1.1, s * 1.2, B))
        self.assertEqual(got.dtype, torch.float32)
        torch.testing.assert_close(got, expected, atol=2e-6, rtol=1e-5)

    def test_grouped_sampling_density_entropy_and_isolation(self):
        m = model(grouped=True)
        net = m.a2c_network
        with torch.no_grad():
            net.noise_eigadd_logsig.copy_(torch.tensor([
                [-1., -.8, -.6], [0., .2, .4], [.6, .8, 1.]]))
            net.logstd.copy_(torch.randn(3, 7) * .2)
        # Permuted group IDs also model SAPG relabeling: the current obs must
        # determine scales, never original row order or rollout group index.
        obs = torch.tensor([[0., 1.], [50., 2.], [25., 3.], [0., 4.]])
        idx = torch.tensor([2, 0, 1, 2])
        expected_s = net.noise_eigadd_logsig[idx].exp()
        eps = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
        with patch.object(torch, 'randn_like', return_value=torch.zeros(4, 7)), \
                patch.object(torch, 'randn', return_value=eps):
            sample = m({'is_train': False, 'obs': obs.clone()})
        torch.testing.assert_close(sample['actions'], (eps * expected_s) @ net.noise_eigadd_basis)
        torch.testing.assert_close(sample['eigen_sigmas'], expected_s)
        distr = dense(sample['mus'], sample['sigmas'], expected_s, net.noise_eigadd_basis)
        train = m({'is_train': True, 'obs': obs.clone(), 'prev_actions': sample['actions'].detach()})
        torch.testing.assert_close(sample['neglogpacs'], -distr.log_prob(sample['actions']), atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(train['prev_neglogp'], sample['neglogpacs'])
        torch.testing.assert_close(train['entropy'], distr.entropy(), atol=2e-6, rtol=1e-5)
        # Entropy pressure from the exploratory group must leave both other
        # groups' eigen parameters untouched, including the leader (ID 0).
        (-train['entropy'][1]).backward()
        self.assertGreater(float(net.noise_eigadd_logsig.grad[0].norm()), 0.)
        self.assertEqual(int(torch.count_nonzero(net.noise_eigadd_logsig.grad[1:])), 0)
        self.assertEqual(int(torch.count_nonzero(net.logstd.grad[1:])), 0)
        old_leader = net.noise_eigadd_logsig[2].detach().clone()
        torch.optim.SGD(m.parameters(), lr=.1).step()
        torch.testing.assert_close(net.noise_eigadd_logsig[2], old_leader, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'per-row scales'):
            m.neglogp(sample['actions'], sample['mus'], sample['sigmas'], sample['sigmas'].log())
        old = m.state_dict()
        old['a2c_network.noise_eigadd_logsig'] = old['a2c_network.noise_eigadd_logsig'][0]
        with self.assertRaisesRegex(RuntimeError, 'size mismatch'):
            m.load_state_dict(old)

    def test_native_builder_initializes_independent_group_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / 'basis.npz'
            basis = np.eye(3, 7, dtype=np.float32)
            gains = np.array([.25, 1., 4.])
            np.savez(artifact, basis=basis, joint_gains=np.ones(7),
                     eigen_gains=gains, names=np.array(['a', 'b', 'c']))
            params = {'model': {'name': 'continuous_a2c_logstd'}, 'network': {
                'name': 'actor_critic', 'separate': False,
                'space': {'continuous': {'mu_activation': 'None', 'sigma_activation': 'None',
                    'mu_init': {'name': 'default'}, 'sigma_init': {'name': 'const_initializer', 'val': 0},
                    'fixed_sigma': 'coef_cond', 'noise_eigen_additive': str(artifact)}},
                'mlp': {'units': [16], 'activation': 'elu', 'd2rl': False,
                    'initializer': {'name': 'default'}, 'regularizer': {'name': 'None'}}}}
            m = model_builder.ModelBuilder().load(params).build(dict(
                actions_num=7, input_shape=(3,), num_seqs=6, value_size=1,
                normalize_value=False, normalize_input=False, type='extra_param',
                coef_ids=torch.linspace(50., 0., 6), coef_id_idx=2))
            s = m.a2c_network.noise_eigadd_logsig
            self.assertEqual(tuple(s.shape), (6, 3))
            torch.testing.assert_close(s.exp(), torch.tensor(gains).float().sqrt().expand(6, -1))
            with torch.no_grad():
                s[0].add_(1.)
            torch.testing.assert_close(s[1:].exp(), torch.tensor(gains).float().sqrt().expand(5, -1))

    def test_collection_and_minibatch_reference_updates(self):
        for eigen in (False, True):
            for rnn in (False, True):
                with self.subTest(eigen=eigen, rnn=rnn):
                    self.check_training(eigen, rnn)
                    if eigen:
                        self.check_training(eigen, rnn, grouped=True)

    def check_training(self, eigen, rnn, grouped=False):
        agent = a2c_continuous.A2CAgent.__new__(a2c_continuous.A2CAgent)
        agent.model = model(eigen, grouped)
        def init_buffer(this):
            this.experience_buffer = SimpleNamespace(tensor_dict={'sigmas': torch.zeros(2, 4, 7)})
        with patch.object(a2c_common.A2CBase, 'init_tensors', init_buffer):
            agent.init_tensors()
        self.assertEqual('eigen_sigmas' in agent.tensor_list, eigen)
        if eigen:
            self.assertEqual(agent.experience_buffer.tensor_dict['eigen_sigmas'].shape, (2, 4, 3))
        agent.is_rnn, agent.seq_length = rnn, 2
        agent.zero_rnn_on_done = False
        agent.normalize_value = agent.normalize_advantage = agent.has_central_value = False
        agent.dataset = datasets.PPODataset(8, 4, False, rnn, 'cpu', 2)
        obs = torch.randn(8, 2)
        if grouped:
            obs[:, 0] = torch.tensor([50., 25., 0., 50., 25., 0., 25., 0.])
        with torch.no_grad():
            rollout = agent.model({'is_train': False, 'obs': obs.clone()})
        batch = {k: v.clone() for k, v in rollout.items() if isinstance(v, torch.Tensor)}
        batch.update(obses=obs, returns=torch.ones(8, 1), dones=torch.zeros(8),
                     rnn_states=[torch.zeros(1, 4, 2)] if rnn else None,
                     rnn_masks=torch.tensor([1., 0., 1., 1., 1., 1., 1., 1.]) if rnn else None)
        agent.prepare_dataset(batch)
        # Force a change in eigen scales without touching mean/IID sigma.
        if eigen:
            with torch.no_grad():
                agent.model.a2c_network.noise_eigadd_logsig.add_(math.log(1.2))
            torch.testing.assert_close(batch['eigen_sigmas'], torch.ones(8, 3))
        agent.e_clip, agent.ppo = 0.2, True
        agent.mixed_precision = agent.multi_gpu = False
        agent.has_value_loss, agent.clip_value = True, False
        agent.ppo_device = 'cpu'
        agent.bound_loss_type = 'bound'
        agent.bounds_loss_coef = 0.
        agent.critic_coef, agent.entropy_coef, agent.last_lr = 1., .01, .001
        agent.expl_type, agent.config = 'none', {}
        agent.actor_loss_func = a2c_common.common_losses.actor_loss
        agent.optimizer = torch.optim.SGD(agent.model.parameters(), lr=agent.last_lr)
        agent.scaler = torch.cuda.amp.GradScaler(enabled=False)
        agent.diagnostics = SimpleNamespace(mini_batch=lambda *args: None)
        def step():
            agent.optimizer.step()
            return torch.zeros(1)
        agent.trancate_gradients_and_step = step
        for _ in range(2):
            data = agent.dataset[0]
            with torch.no_grad():
                current = agent.model({'is_train': True, 'obs': data['obs'].clone(), 'prev_actions': data['actions']})
                if eigen:
                    B = agent.model.a2c_network.noise_eigadd_basis
                    expected = kl_divergence(
                        dense(current['mus'], current['sigmas'], current['eigen_sigmas'], B),
                        dense(data['mu'], data['sigma'], data['eigen_sigma'], B))
                else:
                    expected = torch_ext.policy_kl(current['mus'], current['sigmas'], data['mu'], data['sigma'], False)
                expected = (expected * data['rnn_masks']).sum() / len(expected) if rnn else expected.mean()
            agent.calc_gradients(data)
            result = agent.train_result
            torch.testing.assert_close(result[3], expected, atol=2e-6, rtol=1e-4)
            new_mu, new_std, new_eigen = result[6:9]
            agent.dataset.update_mu_sigma(new_mu, new_std, new_eigen)
            if eigen:
                torch.testing.assert_close(agent.dataset[0]['eigen_sigma'], current['eigen_sigmas'])
                # Minibatch 1 must retain its own rollout reference.
                torch.testing.assert_close(agent.dataset[1]['eigen_sigma'], torch.ones(4, 3))
                with self.assertRaises(ValueError):
                    agent.dataset.update_mu_sigma(new_mu, new_std)


        # Run the native epoch loop too: its result tuple and scheduler wiring
        # must propagate the pre-update eigen snapshot through every minibatch.
        agent.frame, agent.epoch_num = 0, 1
        agent.vec_env = SimpleNamespace(set_train_info=lambda *args: None)
        agent.normalize_rms_advantage = agent.normalize_input = False
        agent.epochs_between_resets = 0
        agent.mini_epochs_num, agent.schedule_type = 2, 'standard'
        agent.scheduler = schedulers.AdaptiveScheduler(0.016)
        agent.algo_observer = SimpleNamespace(after_steps=lambda: None)
        agent.diagnostics.mini_epoch = lambda *args: None
        batch.update(played_frames=8, step_time=0.)
        if 'ps_extras' in inspect.getsource(a2c_common.ContinuousA2CBase.train_epoch):
            agent.play_steps = lambda: (batch.copy(), {'mb_intr_rewards': None, 'rewards': torch.zeros(8)})
        else:
            agent.play_steps = agent.play_steps_rnn = lambda: batch.copy()
        epoch_result = agent.train_epoch()
        self.assertEqual(len(epoch_result[8]), 2)
        self.assertTrue(all(torch.isfinite(v).all() for v in epoch_result[8]))


if __name__ == '__main__':
    unittest.main()
