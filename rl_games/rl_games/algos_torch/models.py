import rl_games.algos_torch.layers
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import rl_games.common.divergence as divergence
from rl_games.common.extensions.distributions import CategoricalMasked
from torch.distributions import Categorical
from rl_games.algos_torch.sac_helper import SquashedNormal
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats

class BaseModel():
    def __init__(self, model_class):
        self.model_class = model_class

    def is_rnn(self):
        return False

    def is_separate_critic(self):
        return False

    def get_value_layer(self):
        return None

    def build(self, config):
        obs_shape = config['input_shape']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        value_size = config.get('value_size', 1)
        extra_info_start_idx = config.get('coef_id_idx', None)
        assert not 'coef_id_idx' in config or len(obs_shape) == 1
        return self.Network(self.network_builder.build(self.model_class, **config), obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size, extra_info_start_idx=extra_info_start_idx)

class BaseModelNetwork(nn.Module):
    def __init__(self, obs_shape, normalize_value, normalize_input, value_size, extra_info_start_idx, **kwargs):
        nn.Module.__init__(self)
        self.obs_shape = obs_shape
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input
        self.value_size = value_size
        self.extra_info_start_idx = extra_info_start_idx

        if normalize_value:
            self.value_mean_std = RunningMeanStd((self.value_size,)) #   GeneralizedMovingStats((self.value_size,)) #   
        if normalize_input:
            if isinstance(obs_shape, dict):
                self.running_mean_std = RunningMeanStdObs(obs_shape)
            else:
                self.running_mean_std = RunningMeanStd((extra_info_start_idx,) if extra_info_start_idx is not None else obs_shape)

    def norm_obs(self, observation):
        with torch.no_grad():
            if self.normalize_input:
                return torch.cat([self.running_mean_std(observation[:,:self.extra_info_start_idx]), observation[:,self.extra_info_start_idx:]], dim=1) if self.extra_info_start_idx is not None else self.running_mean_std(observation)
            else:
                return observation

    def denorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, denorm=True) if self.normalize_value else value

class ModelA2C(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self,**kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
        
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()            

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            logits, value, states = self.a2c_network(input_dict)

            if is_train:
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                prev_neglogp = -categorical.log_prob(prev_actions)
                entropy = categorical.entropy()
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : categorical.logits,
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states
                }
                return result
            else:
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                selected_action = categorical.sample().long()
                neglogp = -categorical.log_prob(selected_action)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'logits' : categorical.logits,
                    'rnn_states' : states
                }
                return  result

class ModelA2CMultiDiscrete(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
        
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete_list(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            logits, value, states = self.a2c_network(input_dict)
            if is_train:
                if action_masks is None:
                    categorical = [Categorical(logits=logit) for logit in logits]
                else:   
                    categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]
                prev_actions = torch.split(prev_actions, 1, dim=-1)
                prev_neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, prev_actions)]
                prev_neglogp = torch.stack(prev_neglogp, dim=-1).sum(dim=-1)
                entropy = [c.entropy() for c in categorical]
                entropy = torch.stack(entropy, dim=-1).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : [c.logits for c in categorical],
                    'values' : value,
                    'entropy' : torch.squeeze(entropy),
                    'rnn_states' : states
                }
                return result
            else:
                if action_masks is None:
                    categorical = [Categorical(logits=logit) for logit in logits]
                else:   
                    categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]                
                
                selected_action = [c.sample().long() for c in categorical]
                neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, selected_action)]
                selected_action = torch.stack(selected_action, dim=-1)
                neglogp = torch.stack(neglogp, dim=-1).sum(dim=-1)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'logits' : [c.logits for c in categorical],
                    'rnn_states' : states
                }
                return  result

class ModelA2CContinuous(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
            
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            p = p_dict['mu'], p_dict['sigma']
            q = q_dict['mu'], q_dict['sigma']
            return divergence.d_kl_normal(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, sigma, value, states = self.a2c_network(input_dict)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)

            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'value' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result
            else:
                selected_action = distr.sample().squeeze()
                neglogp = -distr.log_prob(selected_action).sum(dim=-1)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return  result          


class ModelA2CContinuousLogStd(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            eigen_sigma = self._eigadd_sigma(input_dict['obs'])
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                corr = self._corr_factor()
                if corr is not None:
                    # exact entropy of the correlated Gaussian: the blend's
                    # logdet must stay in the graph so entropy_coef sees w
                    entropy = entropy + corr[3]
                elif getattr(self.a2c_network,
                             'noise_eigadd_basis', None) is not None:
                    # exact entropy of the additive Gaussian: diagonal part
                    # plus the capacitance logdet, kept in the graph so
                    # entropy_coef sees both sigma and the eigen loudness
                    entropy = entropy + 0.5 * self._eigadd_logdet_k(sigma, eigen_sigma)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd, eigen_sigma)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                if getattr(self.a2c_network, 'noise_eigadd_basis', None) is not None:
                    result['eigen_sigmas'] = eigen_sigma
                return result
            else:
                corr = self._corr_factor()
                L = getattr(self.a2c_network, 'noise_corr_chol', None)
                B = getattr(self.a2c_network, 'noise_eigsig_basis', None)
                A8 = getattr(self.a2c_network, 'noise_eigadd_basis', None)
                if A8 is not None:
                    # additive eigen noise: iid per-joint sample plus an
                    # independent sample along the K eigen directions
                    s8 = eigen_sigma
                    eps = torch.randn_like(mu)
                    eps8 = torch.randn(mu.shape[0], s8.shape[-1],
                                       device=mu.device, dtype=mu.dtype)
                    selected_action = mu + sigma * eps + (s8 * eps8) @ A8
                elif corr is not None:
                    eps = torch.randn_like(mu)
                    selected_action = mu + sigma * (eps @ corr[0].T)
                elif L is not None:
                    eps = torch.randn_like(mu)
                    selected_action = mu + sigma * (eps @ L.T)
                elif B is not None:
                    # eigen-sigma: noise sampled in the fixed eigenbasis, the
                    # policy's own per-dim sigma is per-DIRECTION loudness
                    eps = torch.randn_like(mu)
                    selected_action = mu + (sigma * eps) @ B
                else:
                    selected_action = distr.sample()
                # selected_action = distr.mean # DEBUG
                neglogp = self.neglogp(selected_action, mu, sigma, logstd, eigen_sigma)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                if getattr(self.a2c_network, 'noise_eigadd_basis', None) is not None:
                    result['eigen_sigmas'] = eigen_sigma
                return result

        def _corr_factor(self):
            """Learnable per-direction blend C(w) = D^-1/2 V diag(v) V^T D^-1/2.

            v_i = (1-w_i) + w_i * lam_i on the fixed human eigenbasis V;
            returns (A, v, d, logdet_A) with A A^T = C(w), unit diagonal.
            """
            net = self.a2c_network
            logit = getattr(net, 'noise_corr_logit', None)
            if logit is None:
                return None
            V = net.noise_corr_eigvecs
            lam = net.noise_corr_eigvals
            w = torch.sigmoid(logit)
            v = 1.0 - w + w * lam
            if getattr(net, 'noise_corr_unit_diag', True):
                M = (V * v) @ V.T
                d = torch.diagonal(M)
            else:
                # covariance blend: per-joint budgets from data. Renormalize so
                # total variance is pinned at D for EVERY dial position - the
                # dials can only redistribute shape, never inflate temperature
                # (sigma remains the sole loudness knob).
                v = v * (v.numel() / v.sum())
                d = torch.ones_like(v)
            A = (V * torch.sqrt(v)) * torch.rsqrt(d).unsqueeze(-1)
            logdet_a = 0.5 * (torch.log(v).sum() - torch.log(d).sum())
            return A, v, d, logdet_a

        def _eigadd_sigma(self, obs):
            net = self.a2c_network
            logsig = getattr(net, 'noise_eigadd_logsig', None)
            if logsig is None:
                return None
            if logsig.dim() == 2:
                # Use the same group identifiers as the conditional IID head,
                # including relabeled SAPG experience and shuffled RNN batches.
                ids = (obs[:, net.sigma_id_idx].reshape(-1, 1)
                       == net.sigma_ids).float().argmax(dim=1)
                return logsig[ids].exp()
            return logsig.exp().expand(obs.shape[0], -1)

        def _eigadd_chol(self, std, eigen_sigma):
            """Cholesky of K = I + S B D^-1 B^T S for Sigma = D + B^T S^2 B.

            B is the (K, n) eigen direction block (arbitrary, not required to
            be orthonormal), S = diag(exp(logsig)), D = diag(std^2). K is
            (N, K, K) when std is per-row.
            """
            net = self.a2c_network
            B8 = net.noise_eigadd_basis
            s = eigen_sigma
            invvar = std.pow(-2)
            if invvar.dim() == 1:
                invvar = invvar.unsqueeze(0)
            # G[n, a, b] = sum_i B[a, i] * invvar[n, i] * B[b, i], as a batched
            # matmul rather than a 3-operand torch.einsum: the einsum path
            # (opt_einsum.contract_path on its first call) pinned the whole
            # first-rollout call stack for the life of the process, holding
            # one rollout's buffers (~5 GB at 24576 envs) forever
            # (measured 2026-09-06: peak 3451 -> 2571 MiB at 4096 envs,
            # identical to the plain arm once einsum was gone).
            G = torch.matmul(B8.unsqueeze(0) * invvar.unsqueeze(1), B8.t())
            K = (s.unsqueeze(-1) * s.unsqueeze(-2)) * G
            K = K + torch.eye(s.shape[-1], device=K.device, dtype=K.dtype)
            return torch.linalg.cholesky(K)

        def _eigadd_logdet_k(self, std, eigen_sigma):
            L = self._eigadd_chol(std, eigen_sigma)
            return 2.0 * torch.log(
                torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

        def _eigadd_neglogp(self, x, mean, std, logstd, eigen_sigma):
            # Sigma = D + B^T S^2 B: Woodbury for the quadratic form and the
            # matrix determinant lemma for the logdet, both exact via the
            # K x K capacitance.
            net = self.a2c_network
            B8 = net.noise_eigadd_basis
            s = eigen_sigma
            delta = x - mean
            y2 = ((delta / std) ** 2).sum(dim=-1)
            w = ((delta / std.pow(2)) @ B8.t()) * s
            L = self._eigadd_chol(std, eigen_sigma)
            t = torch.cholesky_solve(w.unsqueeze(-1), L).squeeze(-1)
            logdet_k = 2.0 * torch.log(
                torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
            return 0.5 * (y2 - (w * t).sum(dim=-1)) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1) + 0.5 * logdet_k

        def neglogp(self, x, mean, std, logstd, eigen_sigma=None):
            if getattr(self.a2c_network, 'noise_eigadd_basis', None) is not None:
                if eigen_sigma is None:
                    raise ValueError('additive eigen likelihood requires per-row scales')
                return self._eigadd_neglogp(x, mean, std, logstd, eigen_sigma)
            corr = self._corr_factor()
            if corr is not None:
                A, v, d, logdet_a = corr
                y = (x - mean) / std
                # A^-1 = diag(v^-1/2) V^T diag(d^1/2)
                z = ((y * torch.sqrt(d)) @ self.a2c_network.noise_corr_eigvecs) \
                    * torch.rsqrt(v)
                return 0.5 * (z**2).sum(dim=-1) \
                    + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                    + logstd.sum(dim=-1) + logdet_a
            L = getattr(self.a2c_network, 'noise_corr_chol', None)
            if L is not None:
                z = torch.linalg.solve_triangular(
                    L, ((x - mean) / std).unsqueeze(-1), upper=False
                ).squeeze(-1)
                logdet = torch.log(torch.diagonal(L)).sum()
                return 0.5 * (z**2).sum(dim=-1) \
                    + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                    + logstd.sum(dim=-1) + logdet
            B = getattr(self.a2c_network, 'noise_eigsig_basis', None)
            if B is not None:
                # rotation is orthonormal (|det|=1): diag normal in z coords
                z = (x - mean) @ B.t()
                return 0.5 * ((z / std)**2).sum(dim=-1) \
                    + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                    + logstd.sum(dim=-1)
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)

class ModelMultiA2CContinuousLogStd(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_networks, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_networks = a2c_networks
            self.network_ids = kwargs['network_ids']
            self.coef_id_idx = kwargs['extra_info_start_idx']
        
        def split_input_dicts(self, input_dict):
            input_dicts = []
            indices_arr = []
            rnn_indices_arr = []
            for _, id in enumerate(self.network_ids):
                indices_arr.append(torch.where(input_dict['obs'][:,self.coef_id_idx] == id)[0])
                if indices_arr[-1].shape[0] == 0:
                    input_dicts.append({})
                    continue
                if 'rnn_states' in input_dict and input_dict['rnn_states'] is not None:
                    multiplier = input_dict['obs'].shape[0] // input_dict['rnn_states'][0].shape[1]
                    rnn_indices = indices_arr[-1][::multiplier] // multiplier
                else:
                    rnn_indices = None
                rnn_indices_arr.append(rnn_indices)
                new_dict = {}
                for k in input_dict:
                    if k == 'obs':
                        new_dict[k] = input_dict[k][indices_arr[-1],:self.coef_id_idx]
                    elif k in ['is_train', 'seq_length']:
                        new_dict[k] = input_dict[k]
                    elif k in ['prev_actions', 'dones']:
                        if input_dict[k] is None:
                            new_dict[k] = None
                        else:
                            new_dict[k] = input_dict[k][indices_arr[-1]]
                    elif k == 'rnn_states':
                        if rnn_indices is None:
                            new_dict[k] = None
                        else:
                            new_dict[k] = [s[:, rnn_indices, :] for s in input_dict[k]]
                input_dicts.append(new_dict)
            return input_dicts, indices_arr, rnn_indices_arr

        def concat_results(self, results, indices_arr, axis=0):
            results = torch.cat(results, dim=axis)
            return_val = torch.zeros_like(results)
            cat_indices = torch.cat(indices_arr, dim=0)
            if axis == 0:
                return_val[cat_indices] = results
            else:
                return_val[:, cat_indices] = results
            return return_val
            
            
        def is_rnn(self):
            return self.a2c_networks[0].is_rnn()

        def get_value_layer(self):
            return self.a2c_networks[0].get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_networks[0].get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dicts, indices_arr, rnn_indices_arr = self.split_input_dicts(input_dict)
            for in_dict in input_dicts:
                if in_dict == {}:
                    continue
                in_dict['obs'] = self.norm_obs(in_dict['obs'])
            
            mus, logstds, values, statess = [], [], [], []
            
            for i, in_dict in enumerate(input_dicts):
                if in_dict == {}:
                    continue
                mu, logstd, value, states = self.a2c_networks[i](in_dict)
                mus.append(mu)
                logstds.append(logstd)
                values.append(value)
                statess.append(states)
            
            mu = self.concat_results(mus, indices_arr)
            logstd = self.concat_results(logstds, indices_arr)
            value = self.concat_results(values, indices_arr)
            if self.is_rnn():
                states = tuple([self.concat_results(t, rnn_indices_arr, axis=1) for t in zip(*statess)])
            else:
                states = None

            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)
        
    def build(self, config):
        obs_shape = config['input_shape']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        value_size = config.get('value_size', 1)
        network_ids = config.get('coef_ids')
        extra_info_start_idx = config.get('coef_id_idx', obs_shape)
        assert not 'coef_id_idx' in config or len(obs_shape) == 1
        return self.Network(nn.ModuleList([self.network_builder.build(self.model_class, **config) for _ in network_ids]), obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size, extra_info_start_idx=extra_info_start_idx, network_ids=network_ids)


class ModelCentralValue(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def kl(self, p_dict, q_dict):
            return None # or throw exception?

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            value, states = self.a2c_network(input_dict)
            if not is_train:
                value = self.denorm_value(value)

            result = {
                'values': value,
                'rnn_states': states
            }
            return result



class ModelSACContinuous(BaseModel):

    def __init__(self, network):
        BaseModel.__init__(self, 'sac')
        self.network_builder = network
    
    class Network(BaseModelNetwork):
        def __init__(self, sac_network,**kwargs):
            BaseModelNetwork.__init__(self,**kwargs)
            self.sac_network = sac_network

        def critic(self, obs, action):
            return self.sac_network.critic(obs, action)

        def critic_target(self, obs, action):
            return self.sac_network.critic_target(obs, action)

        def actor(self, obs):
            return self.sac_network.actor(obs)
        
        def is_rnn(self):
            return False

        def forward(self, input_dict):
            is_train = input_dict.pop('is_train', True)
            mu, sigma = self.sac_network(input_dict)
            dist = SquashedNormal(mu, sigma)
            return dist



