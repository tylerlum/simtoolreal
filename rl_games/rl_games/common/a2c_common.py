import copy
import math
import os
import signal
from omegaconf import DictConfig

from rl_games.common import vecenv

from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats
from rl_games.algos_torch.self_play_manager import SelfPlayManager
from rl_games.algos_torch import torch_ext
from rl_games.common import schedulers
from rl_games.common.custom_utils import create_sinusoidal_encoding, filter_leader, shuffle_batch, swap_and_flatten01
from rl_games.common.experience import ExperienceBuffer
from rl_games.common.interval_summary_writer import IntervalSummaryWriter
from rl_games.common.diagnostics import DefaultDiagnostics, PpoDiagnostics
from rl_games.algos_torch import  model_builder
from rl_games.interfaces.base_algorithm import  BaseAlgorithm
import numpy as np
import time
import gym

from tensorboardX import SummaryWriter
import torch 
from torch import nn
import torch.distributed as dist
 
from time import sleep

from rl_games.common import common_losses


ROLLING_CHECKPOINT_INTERVAL_SECONDS = 15 * 60
ARCHIVAL_CHECKPOINT_INTERVAL_SECONDS = 4 * 60 * 60


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


def print_statistics(print_stats, curr_frames, step_time, step_inference_time, total_time, epoch_num, max_epochs, frame, max_frames):
    if print_stats:
        step_time = max(step_time, 1e-9)
        fps_step = curr_frames / step_time
        fps_step_inference = curr_frames / step_inference_time
        fps_total = curr_frames / total_time

        # Prepare formatted strings
        fps_strs = [
            f"fps step                    : {fps_step:,.0f}",
            f"fps step + policy inference : {fps_step_inference:,.0f}",
            f"fps total                   : {fps_total:,.0f}"
        ]

        epoch_str = f"{epoch_num:,.0f}"
        if max_epochs != -1:
            epoch_str += f" / {max_epochs:,.0f}"

        frame_str = f"{frame:,.0f}"
        if max_frames != -1:
            frame_str += f" / {max_frames:,.0f}"

        # Print neatly
        print("\nStatistics:")
        for s in fps_strs:
            print(f"  {s}")
        print(f"  epoch                       : {epoch_str}")
        print(f"  frames                      : {frame_str}\n")


class A2CBase(BaseAlgorithm):

    def __init__(self, base_name, params):

        self.config = config = params['config']

        self.population_based_training = config.get('population_based_training', False)

        # This helps in PBT when we need to restart an experiment with the exact same name, rather than
        # generating a new name with the timestamp every time.
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name']
        self.policy_idx = int(self.experiment_name.split('_')[0])

        self.config = config
        self.algo_observer = config['features']['observer']
        self.algo_observer.before_init(base_name, config, self.experiment_name)
        self.load_networks(params)

        self.multi_gpu = config.get('multi_gpu', False)

        # multi-gpu/multi-node data
        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        self.curr_frames = 0

        if self.multi_gpu:
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            import hashlib
            dist.init_process_group("gloo", rank=self.global_rank, world_size=self.world_size, init_method=f'tcp://127.0.0.1:{23400 + int(hashlib.md5(self.experiment_name[3:].encode("utf-8")).hexdigest(), 16) % 500}')

            self.device_name = 'cuda:0' # DEBUG
            config['device'] = self.device_name
            if self.global_rank != 0:
                config['print_stats'] = False
                config['lr_schedule'] = None

        self.use_diagnostics = config.get('use_diagnostics', False)

        if self.use_diagnostics and self.global_rank == 0:
            self.diagnostics = PpoDiagnostics()
        else:
            self.diagnostics = DefaultDiagnostics()

        self.network_path = config.get('network_path', "./nn/")
        self.log_path = config.get('log_path', "runs/")
        self.env_config = config.get('env_config', {})
        self.num_actors = config['num_actors']
        self.env_name = config['env_name']

        self.vec_env = config.get('vec_env', None)
        if self.vec_env is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
        self.env_info = config.get('env_info') or self.vec_env.get_env_info()

        self.ppo_device = config.get('device', 'cuda:0') # DEBUG
        self.value_size = self.env_info.get('value_size',1)
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.central_value_config = self.config.get('central_value_config', None)
        self.has_central_value = self.central_value_config is not None
        self.truncate_grads = self.config.get('truncate_grads', False)

        if self.has_central_value:
            self.state_space = self.env_info.get('state_space', None)
            if isinstance(self.state_space,gym.spaces.Dict):
                self.state_shape = {}
                for k,v in self.state_space.spaces.items():
                    self.state_shape[k] = v.shape
            else:
                self.state_shape = self.state_space.shape

        self.self_play_config = self.config.get('self_play_config', None)
        self.has_self_play_config = self.self_play_config is not None

        self.self_play = config.get('self_play', False)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        # TODO: do we still need it?
        self.ppo = config.get('ppo', True)
        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'
        self.schedule_type = config.get('schedule_type', 'legacy')

        # Setting learning rate scheduler
        if self.is_adaptive_lr:
            self.kl_threshold = config['kl_threshold']
            self.scheduler = schedulers.AdaptiveScheduler(self.kl_threshold)

        elif self.linear_lr:
            
            if self.max_epochs == -1 and self.max_frames == -1:
                print("Max epochs and max frames are not set. Linear learning rate schedule can't be used, switching to the contstant (identity) one.")
                self.scheduler = schedulers.IdentityScheduler()
            else:
                use_epochs = True
                max_steps = self.max_epochs

                if self.max_epochs == -1:
                    use_epochs = False
                    max_steps = self.max_frames

                self.scheduler = schedulers.LinearScheduler(float(config['learning_rate']), 
                    max_steps = max_steps,
                    use_epochs = use_epochs, 
                    apply_to_entropy = config.get('schedule_entropy', False),
                    start_entropy_coef = config.get('entropy_coef'))
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.horizon_length = config['horizon_length']

        # seq_length is used only with rnn policy and value functions
        if 'seq_len' in config:
            print('WARNING: seq_len is deprecated, use seq_length instead')

        self.seq_length = self.config.get('seq_length', 4)
        print('seq_length:', self.seq_length)
        self.bptt_len = self.config.get('bptt_length', self.seq_length) # not used right now. Didn't show that it is usefull
        self.zero_rnn_on_done = self.config.get('zero_rnn_on_done', True)

        self.normalize_advantage = config['normalize_advantage']
        self.normalize_rms_advantage = config.get('normalize_rms_advantage', False)
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        self.truncate_grads = self.config.get('truncate_grads', False)

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k,v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
 
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.games_to_track = self.config.get('games_to_track', 3000)
        print('current training device:', self.ppo_device)
        self.game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_shaped_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        self.obs = None
        # self.games_num = self.config['minibatch_size'] // self.seq_length # it is used only for current rnn implementation

        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors

        assert(('minibatch_size_per_env' in self.config) or ('minibatch_size' in self.config))
        self.minibatch_size_per_env = self.config.get('minibatch_size_per_env', 0)
        self.minibatch_size = self.config.get('minibatch_size', self.num_actors * self.minibatch_size_per_env)

        self.num_minibatches = self.batch_size // self.minibatch_size
        print('num_minibatches:', self.num_minibatches)
        #assert(self.batch_size % self.minibatch_size == 0)

        self.mini_epochs_num = self.config['mini_epochs']

        self.mixed_precision = self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision and self.ppo_device != 'cpu')

        self.last_lr = self.config['learning_rate']
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0
        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')
        self.batch_dir = os.path.join(self.train_dir, 'batches')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)
        os.makedirs(self.batch_dir, exist_ok=True)

        self.entropy_coef = self.config['entropy_coef']

        if self.global_rank == 0:
            writer = SummaryWriter(self.summaries_dir)
            if self.population_based_training:
                self.writer = IntervalSummaryWriter(writer, self.config)
            else:
                self.writer = writer
        else:
            self.writer = None

        self.value_bootstrap = self.config.get('value_bootstrap')
        self.use_smooth_clamp = self.config.get('use_smooth_clamp', False)

        if self.use_smooth_clamp:
            self.actor_loss_func = common_losses.smoothed_actor_loss
        else:
            self.actor_loss_func = common_losses.actor_loss

        if self.normalize_advantage and self.normalize_rms_advantage:
            momentum = self.config.get('adv_rms_momentum', 0.5)
            self.advantage_mean_std = GeneralizedMovingStats((1,), momentum=momentum).to(self.ppo_device)

        self.is_tensor_obses = False

        self.last_rnn_indices = None # Unused
        self.last_state_indices = None

        #self_play
        if self.has_self_play_config:
            print('Initializing SelfPlay Manager')
            self.self_play_manager = SelfPlayManager(self.self_play_config, self.writer)

        # features
        self.algo_observer = config['features']['observer']

        self.soft_aug = config['features'].get('soft_augmentation', None)
        self.has_soft_aug = self.soft_aug is not None
        # soft augmentation not yet supported
        assert not self.has_soft_aug
        
        self.use_others_experience = config.get('use_others_experience')
        
        self.expl_type = config.get('expl_type', 'none')

        if self.expl_type != 'none':
            if self.expl_type.startswith('mixed_expl'):
                self.intr_coef_block_size = config.get('expl_coef_block_size')
                assert self.num_actors % self.intr_coef_block_size == 0
                env_ids = torch.arange(self.num_actors // self.intr_coef_block_size).repeat_interleave(self.intr_coef_block_size).to(self.ppo_device)
                embedding_genvec = torch.linspace(50.0, 0.0, self.num_actors // self.intr_coef_block_size).to(self.ppo_device)[env_ids]
                if 'disjoint' in self.expl_type or 'learn_param' in self.expl_type:
                    self.intr_reward_coef_embd = embedding_genvec.reshape(-1,1)
                else:
                    self.intr_reward_coef_embd = create_sinusoidal_encoding(embedding_genvec, config.get('expl_reward_coef_embd_size', 32), n=100).to(self.ppo_device)
            else:
                raise NotImplementedError
            expl_reward_type = config.get('expl_reward_type')
            if expl_reward_type == 'entropy':
                self.intr_reward_coef = torch.linspace(0.5, 0.0, self.num_actors // self.intr_coef_block_size).to(self.ppo_device)[env_ids] * config.get('expl_reward_coef_scale')
                self.intr_reward_model = None
            elif expl_reward_type == 'none':
                self.intr_reward_coef = torch.linspace(0.0, 0.0, self.num_actors // self.intr_coef_block_size).to(self.ppo_device)[env_ids]
                self.intr_reward_model = None
            else:
                raise NotImplementedError
        else:
            self.intr_reward_coef = None
            self.intr_reward_coef_embd = None
            self.intr_reward_model = None
        
        self.maybe_multiprocess = not self.expl_type.startswith('mixed_expl')
        
        self.ignore_env_boundary = config.get('good_reset_boundary', 0)
        if self.expl_type.startswith('mixed_expl'):
            self.ignore_env_boundary = max(self.ignore_env_boundary, self.num_actors - self.intr_coef_block_size)
        

    def trancate_gradients_and_step(self):
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))

            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.world_size
                    )
                    offset += param.numel()
        else:
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))

            all_grads = torch.cat(all_grads_list)
        
        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        
        return all_grads

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        if self.config.get('expl_type').startswith('mixed_expl') and 'disjoint' in self.config.get('expl_type'):
            params['model']['name'] = 'multi_' + params['model']['name']
        self.config['network'] = builder.load(params)
        has_central_value_net = self.config.get('central_value_config') is not  None
        if has_central_value_net:
            print('Adding Central Value Network')
            if 'model' not in params['config']['central_value_config']:
                params['config']['central_value_config']['model'] = {'name': 'central_value'}
            network = builder.load(params['config']['central_value_config'])
            self.config['central_value_config']['network'] = network

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        # do we need scaled time?
        self.diagnostics.send_info(self.writer)
        self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer.add_scalar('performance/step_time', step_time, frame)
        self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)

        self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), frame)
        _eig_names = getattr(getattr(self.model, 'a2c_network', None),
                             'noise_eigsig_names', None)
        if _eig_names:
            _sig = getattr(self.model.a2c_network, 'sigma', None)
            if _sig is not None:
                _sv = torch.exp(_sig.detach()).flatten().cpu().tolist()
                _vals = [(_n, _s) for _n, _s in zip(_eig_names, _sv)
                         if not _n.startswith('arm')]
                for _n, _s in _vals:
                    self.writer.add_scalar(f'info/noise_sigma/{_n}', _s, frame)
                self.writer.add_scalar(
                    'info/noise_sigma_mean',
                    sum(_s for _, _s in _vals) / len(_vals), frame)
        _net = getattr(self.model, 'a2c_network', None)
        _add_names = getattr(_net, 'noise_eigadd_names', None)
        if _add_names:
            _ls = getattr(_net, 'noise_eigadd_logsig', None)
            _sig = getattr(_net, 'sigma', None)
            if _ls is not None:
                with torch.no_grad():
                    _s = torch.exp(_ls.detach())
                    _sv = _s.cpu().tolist()
                    for _n, _si in zip(_add_names, _sv):
                        self.writer.add_scalar(
                            f'info/noise_eigadd_sigma/{_n}', _si, frame)
                    self.writer.add_scalar('info/noise_eigadd_sigma_mean',
                                           sum(_sv) / len(_sv), frame)
                    if isinstance(_sig, torch.Tensor):
                        # decoded eigen-noise RMS over the n action channels:
                        # diag(B^T S^2 B)_j = sum_k s_k^2 B[k, j]^2
                        _B = _net.noise_eigadd_basis
                        _diag = ((_s ** 2).unsqueeze(-1) * _B ** 2).sum(dim=0)
                        _eig_rms = float(torch.sqrt(_diag.mean()))
                        _jsig = torch.exp(_sig.detach()).flatten()
                        _joint_rms = float(torch.sqrt((_jsig ** 2).mean()))
                        self.writer.add_scalar(
                            'info/noise_eigadd_eigen_rms', _eig_rms, frame)
                        self.writer.add_scalar(
                            'info/noise_eigadd_joint_rms', _joint_rms, frame)
                        self.writer.add_scalar(
                            'info/noise_eigadd_eigen_joint_ratio',
                            _eig_rms / _joint_rms, frame)
        _corr_logit = getattr(getattr(self.model, 'a2c_network', None), 'noise_corr_logit', None)
        if _corr_logit is not None:
            _w = torch.sigmoid(_corr_logit.detach()).cpu()
            _names = getattr(getattr(self.model, 'a2c_network', None),
                             'noise_corr_dial_names', None)
            if _names:
                _vals = [(_n, _wi) for _n, _wi in zip(_names, _w.tolist())
                         if not _n.startswith('arm')]
                self.writer.add_scalar('info/noise_corr_w_mean',
                                       sum(v for _, v in _vals) / len(_vals), frame)
                for _n, _wi in _vals:
                    self.writer.add_scalar(f'info/noise_corr_w/{_n}', _wi, frame)
            else:
                self.writer.add_scalar('info/noise_corr_w_mean', _w.mean().item(), frame)
                for _i, _wi in enumerate(_w.tolist()):
                    self.writer.add_scalar(f'info/noise_corr_w/dir_{_i:02d}', _wi, frame)
        self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
        self.writer.add_scalar('info/lr_mul', lr_mul, frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
        self.writer.add_scalar('info/kl', torch_ext.mean_list(kls).item(), frame)
        self.writer.add_scalar('info/epochs', epoch_num, frame)
        print(f"\nPolicy {self.policy_idx}:", end=' ')
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    def set_eval(self):
        self.model.eval()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.train()

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_mul', 1.0)
        
        #if self.has_central_value:
        #    self.central_value_net.update_lr(lr)

    def get_action_values(self, obs, rnn_states=None):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict

    def get_values(self, obs, rnn_states):
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    'actions' : None,
                    'is_done': self.dones,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs['obs'])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : processed_obs,
                    'rnn_states' : rnn_states
                }
                result = self.model(input_dict)
                value = result['values']
            return value

    @property
    def device(self):
        return self.ppo_device

    def reset_envs(self):
        self.obs = self.env_reset()

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device, self.intr_reward_coef_embd.shape[-1] if self.intr_reward_coef_embd is not None else None)

        val_shape = (self.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        
        if not hasattr(self, 'current_rewards') or self.current_rewards is None:
            self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
            self.current_shaped_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
            self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
            self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)
        else:
            self.current_rewards = self.current_rewards.to(self.ppo_device)
            self.current_shaped_rewards = self.current_shaped_rewards.to(self.ppo_device)
            self.current_lengths = self.current_lengths.to(self.ppo_device)
            self.dones = self.dones.to(self.ppo_device)

        if self.is_rnn:
            if not hasattr(self, 'rnn_states') or self.rnn_states is None:
                self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

            total_agents = self.num_agents * self.num_actors
            num_seqs = self.horizon_length // self.seq_length
            assert((self.horizon_length * total_agents // self.num_minibatches) % self.seq_length == 0)
            self.mb_rnn_states = [torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.rnn_states]

    def init_rnn_from_model(self, model):
        self.is_rnn = self.model.is_rnn()

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.ppo_device)
            else:
                obs = torch.FloatTensor(obs).to(self.ppo_device)
        return obs.to(self.ppo_device)

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions.to(self.vec_env.env.device))
        obs = self.obs_to_tensors(obs)   
        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            rewards, dones, infos = rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            rewards, dones, infos = torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos
        
        if self.intr_reward_model is not None:
            with torch.no_grad():
                intr_rewards = self.intr_reward_model({"obs" : obs['obs'], "ids" : self.intr_reward_coef_embd[:,0] if self.intr_reward_coef_embd is not None else None}).unsqueeze(-1)
            with torch.enable_grad():
                intr_rew_loss = self.intr_reward_model.update({"obs" : obs['obs'], "ids" : self.intr_reward_coef_embd[:,0]  if self.intr_reward_coef_embd is not None else None})
        else:
            intr_rewards = torch.zeros_like(rewards)
        
        tr_obs = self.obs_to_tensors(obs)
        if self.intr_reward_coef_embd is not None:
            tr_obs['obs'] = torch.cat([tr_obs['obs'], self.intr_reward_coef_embd], dim=1)
            tr_obs['states'] = torch.cat([tr_obs['states'], self.intr_reward_coef_embd], dim=1)
        
        return tr_obs, rewards, intr_rewards, dones, infos

    def env_reset(self):
        obs = self.vec_env.reset()
        obs = self.obs_to_tensors(obs)
        if self.intr_reward_coef_embd is not None:
            obs['obs'] = torch.cat([obs['obs'], self.intr_reward_coef_embd], dim=1)
            obs['states'] = torch.cat([obs['states'], self.intr_reward_coef_embd], dim=1)
        return obs

    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    def discount_values_masks(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards, mb_masks):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)
            masks_t = mb_masks[t].unsqueeze(1)
            delta = (mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_extrinsic_values[t])
            mb_advs[t] = lastgaelam = (delta + self.gamma * self.tau * nextnonterminal * lastgaelam) * masks_t
        return mb_advs

    def clear_stats(self):
        batch_size = self.num_agents * self.num_actors
        self.game_rewards.clear()
        self.game_shaped_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.algo_observer.after_clear_stats()

    def _start_checkpoint_schedule(self):
        now = time.monotonic()
        self._next_rolling_checkpoint_time = now + ROLLING_CHECKPOINT_INTERVAL_SECONDS
        self._next_archival_checkpoint_time = now + ARCHIVAL_CHECKPOINT_INTERVAL_SECONDS

    @staticmethod
    def _advance_checkpoint_time(due_time, now, interval):
        skipped_intervals = int((now - due_time) // interval) + 1
        return due_time + skipped_intervals * interval

    def _save_periodic_checkpoints(self, checkpoint_name, override_state=None):
        now = time.monotonic()
        if now >= self._next_archival_checkpoint_time:
            archive_path = os.path.join(self.nn_dir, 'last_' + checkpoint_name)
            if override_state is None:
                self.save(archive_path)
            else:
                self.save(archive_path, override_state)
            self._next_archival_checkpoint_time = self._advance_checkpoint_time(
                self._next_archival_checkpoint_time,
                now,
                ARCHIVAL_CHECKPOINT_INTERVAL_SECONDS,
            )

        if now >= self._next_rolling_checkpoint_time:
            self._save_rolling_checkpoint(override_state)
            self._next_rolling_checkpoint_time = self._advance_checkpoint_time(
                self._next_rolling_checkpoint_time,
                now,
                ROLLING_CHECKPOINT_INTERVAL_SECONDS,
            )

    def _save_rolling_checkpoint(self, override_state=None):
        # Materialize the new checkpoint under a scratch name first, then
        # rotate model.pth -> model.pth.old, then move the new file into
        # place, so a complete rolling checkpoint (model.pth or
        # model.pth.old) exists on disk at every point in the sequence.
        latest_dir = os.path.join(self.experiment_dir, 'last')
        torch_ext.safe_filesystem_op(os.makedirs, latest_dir, exist_ok=True)
        latest_path = os.path.join(latest_dir, 'model.pth')
        staging_base = os.path.join(latest_dir, 'model_next')
        if override_state is None:
            self.save(staging_base)
        else:
            self.save(staging_base, override_state)
        if os.path.exists(latest_path):
            os.replace(latest_path, latest_path + '.old')
        os.replace(staging_base + '.pth', latest_path)

    def _install_exit_signal_handlers(self):
        self._exit_signal = None
        self._exit_checkpoint_saved = False
        self._previous_signal_handlers = {}

        def _request_exit(signum, frame):
            if self._exit_signal is None:
                self._exit_signal = signum
                print('Received signal {}: will save a checkpoint and exit '
                      'at the next epoch boundary'.format(signum))
                # A repeated signal falls through to the previous handler.
                previous = self._previous_signal_handlers.get(signum)
                if previous is not None:
                    signal.signal(signum, previous)

        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                self._previous_signal_handlers[signum] = signal.signal(signum, _request_exit)
            except (ValueError, OSError):
                # Not in the main thread, or the signal is unavailable.
                pass

    def _restore_exit_signal_handlers(self):
        for signum, previous in getattr(self, '_previous_signal_handlers', {}).items():
            if previous is not None:
                try:
                    signal.signal(signum, previous)
                except (ValueError, OSError):
                    pass
        self._previous_signal_handlers = {}

    def _save_exit_checkpoint(self):
        # Final checkpoint written when the train loop exits on SIGTERM or
        # SIGINT: archival-style name plus a refresh of the rolling
        # last/model.pth so resume glue picks up the newest state.
        if getattr(self, '_exit_checkpoint_saved', False):
            return
        self._exit_checkpoint_saved = True
        if self.global_rank != 0:
            return
        checkpoint_name = 'last_' + self.config['name'] + '_ep_' + str(self.epoch_num) \
            + '_rew_' + str(self.mean_rewards).replace('[', '_').replace(']', '_')
        print('Termination signal {}: saving checkpoint {}'.format(
            self._exit_signal, checkpoint_name))
        self.save(os.path.join(self.nn_dir, checkpoint_name))
        self._save_rolling_checkpoint()

    def save_shutdown_checkpoint(self):
        """Best-effort rolling checkpoint for host shutdown paths.

        Returns True when a checkpoint was written. No-op when the train
        loop already wrote its exit checkpoint (clean or signal exit) or
        training never started.
        """
        if getattr(self, '_exit_checkpoint_saved', False):
            return False
        if self.global_rank != 0:
            return False
        if getattr(self, 'model', None) is None:
            return False
        self._save_rolling_checkpoint()
        return True

    def update_epoch(self):
        pass

    def train(self):
        pass

    def prepare_dataset(self, batch_dict):
        pass

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

    def train_actor_critic(self, obs_dict, opt_step=True):
        pass

    def calc_gradients(self):
        pass

    def get_central_value(self, obs_dict):
        return self.central_value_net.get_value(obs_dict)

    def train_central_value(self):
        return self.central_value_net.train_net()

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = self.epoch_num
        state['frame'] = self.frame
        state['optimizer'] = self.optimizer.state_dict()
        # Adaptive-KL scheduler state: without these a warm resume restarts
        # the LR at the config value (up to ~30x above its adapted level).
        state['last_lr'] = self.last_lr
        state['entropy_coef'] = self.entropy_coef

        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state['last_mean_rewards'] = self.last_mean_rewards

        state['trackers'] = {}
        state['trackers']['game_rewards'] = self.game_rewards.state_dict()
        state['trackers']['game_shaped_rewards'] = self.game_shaped_rewards.state_dict()
        state['trackers']['game_lengths'] = self.game_lengths.state_dict()

        if self.vec_env is not None:
            env_state = self.vec_env.get_env_state()
            state['env_state'] = env_state

        if self.intr_reward_model is not None:
            state['intr_reward_model'] = self.intr_reward_model.state_dict()
        
        state['rnn_states'] = self.rnn_states
        state['dones'] = self.dones
        state['obs'] = self.obs
        state['current_rewards'] = self.current_rewards
        state['current_shaped_rewards'] = self.current_shaped_rewards
        state['current_lengths'] = self.current_lengths
     
        return state

    def set_full_state_weights(self, weights, set_epoch=True):

        self.set_weights(weights)
        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']

        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])

        self.optimizer.load_state_dict(weights['optimizer'])
        # Restore the adaptive-KL scheduler state. Precedence per key:
        # 1) the value persisted in the checkpoint; 2) an explicit
        # RLG_RESUME_LAST_LR / RLG_RESUME_ENTROPY_COEF override (rollbacks
        # onto pre-fix checkpoints, values read from wandb history);
        # 3) legacy behavior: last_lr falls back to the optimizer's restored
        # param-group LR and entropy_coef keeps its init (config) value.
        if 'last_lr' in weights:
            self.last_lr = weights['last_lr']
            lr_source = 'checkpoint'
        elif os.environ.get('RLG_RESUME_LAST_LR'):
            self.last_lr = float(os.environ['RLG_RESUME_LAST_LR'])
            lr_source = 'override'
        else:
            self.last_lr = weights['optimizer']['param_groups'][0]['lr']
            lr_source = 'legacy-optimizer'
        if 'entropy_coef' in weights:
            self.entropy_coef = weights['entropy_coef']
        elif os.environ.get('RLG_RESUME_ENTROPY_COEF'):
            self.entropy_coef = float(os.environ['RLG_RESUME_ENTROPY_COEF'])
        if lr_source != 'legacy-optimizer':
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr * param_group.get('lr_mul', 1.0)
        # The simulator's internal solver state is not restorable, so the
        # first post-resume rollout is polluted: collect and count it, but
        # skip the PPO update on it (flag consumed by train_epoch).
        self._discard_post_resume_rollout = True
        print(f'[warm-resume] adaptive LR restored to {self.last_lr:.6g} '
              f'(source: {lr_source}), entropy_coef {self.entropy_coef:.6g}; '
              'first post-resume rollout will skip its PPO update')

        self.last_mean_rewards = weights.get('last_mean_rewards', -1000000000)

        # restore trackers
        if 'trackers' in weights:
            if weights['trackers']['game_rewards']['mean'].shape != self.game_rewards.mean.shape:
                weights['trackers']['game_rewards']['mean'] = weights['trackers']['game_rewards']['mean'].reshape(self.game_rewards.mean.shape)
            if weights['trackers']['game_shaped_rewards']['mean'].shape != self.game_shaped_rewards.mean.shape:
                weights['trackers']['game_shaped_rewards']['mean'] = weights['trackers']['game_shaped_rewards']['mean'].reshape(self.game_shaped_rewards.mean.shape)
            if weights['trackers']['game_lengths']['mean'].shape != self.game_lengths.mean.shape:
                weights['trackers']['game_lengths']['mean'] = weights['trackers']['game_lengths']['mean'].reshape(self.game_lengths.mean.shape)
            self.game_rewards.load_state_dict(weights['trackers']['game_rewards'], strict=False)
            self.game_shaped_rewards.load_state_dict(weights['trackers']['game_shaped_rewards'], strict=False)
            self.game_lengths.load_state_dict(weights['trackers']['game_lengths'], strict=False)
        
        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

        # Skip loading if the shapes don't match
        SKIP = "current_rewards" not in weights or weights["current_rewards"].shape[0] != self.num_actors
        if SKIP:
            print("Skipping loading of many things in a2c_common.set_full_state_weights because the shapes don't match")
            print(f"self.num_actors = {self.num_actors}, weights['current_rewards'].shape = {weights['current_rewards'].shape if 'current_rewards' in weights else 'not in weights'}")

        for key in ['rnn_states', 'dones', 'obs', 'current_rewards', 'current_shaped_rewards', 'current_lengths']:
            if key in weights:
                if not SKIP:
                    setattr(self, key, weights[key])
        
        if self.intr_reward_model is not None:
            if 'intr_reward_model' in weights:
                self.intr_reward_model.load_state_dict(weights['intr_reward_model'])
            else:
                print('WARNING: no intr_reward_model in checkpoint')
        
    def get_weights(self):
        state = self.get_stats_weights()
        state['model'] = self.model.state_dict()
        return state

    def get_stats_weights(self, model_stats=False):
        state = {}
        if self.mixed_precision:
            state['scaler'] = self.scaler.state_dict()
        if self.has_central_value:
            state['central_val_stats'] = self.central_value_net.get_stats_weights(model_stats)
        if model_stats:
            if self.normalize_input:
                state['running_mean_std'] = self.model.running_mean_std.state_dict()
            if self.normalize_value:
                state['reward_mean_std'] = self.model.value_mean_std.state_dict()

        return state

    def set_stats_weights(self, weights):
        if self.normalize_rms_advantage:
            self.advantage_mean_std.load_state_dic(weights['advantage_mean_std'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_value and 'normalize_value' in weights:
            self.model.value_mean_std.load_state_dict(weights['reward_mean_std'])
        if self.mixed_precision and 'scaler' in weights:
            self.scaler.load_state_dict(weights['scaler'])

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        self.set_stats_weights(weights)

    def get_param(self, param_name):
        if param_name in [
            "grad_norm",
            "critic_coef", 
            "bounds_loss_coef",
            "entropy_coef",
            "kl_threshold",
            "gamma",
            "tau",
            "mini_epochs_num",
            "e_clip",
            ]:
            return getattr(self, param_name)
        elif param_name == "learning_rate":
            return self.last_lr
        else:
            raise NotImplementedError(f"Can't get param {param_name}")       

    def set_param(self, param_name, param_value):
        if param_name in [
            "grad_norm",
            "critic_coef", 
            "bounds_loss_coef",
            "entropy_coef",
            "gamma",
            "tau",
            "mini_epochs_num",
            "e_clip",
            ]:
            setattr(self, param_name, param_value)
        elif param_name == "learning_rate":
            if self.global_rank == 0:
                if self.is_adaptive_lr:
                    raise NotImplementedError("Can't directly mutate LR on this schedule")
                else:
                    self.learning_rate = param_value

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate * param_group.get('lr_mul', 1.0)
        elif param_name == "kl_threshold":
            if self.global_rank == 0:
                if self.is_adaptive_lr:
                    self.kl_threshold = param_value
                    self.scheduler.kl_threshold = param_value
                else:
                    raise NotImplementedError("Can't directly mutate kl threshold")
        else:
            raise NotImplementedError(f"No param found for {param_value}")

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k,v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def play_steps(self):
        update_list = self.update_list
        step_time = 0.0
        
        if self.is_rnn:
            mb_rnn_states = self.mb_rnn_states
            rnn_state_buffer = [torch.zeros((self.horizon_length, *s.shape), dtype=s.dtype, device=s.device) for s in self.rnn_states]

        for n in range(self.horizon_length):
            if self.is_rnn:
                if n % self.seq_length == 0:
                    for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                        mb_s[n // self.seq_length,:,:,:] = s

                for i, s in enumerate(self.rnn_states):
                    rnn_state_buffer[i][n,:,:,:] = s

                if self.has_central_value:
                    self.central_value_net.pre_step_rnn(n)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self.rnn_states)

            if self.is_rnn:
                self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones.byte())

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, intr_rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            intr_rewards = self.rewards_shaper(intr_rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('intr_rewards', n, intr_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            if self.is_rnn and len(all_done_indices) > 0:
                if self.zero_rnn_on_done:
                    for s in self.rnn_states:
                        s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                if self.has_central_value:
                    self.central_value_net.post_step_rnn(all_done_indices)

            if len(env_done_indices[env_done_indices >= self.ignore_env_boundary]) > 0:
                indices = env_done_indices[env_done_indices >= self.ignore_env_boundary].view(-1, 1)
                self.game_rewards.update(self.current_rewards[indices])
                self.game_shaped_rewards.update(self.current_shaped_rewards[indices])
                self.game_lengths.update(self.current_lengths[indices])
            self.algo_observer.process_infos(infos, env_done_indices, ignore_env_boundary=self.ignore_env_boundary)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs, self.rnn_states)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()

        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_intr_rewards = self.experience_buffer.tensor_dict['intr_rewards']
        if self.intr_reward_coef is not None:
            mb_total_rewards = mb_rewards + self.intr_reward_coef.unsqueeze(0).unsqueeze(2) * mb_intr_rewards
        else:
            mb_total_rewards = mb_rewards
        
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_total_rewards)
        mb_returns = mb_advs + mb_values
        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)

        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        if self.is_rnn:
            states = []
            for mb_s in mb_rnn_states:
                t_size = mb_s.size()[0] * mb_s.size()[2]
                h_size = mb_s.size()[3]
                states.append(mb_s.permute(1,2,0,3).reshape(-1,t_size, h_size))

            batch_dict['rnn_states'] = states
        batch_dict['step_time'] = step_time
        
        extras = {
            'rewards' : mb_rewards, 
            'obs' : self.experience_buffer.tensor_dict['obses'], 
            'last_obs' : self.obs,
            'states' : self.experience_buffer.tensor_dict.get('states', None),
            'dones' : mb_fdones,
            'last_dones' : fdones,
            'rnn_states' : rnn_state_buffer if self.is_rnn else None,
            'last_rnn_states' : self.rnn_states,
            'mb_intr_rewards' : mb_intr_rewards if self.intr_reward_model is not None else None,
            'mb_extr_rewards' : mb_rewards,
        }
        return batch_dict, extras
    
    def augment_batch_for_mixed_expl(self, batch_dict, extras, repeat_idxs=None):
        new_batch_dict = {}
        num_blocks = self.num_actors // self.intr_coef_block_size
        if repeat_idxs is None:
            num_repeat = min(num_blocks, int(self.config['off_policy_ratio']) + 1)
            repeat_idxs = [0] + [int(x) for x in np.random.choice(range(1, self.num_actors // self.intr_coef_block_size), num_repeat-1, replace=False)]
            if self.multi_gpu:
                dist.broadcast_object_list(repeat_idxs, 0)
        for key, val in batch_dict.items():
            if key in ['played_frames', 'step_time']:
                new_batch_dict[key] = val
            elif key == 'obses':
                intr_coef_embd = torch.cat([torch.roll(self.intr_reward_coef_embd, self.intr_coef_block_size*i, dims=0) for i in repeat_idxs], dim=0)
                obses = torch.cat([val]*len(repeat_idxs), dim=0)
                obses[:, -self.intr_reward_coef_embd.shape[-1]:] = intr_coef_embd.repeat_interleave(self.horizon_length, dim=0)
                mask = torch.zeros(len(obses), dtype=torch.bool, device=obses.device)
                mask[len(val):] = True
                if self.use_others_experience == 'lf': # leader follower type update
                    obses = filter_leader(obses, len(val), repeat_idxs, num_blocks)
                    mask = filter_leader(mask, len(val), repeat_idxs, num_blocks)
                new_batch_dict[key] = obses
                new_batch_dict['off_policy_mask'] = mask
            elif key == 'states':
                intr_coef_embd = torch.cat([torch.roll(self.intr_reward_coef_embd, self.intr_coef_block_size*i, dims=0) for i in repeat_idxs], dim=0)
                states = torch.cat([val]*len(repeat_idxs), dim=0)
                states[:, -self.intr_reward_coef_embd.shape[-1]:] = intr_coef_embd.repeat_interleave(self.horizon_length, dim=0)
                if self.use_others_experience == 'lf':
                    states = filter_leader(states, len(val), repeat_idxs, num_blocks)
                new_batch_dict[key] = states
            elif key in ['values', 'returns']:
                pass  # handled below
            elif key == 'rnn_states':
                if val is not None:
                    new_batch_dict[key] = [torch.cat([val[i]]*len(repeat_idxs), dim=1) for i in range(len(val))]
                    if self.use_others_experience == 'lf':
                        new_batch_dict[key] = [filter_leader(new_batch_dict[key][i], val[i].shape[1], repeat_idxs, num_blocks) for i in range(len(val))]
                else:
                    new_batch_dict[key] = None
            else:
                new_batch_dict[key] = torch.cat([val]*len(repeat_idxs), dim=0)
                if self.use_others_experience == 'lf': # leader follower type update
                    new_batch_dict[key] = filter_leader(new_batch_dict[key], len(val), repeat_idxs, num_blocks)

        new_returns_list = [batch_dict['returns']]
        new_values_list = [batch_dict['values']]
        
        for r_k in repeat_idxs[1:]:
            mb_rewards = extras['rewards']
            mb_obs = extras['obs']
            last_obs_and_states = extras['last_obs']
            last_rnn_states = extras['last_rnn_states']
            mb_states = extras['states']
            mb_rnn_states = extras['rnn_states']
            
            mb_obs[:,:, -self.intr_reward_coef_embd.shape[-1]:] = torch.roll(self.intr_reward_coef_embd, self.intr_coef_block_size*r_k, dims=0)
            last_obs_and_states['obs'][:,-self.intr_reward_coef_embd.shape[-1]:] = torch.roll(self.intr_reward_coef_embd, self.intr_coef_block_size*r_k, dims=0)
            
            flattened_rnn_states = [rnn_s.transpose(0, 1).reshape(rnn_s.transpose(0, 1).shape[0], -1, *rnn_s.shape[3:]) for rnn_s in mb_rnn_states] if mb_rnn_states is not None else None

            flattened_mb_obs = mb_obs.reshape(-1, *mb_obs.shape[2:])
            flattened_mb_states = mb_states.reshape(-1, *mb_states.shape[2:]) if mb_states is not None else None
            
            mb_values = []
            for i in range((flattened_mb_obs.shape[0] + 8191) // 8192):
                mb_values.append(self.get_values({
                    'obs': flattened_mb_obs[i*8192:(i+1)*8192], 
                    'states': flattened_mb_states[i*8192:(i+1)*8192] if mb_states is not None else None
                    }, rnn_states=[s[:, i*8192:(i+1)*8192] for s in flattened_rnn_states] if flattened_rnn_states is not None else None))
            mb_values = torch.cat(mb_values, dim=0)
            last_values = self.get_values(last_obs_and_states, last_rnn_states)
            
            mb_values = mb_values.reshape(*mb_obs.shape[:2], *mb_values.shape[1:])
            mb_values = torch.cat([mb_values, last_values.unsqueeze(0)], dim=0)

            mb_fdones = extras['dones']
            fdones = extras['last_dones']

            mb_fdones = torch.cat([mb_fdones, fdones.unsqueeze(0)], dim=0)
            mb_returns = mb_rewards + (torch.roll(self.intr_reward_coef, self.intr_coef_block_size*r_k, dims=0).unsqueeze(0).unsqueeze(2) * extras['mb_intr_rewards'] if extras['mb_intr_rewards'] is not None else 0)  + self.gamma * mb_values[1:] * (1 - mb_fdones[1:]).unsqueeze(-1)
            
            new_returns_list.append(swap_and_flatten01(mb_returns))
            new_values_list.append(swap_and_flatten01(mb_values[:-1]))
        
        new_batch_dict['returns'] = torch.cat(new_returns_list, dim=0)
        new_batch_dict['values'] = torch.cat(new_values_list, dim=0)
        if self.use_others_experience == 'lf':
            new_batch_dict['returns'] = filter_leader(new_batch_dict['returns'], len(batch_dict['returns']), repeat_idxs, num_blocks)
            new_batch_dict['values'] = filter_leader(new_batch_dict['values'], len(batch_dict['values']), repeat_idxs, num_blocks)
        
        # reset obs and last obs in extras
        extras['obs'][:,:, -self.intr_reward_coef_embd.shape[-1]:] = self.intr_reward_coef_embd
        extras['last_obs']['obs'][:,-self.intr_reward_coef_embd.shape[-1]:] = self.intr_reward_coef_embd

        return new_batch_dict


class DiscreteA2CBase(A2CBase):

    def __init__(self, base_name, params):
        A2CBase.__init__(self, base_name, params)
    
        batch_size = self.num_agents * self.num_actors
        action_space = self.env_info['action_space']
        if type(action_space) is gym.spaces.Discrete:
            self.actions_shape = (self.horizon_length, batch_size)
            self.actions_num = action_space.n
            self.is_multi_discrete = False
        if type(action_space) is gym.spaces.Tuple:
            self.actions_shape = (self.horizon_length, batch_size, len(action_space)) 
            self.actions_num = [action.n for action in action_space]
            self.is_multi_discrete = True
        self.is_discrete = True

    def init_tensors(self):
        A2CBase.init_tensors(self)
        self.update_list = ['actions', 'neglogpacs', 'values']
        if self.use_action_masks:
            self.update_list += ['action_masks']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']

    def train_epoch(self):
        super().train_epoch()

        self.set_eval()
        play_time_start = time.time()

        with torch.no_grad():
            batch_dict = self.play_steps()

        self.set_train()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        a_losses = []
        c_losses = []
        entropies = []
        kls = []
        if self.has_central_value:
            self.train_central_value()

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.world_size

            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)
            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul

    def prepare_dataset(self, batch_dict):
        rnn_masks = batch_dict.get('rnn_masks', None)
        returns = batch_dict['returns']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        dones = batch_dict['dones']
        rnn_states = batch_dict.get('rnn_states', None)
        
        obses = batch_dict['obses']
        advantages = returns - values

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        
        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    if os.getenv("LOCAL_RANK") and os.getenv("WORLD_SIZE"):
                        mean, var = torch_ext.dist_mean_var_count(advantages.mean(), advantages.var(), len(advantages))
                        std = torch.sqrt(var)
                    else:
                        mean, std = advantages.mean(), advantages.std()
                    
                    advantages = (advantages - mean) / (std + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks

        if self.use_action_masks:
            dataset_dict['action_masks'] = batch_dict['action_masks']

        self.dataset.update_values_dict(dataset_dict)
        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['dones'] = dones
            dataset_dict['obs'] = batch_dict['states'] 
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train(self):
        self.init_tensors()
        start_time = time.time()
        self._start_checkpoint_schedule()
        self._install_exit_signal_handlers()
        total_time = 0
        rep_count = 0
        # self.frame = 0  # loading from checkpoint
        if self.obs is None:
            self.obs = self.env_reset()
        else:
            self.obs = self.obs_to_tensors(self.obs)

        if self.multi_gpu:
            torch.cuda.set_device(self.local_rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            if self._exit_signal is not None:
                self._save_exit_checkpoint()
                self._restore_exit_signal_handlers()
                return self.last_mean_rewards, self.epoch_num
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            total_time += sum_time
            curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
            self.frame += curr_frames
            should_exit = False

            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch = epoch_num)
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time

                frame = self.frame // self.num_agents

                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, 
                                scaled_time, scaled_play_time, curr_frames)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                periodic_mean_reward = (
                    self.game_rewards.get_mean()[0]
                    if self.game_rewards.current_size > 0
                    else self.mean_rewards
                )
                periodic_checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(periodic_mean_reward)
                self._save_periodic_checkpoints(periodic_checkpoint_name)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], frame)


                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, frame)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    # removed equal signs (i.e. "rew=") from the checkpoint name since it messes with hydra CLI parsing
                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))
                        torch_ext.safe_filesystem_op(os.makedirs, os.path.join(self.experiment_dir, 'best'), exist_ok=True)
                        torch_ext.safe_symlink(os.path.relpath(os.path.join(self.nn_dir, self.config['name'] + '.pth'), start=os.path.join(self.experiment_dir, 'best')), os.path.join(self.experiment_dir, 'best', 'model.pth'))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.bool().item()

            if should_exit:
                self._exit_checkpoint_saved = True
                self._restore_exit_signal_handlers()
                return self.last_mean_rewards, epoch_num


class ContinuousA2CBase(A2CBase):

    def __init__(self, base_name, params):
        A2CBase.__init__(self, base_name, params)

        self.is_discrete = False
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.bounds_loss_coef = self.config.get('bounds_loss_coef', None)

        self.clip_actions = self.config.get('clip_actions', True)

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)

    def preprocess_actions(self, actions):
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

    def init_tensors(self):
        A2CBase.init_tensors(self)
        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']

    def train_epoch(self):
        super().train_epoch()

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            orig_batch_dict, ps_extras = self.play_steps()
            
            if self.expl_type.startswith('mixed_expl') and self.use_others_experience != 'none':
                batch_dict = self.augment_batch_for_mixed_expl(orig_batch_dict, ps_extras)
            else:
                batch_dict = orig_batch_dict
            if self.expl_type.startswith('mixed_expl'):
                batch_dict = shuffle_batch(batch_dict, self.seq_length)

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        if getattr(self, '_discard_post_resume_rollout', False):
            # First rollout after a checkpoint restore: the simulator's
            # internal solver state is not restorable, so this batch is
            # polluted. It was collected and counted normally; skip the PPO
            # update on it (audit 2026-08-29: c_loss shock 10-66x otherwise).
            self._discard_post_resume_rollout = False
            print('[warm-resume] first post-resume rollout discarded: '
                  'PPO update skipped, frames still counted')
            zero = torch.zeros(1, device=self.ppo_device)
            play_time = play_time_end - play_time_start
            extra_infos = {
                'on_policy_contrib': [0.0],
                'off_policy_contrib': [0.0],
                'on_policy_grads': [torch.zeros(1)],
                'off_policy_grads': [torch.zeros(1)],
                # [] on purpose: the epoch logger indexes entropies as
                # [minibatch, block] and skips the block loop only on the
                # empty list; a scalar stub is 1-D and IndexErrors there.
                'entropies': [],
                'mb_intr_rewards': ps_extras['mb_intr_rewards'],
                'mb_extr_rewards': ps_extras['rewards'],
            }
            return (batch_dict['step_time'], play_time, 0.0, play_time,
                    [zero], [zero], [zero], [zero], [zero],
                    self.last_lr, 1.0, extra_infos)
        self.prepare_dataset(batch_dict)

        ret_val = self.algo_observer.after_steps()
        if isinstance(ret_val, DictConfig):
            return ret_val
        
        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        extra_infos = {
            'on_policy_contrib' : [],
            'off_policy_contrib' : [],
            'on_policy_grads' : [],
            'off_policy_grads' : [],
            'entropies' : [],
            'mb_intr_rewards' : ps_extras['mb_intr_rewards'],
            'mb_extr_rewards' :ps_extras['rewards']
        }

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss, extras = self.train_actor_critic(self.dataset[i])
                extra_infos['on_policy_contrib'].append(extras['on_policy_contrib'])
                extra_infos['on_policy_grads'].append(extras['on_policy_grads'])
                extra_infos['off_policy_contrib'].append(extras['off_policy_contrib'])
                extra_infos['off_policy_grads'].append(extras['off_policy_grads'])
                if 'entropies' in extras:
                    extra_infos['entropies'].append(extras['entropies'])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    av_kls = kl
                    if self.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.world_size
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.world_size
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start
        DECIMALS = 3
        print("\nTiming:")
        print(f"  Play time               : {play_time:.{DECIMALS}f} s")
        print(f"  Update time             : {update_time:.{DECIMALS}f} s")
        print(f"  Time to train epoch     : {total_time:.{DECIMALS}f} s\n")

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul, extra_infos

    def prepare_dataset(self, batch_dict, train_value_mean_std=True):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = returns - values

        if self.normalize_value:
            if train_value_mean_std:
                self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    if os.getenv("LOCAL_RANK") and os.getenv("WORLD_SIZE"):
                        mean, var, _ = torch_ext.dist_mean_var_count(advantages.mean(), advantages.var(), len(advantages))
                        std = torch.sqrt(var)
                    else:
                        mean, std = advantages.mean(), advantages.std()
                    advantages = (advantages - mean) / (std + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas
        dataset_dict['off_policy_mask'] = batch_dict.get('off_policy_mask', None)

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['dones'] = dones
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train(self):
        self.init_tensors()
        start_time = time.time()
        self._start_checkpoint_schedule()
        self._install_exit_signal_handlers()
        total_time = 0
        rep_count = 0
        if self.obs is None:
            self.obs = self.env_reset()
        else:
            self.obs = self.obs_to_tensors(self.obs)
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])
            
        # for _ in range(0):
        #     self.play_steps()
        # print("Warmup done")

        while True:
            if self._exit_signal is not None:
                self._save_exit_checkpoint()
                self._restore_exit_signal_handlers()
                return self.last_mean_rewards, self.epoch_num
            epoch_num = self.update_epoch()
            ret_val = self.train_epoch()
            if isinstance(ret_val, DictConfig):
                return (ret_val, self.vec_env)
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul, extra_infos = ret_val
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch = epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame,
                                scaled_time, scaled_play_time, curr_frames)

                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                if self.multi_gpu:
                    # gather state from all gpus
                    state = self.get_full_state_weights()
                    state_list = [None] * self.world_size
                    dist.gather_object(state, state_list)
                    all_state_dict = {i: state_list[i] for i in range(self.world_size)}
                else:
                    all_state_dict = None

                periodic_mean_reward = (
                    self.game_rewards.get_mean()[0]
                    if self.game_rewards.current_size > 0
                    else self.mean_rewards
                )
                periodic_checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(periodic_mean_reward)
                self._save_periodic_checkpoints(periodic_checkpoint_name, all_state_dict)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], frame)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, frame)

                    self.writer.add_histogram('auxiliary_stats/off_policy_contrib', np.array(extra_infos['off_policy_contrib']), frame)
                    self.writer.add_histogram('auxiliary_stats/on_policy_contrib', np.array(extra_infos['on_policy_contrib']), frame)

                    # These split gradients are only measured in explicit debug
                    # mode. The default path supplies a zero off-policy vector,
                    # so comparing it with AMP-scaled gradients is meaningless.
                    if os.getenv('LOG_OFF_POLICY_GRADS'):
                        on_policy_grads = torch.stack(extra_infos['on_policy_grads'])
                        off_policy_grads = torch.stack(extra_infos['off_policy_grads'])
                        self.writer.add_scalar('auxiliary_stats/off_on_grad_similarity', torch.cosine_similarity(on_policy_grads, off_policy_grads).diag().mean(),frame)
                        self.writer.add_scalar('auxiliary_stats/off_on_relative_grad_norms', torch.norm(off_policy_grads, dim=-1).mean()/torch.norm(on_policy_grads, dim=-1).mean(), frame)
                    
                    if extra_infos['mb_intr_rewards'] is not None:
                        if hasattr(self, 'intr_coef_block_size'):
                            for bl in range(self.num_actors // self.intr_coef_block_size):
                                self.writer.add_scalar(f'intr_rewards/block_{bl}', extra_infos['mb_intr_rewards'][:,self.intr_coef_block_size*bl:self.intr_coef_block_size*(bl+1)].mean(), frame)
                        else:
                            self.writer.add_scalar(f'intr_rewards/block_0', extra_infos['mb_intr_rewards'].mean(), frame)
                        self.writer.add_scalar(f'intr_rewards/extr_rewards', extra_infos['mb_extr_rewards'].mean(), frame)
                    
                    if extra_infos['entropies'] != []:
                        for bl in range(self.num_actors // self.intr_coef_block_size):
                            self.writer.add_scalar(f'intr_rewards/entropy_block_{bl}', torch.tensor(extra_infos['entropies'])[:, bl].mean(), frame)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= 10:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']), all_state_dict)
                        torch_ext.safe_filesystem_op(os.makedirs, os.path.join(self.experiment_dir, 'best'), exist_ok=True)
                        torch_ext.safe_symlink(os.path.relpath(os.path.join(self.nn_dir, self.config['name'] + '.pth'), start=os.path.join(self.experiment_dir, 'best')), os.path.join(self.experiment_dir, 'best', 'model.pth'))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name), all_state_dict)
                                should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')), all_state_dict)
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')), all_state_dict)
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0
            else:
                if self.multi_gpu:
                    state = self.get_full_state_weights()
                    dist.gather_object(state)

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                self._exit_checkpoint_saved = True
                self._restore_exit_signal_handlers()
                return self.last_mean_rewards, epoch_num

            if should_exit:
                self._exit_checkpoint_saved = True
                self._restore_exit_signal_handlers()
                return self.last_mean_rewards, epoch_num

            # print("Epoch done")
            # time.sleep(3)
