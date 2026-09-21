import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import math
import torch.distributed as dist
import time

numpy_to_torch_dtype_dict = {
    np.dtype('bool')       : torch.bool,
    np.dtype('uint8')      : torch.uint8,
    np.dtype('int8')       : torch.int8,
    np.dtype('int16')      : torch.int16,
    np.dtype('int32')      : torch.int32,
    np.dtype('int64')      : torch.int64,
    np.dtype('float16')    : torch.float16,
    np.dtype('float64')    : torch.float32,
    np.dtype('float32')    : torch.float32,
    #np.dtype('float64')    : torch.float64,
    np.dtype('complex64')  : torch.complex64,
    np.dtype('complex128') : torch.complex128,
}

torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}

def policy_kl_eigadd(p0_mu, p0_sigma, p0_eigen_sigma,
                     p1_mu, p1_sigma, p1_eigen_sigma, basis, reduce=True):
    """Exact KL(p0 || p1) for diag(sigma**2) + B.T @ diag(eigen_sigma**2) @ B.

    The basis is fixed; both IID and eigen scales can differ per sample.
    Use K-dimensional capacitance matrices rather than dense action-space
    inverses. Keep the pre-update eigen scales with the old mean and sigma.
    """
    # AMP may produce half-precision means. KL is a scheduler statistic;
    # evaluate its solves and covariance differences in at least float32.
    dtype = torch.promote_types(p0_mu.dtype, p0_sigma.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    with torch.autocast(device_type=p0_mu.device.type, enabled=False):
        mu0, std0, s0, mu1, std1, s1, B = [
            t.to(dtype=dtype) for t in
            (p0_mu, p0_sigma, p0_eigen_sigma,
             p1_mu, p1_sigma, p1_eigen_sigma, basis)]
        var0, var1 = std0.square(), std1.square()
        invvar1 = var1.reciprocal()
        gram0 = (B * var0.reciprocal().unsqueeze(-2)) @ B.t()
        weighted_basis = B * invvar1.unsqueeze(-2)
        gram1 = weighted_basis @ B.t()
        eye = torch.eye(B.shape[0], device=B.device, dtype=dtype)
        chol0 = torch.linalg.cholesky(
            eye + s0.unsqueeze(-1) * gram0 * s0.unsqueeze(-2))
        chol1 = torch.linalg.cholesky(
            eye + s1.unsqueeze(-1) * gram1 * s1.unsqueeze(-2))
        A = s1.unsqueeze(-1) * weighted_basis
        solved = torch.cholesky_solve(A, chol1)
        # Diagonals of Sigma1^-1 and B Sigma1^-1 B.T via Woodbury.
        precision_diag = invvar1 - (A * solved).sum(dim=-2)
        basis_precision_diag = torch.diagonal(gram1, dim1=-2, dim2=-1) - (
            (s1.unsqueeze(-1) * gram1) * (solved @ B.t())).sum(dim=-2)
        # tr(Sigma1^-1 (Sigma0 - Sigma1)) avoids subtracting n near KL=0.
        trace_delta = ((var0 - var1) * precision_diag).sum(dim=-1)
        trace_delta += ((s0.square() - s1.square()) * basis_precision_diag).sum(dim=-1)
        delta = mu1 - mu0
        projected = (A @ delta.unsqueeze(-1)).squeeze(-1)
        quadratic = (delta.square() * invvar1).sum(dim=-1) - (
            projected * (solved @ delta.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
        logdet_delta = 2.0 * (
            (std1.log() - std0.log()).sum(dim=-1)
            + (torch.diagonal(chol1, dim1=-2, dim2=-1).log()
               - torch.diagonal(chol0, dim1=-2, dim2=-1).log()).sum(dim=-1))
        kl = (0.5 * (trace_delta + quadratic + logdet_delta)).clamp_min(0.0)
        return kl.mean() if reduce else kl


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=True):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu)**2)/(2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1) # returning mean between all steps of sum between all actions
    if reduce:
        return kl.mean()
    else:
        return kl

def mean_mask(input, mask, sum_mask):
    return (input * rnn_masks).sum() / sum_mask

def shape_whc_to_cwh(shape):
    if len(shape) == 3:
        return (shape[2], shape[0], shape[1])
    
    return shape


def shape_cwh_to_whc(shape):
    if len(shape) == 3:
        return (shape[1], shape[2], shape[0])

    return shape

def safe_filesystem_op(func, *args, **kwargs):
    """
    This is to prevent spurious crashes related to saving checkpoints or restoring from checkpoints in a Network
    Filesystem environment (i.e. NGC cloud or SLURM)
    """
    num_attempts = 5
    for attempt in range(num_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            print(f'Exception {exc} when trying to execute {func} with args:{args} and kwargs:{kwargs}...')
            wait_sec = 2 ** attempt
            print(f'Waiting {wait_sec} before trying again...')
            time.sleep(wait_sec)

    raise RuntimeError(f'Could not execute {func}, give up after {num_attempts} attempts...')

def safe_symlink(src, dst):
    def remove_if_present():
        try:
            os.remove(dst)
        except FileNotFoundError:
            pass
    safe_filesystem_op(remove_if_present)
    safe_filesystem_op(os.symlink, src, dst)

def safe_save(state, filename):
    # Atomic save: serialize to a scratch file in the same directory, then
    # os.replace it onto the final name so a torn write can never leave a
    # truncated checkpoint at the destination path.
    tmp_filename = str(filename) + '.tmp'
    safe_filesystem_op(torch.save, state, tmp_filename)
    return safe_filesystem_op(os.replace, tmp_filename, filename)

def safe_load(filename):
    return safe_filesystem_op(torch.load, filename, weights_only=False, map_location='cpu')

def save_checkpoint(filename, state):
    print("=> saving checkpoint '{}'".format(filename + '.pth'))
    safe_save(state, filename + '.pth')

def load_checkpoint(filename):
    print("=> loading checkpoint '{}'".format(filename))
    try:
        state = safe_load(filename)
    except:
        print("=> failed to load checkpoint '{}'. Trying from {}.old".format(filename, filename))
        state = safe_load(filename + '.old')
    return state

def parameterized_truncated_normal(uniform, mu, sigma, a, b):
    normal = torch.distributions.normal.Normal(0, 1, validate_args=False)

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    alpha_normal_cdf = normal.cdf(torch.from_numpy(np.array(alpha)))
    p = alpha_normal_cdf + (normal.cdf(torch.from_numpy(np.array(beta))) - alpha_normal_cdf) * uniform

    p = p.numpy()
    one = np.array(1, dtype=p.dtype)
    epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
    v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
    x = torch.clamp(x, a, b)

    return x

def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
    return parameterized_truncated_normal(uniform, mu, sigma, a, b)

def sample_truncated_normal(shape=(), mu=0.0, sigma=1.0, a=-2, b=2):
    return truncated_normal(torch.from_numpy(np.random.uniform(0, 1, shape)), mu, sigma, a, b)

def variance_scaling_initializer(tensor, mode='fan_in',scale = 2.0):
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    print(fan, scale)
    sigma = np.sqrt(scale / fan)
    with torch.no_grad():
        tensor[:] = sample_truncated_normal(tensor.size(), sigma=sigma)
        return tensor


def random_sample(obs_batch, prob):
    num_batches = obs_batch.size()[0]
    permutation = torch.randperm(num_batches, device=obs_batch.device)
    start = 0
    end = int(prob * num_batches)
    indices = permutation[start:end]
    return torch.index_select(obs_batch, 0, indices)

def mean_list(val):
    return torch.mean(torch.stack(val))

def apply_masks(losses, mask=None):
    sum_mask = None
    if mask is not None:
        mask = mask.unsqueeze(1)
        sum_mask = mask.numel()#
        #sum_mask = mask.sum()
        res_losses = [(l * mask).sum() / sum_mask for l in losses]
    else:
        res_losses = [torch.mean(l) for l in losses]
    
    return res_losses, sum_mask

def dist_mean_var_count(mean: torch.Tensor, var: torch.Tensor, count: int):
    square_sum = var * (count - 1) + (mean ** 2) * count
    sum = mean * count
    count = torch.tensor(count, dtype=torch.float64).to(mean.device)
    dist.all_reduce(square_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    count = int(count.item())
    mean = sum / count
    var = (square_sum - (mean ** 2) * count) / (count - 1)
    return mean, var, count

def normalization_with_masks(values, masks):
    if masks is None:
        if os.getenv("LOCAL_RANK") and os.getenv("WORLD_SIZE"): # multi-gpu
            mean, var, _ = dist_mean_var_count(values.mean(), values.var(), len(values))
            std = torch.sqrt(var)
        else:
            mean = values.mean()
            std = values.std()
        return (values - mean) / (std + 1e-8)
    
    if os.getenv("LOCAL_RANK") and os.getenv("WORLD_SIZE"):
        mean, var = get_mean_var_with_masks(values, masks)
        values_mean, values_var, _ = dist_mean_var_count(mean, var, masks.sum())
    else:
        values_mean, values_var = get_mean_var_with_masks(values, masks)
    values_std = torch.sqrt(values_var)
    normalized_values = (values - values_mean) / (values_std + 1e-8)

    return normalized_values

def get_mean_var_with_masks(values, masks):
    sum_mask = masks.sum()
    values_mask = values * masks
    values_mean = values_mask.sum() / sum_mask
    min_sqr = ((((values_mask)**2)/sum_mask).sum() - ((values_mask/sum_mask).sum())**2)
    values_var = min_sqr * sum_mask / (sum_mask-1)
    return values_mean, values_var

def explained_variance(y_pred,y, masks=None):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """

    if masks is not None:
        masks = masks.unsqueeze(1)
        _, var_y = get_mean_var_with_masks(y_pred,masks)
        _, var_dy = get_mean_var_with_masks(y-y_pred, masks)
    else:
        var_y = torch.var(y)
        var_dy = torch.var(y-y_pred)
    return 1.0 - var_dy/var_y

def policy_clip_fraction(new_neglogp, old_neglogp, clip_param, masks=None):
    logratio = old_neglogp - new_neglogp
    clip_frac = torch.logical_or(
                logratio < math.log(1.0 - clip_param),
                logratio > math.log(1.0 + clip_param),
            ).float()
    if masks is not None:
        clip_frac = clip_frac * masks/masks.sum()
    else:
        clip_frac = clip_frac.mean()
    return clip_frac
    
class CoordConv2d(nn.Conv2d):
    pool = {}
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels + 2, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
    @staticmethod
    def get_coord(x):
        key = int(x.size(0)), int(x.size(2)), int(x.size(3)), x.type()
        if key not in CoordConv2d.pool:
            theta = torch.Tensor([[[1, 0, 0], [0, 1, 0]]])
            coord = torch.nn.functional.affine_grid(theta, torch.Size([1, 1, x.size(2), x.size(3)])).permute([0, 3, 1, 2]).repeat(
                x.size(0), 1, 1, 1).type_as(x)
            CoordConv2d.pool[key] = coord
        return CoordConv2d.pool[key]
    def forward(self, x):
        return torch.nn.functional.conv2d(torch.cat([x, self.get_coord(x).type_as(x)], 1), self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class LayerNorm2d(nn.Module):
    """
    Layer norm the just works on the channel axis for a Conv2d
    Ref:
    - code modified from https://github.com/Scitator/Run-Skeleton-Run/blob/master/common/modules/LayerNorm.py
    - paper: https://arxiv.org/abs/1607.06450
    Usage:
        ln = LayerNormConv(3)
        x = Variable(torch.rand((1,3,4,2)))
        ln(x).size()
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.register_buffer("gamma", torch.ones(features).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("beta", torch.ones(features).unsqueeze(-1).unsqueeze(-1))

        self.eps = eps
        self.features = features

    def _check_input_dim(self, input):
        if input.size(1) != self.gamma.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.features))

    def forward(self, x):
        self._check_input_dim(x)
        x_flat = x.transpose(1,-1).contiguous().view((-1, x.size(1)))
        mean = x_flat.mean(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        std = x_flat.std(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)



class DiscreteActionsEncoder(nn.Module):
    def __init__(self, actions_max, mlp_out, emb_size, num_agents, use_embedding):
        super().__init__()
        self.actions_max = actions_max
        self.emb_size = emb_size
        self.num_agents = num_agents
        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding = torch.nn.Embedding(actions_max, emb_size)
        else:
            self.emb_size = actions_max
        
        self.linear = torch.nn.Linear(self.emb_size * num_agents, mlp_out)

    def forward(self, discrete_actions):
        if self.use_embedding:
            emb = self.embedding(discrete_actions)
        else:
            emb = torch.nn.functional.one_hot(discrete_actions, num_classes=self.actions_max)
        emb = emb.view( -1, self.emb_size * self.num_agents).float()
        emb = self.linear(emb)
        return emb

def get_model_gradients(model):
    grad_list = []
    for param in model.parameters():
        grad_list.append(param.grad)
    return grad_list

def get_mean(v):
    if len(v) > 0:
        mean = np.mean(v)
    else:
        mean = 0
    return mean


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.register_buffer("current_size", torch.tensor(0, dtype = torch.int32))
        self.register_buffer("mean", torch.zeros(in_shape, dtype = torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0).reshape(self.mean.shape)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = torch.tensor(size_sum, dtype = torch.int32)
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = torch.tensor(0, dtype = torch.int32)
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size.item()

    def get_mean(self):
        return self.mean.cpu().numpy()


class IdentityRNN(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(IdentityRNN, self).__init__()
        assert(in_shape == out_shape)
        self.identity = torch.nn.Identity()

    def forward(self, x, h):
        return self.identity(x), h

 
