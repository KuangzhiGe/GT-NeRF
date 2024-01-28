import os
import sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from llff_LOADER import load_llff_data
from blender_loader import load_blender_data, D_load_blender_data
from NeRF_model import NeRF
from Positional_Encoding import get_embedder
from ray import get_rays_numpy, get_rays_torch, ndc_rays
from Hierarchical_Sampling import Hierarchical_Sampling
try:
    from apex import amp
except ImportError:
    pass

# img loss functions
mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

# img transfer
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

# 初始化设备device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化随机种子
np.random.seed(0)
DEBUG = False
# 设置tensor的默认数据类型
torch.set_default_tensor_type('torch.cuda.FloatTensor')

is_straightforward = False
'''
# 一些需要提前设置的参数
log_dir  = './logs/NeRF'    # 日志目录(保存checkpoints & logs)
exp_name = 'paper_ficus_highres' # 实验名称

# # Dataset Options
dataset_type = 'blender'                          # 数据库的类型
database_dir = './data/nerf_synthetic/ficus' # 数据库地址
testskip     = 8                               # 加载数据库时只会选取(1/testskip)数量的test/val集数据

# # Blender Dataset Options
if_white_bkgd = True  # 是否使用白色背景(将透明部分改为白色)
half_res      = False # 缩小图像数据的大小为原来的一半(800x800->400x400)

# # LLFF Dataset Options
factor   = 2     # 对LLFF数据集图片的降采样倍率
lindisp  = False # 在disparity上线性采样，而不是在depth上均匀采样
no_ndc   = False # 不使用标准化设备坐标
spherify = False # 对于360旋转的场景则启用
llffhold = 4     # 选取1/llffhold个数的图片作为测试集

# # Training Options
NeRF_Depth      = 8         # 网络的层数
NeRF_Width      = 256       # 网络的维度
NeRF_fine_Depth = 8         # 精细网络的层数
NeRF_fine_Width = 256       # 精细网络的维度
lr_rate         = 5e-4      # 学习率
lr_rate_decay   = 500       # 学习率衰减系数
chunk           = 1024 * 32 # 同时处理的小批量的光线数量，防止爆内存
netchunk        = 1024 * 64 # 网络同时处理的坐标数量，同样防止爆内存
no_batching     = True      # 是否每次只从单幅图随机选取一些光线
N_rand          = 1024      # 即batch size, 每一步梯度下降随机选取的光线数量
precrop_iters   = 500       # 在central crops上需要训练的次数
precrop_frac    = 0.5       # 图像被选取的部分的百分比大小, for central crops
no_reload       = True      # 不从已保存的checkpoint读取weight数据
ft_path         = None      # 加载checkpoint的文件地址

# # Rendering Options
N_samples       = 64    # 每条光线粗采样的数量
N_importance    = 128   # 每条光线额外细采样的数量(需要额外的fine_NeRF)
perturb         = 1.    # 0. for no jitter ; 1. for jitter
use_viewdirs    = True  # 是否需要论文所述的完整的5D input, 否则采用3D input
L_position      = 10    # Positional Encoding Dimension for Position vector
L_view          = 4     # Positional Encoding Dimension for View vector
raw_noise_std   = 0.    # std dev of noise added to regularize sigma_a output
render_only     = False # 仅渲染，不进行优化，读取weights数据，按照render_poses渲染
render_for_test = False # render_poses替换为渲染test集的poses
render_factor   = 0     # downsampling factor, 加速rendering

# # Logging Options
i_print   = 100   # 每i次训练, 打印训练日志
i_img     = 500   # 每i次训练, 输出图片日志
i_weights = 2000 # 每i次训练, 保存checkpoints阶段性成果
i_testset = 50000 # 每i次训练, 保存test set
i_video   = 50000 # 每i次训练, 使用render_poses生成video
'''
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--nerf_type", type=str, default="original",
                        help='nerf network type')
    parser.add_argument("--is_straightforward", type=bool, default=False,
                        help='if use 6D input instead of time net')
    parser.add_argument("--is_ViT", type=bool, default=False,
                        help='if use cross-attention')

    parser.add_argument("--N_iter", type=int, default=500000,
                        help='num training iterations')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--do_half_precision", action='store_true',
                        help='do half precision training and inference')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--not_zero_canonical", action='store_true',
                        help='if set zero time is not the canonic space')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', default=True,
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--L_positions", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--L_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--input_dims", type=int, default=3,
                        help='input dimension')
    parser.add_argument("--input_dims_view", type=int, default=1,
                        help='input dimension of view')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--use_two_models_for_fine", action='store_true',
                        help='use two models for fine results')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_iters_time", type=int, default=0,
                        help='number of steps to train on central time')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--add_tv_loss", action='store_true',
                        help='evaluate tv loss')
    parser.add_argument("--tv_loss_weight", type=float,
                        default=1.e-4, help='weight of tv loss')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=2,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", type=bool, default= False,
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=10000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=200000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=200000,
                        help='frequency of render_poses video saving')

    return parser

def batchify(fn, chunk):
    # fn : 视为一个函数
    # chunk : 块大小
    # 函数生成一个新版本的fn, 使得其先一个个处理大小为chunk的输入，再合并输出(即使函数小批量地处理input)
    if chunk is None:
        return fn

    def ret(inputs_pos):
        num_batches = inputs_pos.shape[0]

        out_list = []
        for i in range(0, num_batches, chunk):
            out, dx = fn(inputs_pos[i:i+chunk])
            out_list += [out]
        return torch.cat(out_list, 0)
    return ret

def D_batchify(fn, chunk):
    """
    Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs_pos, inputs_time):
        num_batches = inputs_pos.shape[0]

        out_list = []
        dx_list = []
        for i in range(0, num_batches, chunk):
            out, dx = fn(inputs_pos[i:i+chunk], [inputs_time[0][i:i+chunk], inputs_time[1][i:i+chunk]])
            out_list += [out]
            dx_list += [dx]
        return torch.cat(out_list, 0), torch.cat(dx_list, 0)
    return ret

def run_network(inputs, views, fn, embed_pts_fn, embed_views_fn, netchunk=1024 * 64):
    # 对inputs进行预处理(Positional Encoding)
    # 然后应用batchify小批量地一个个处理inputs
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded_pts = embed_pts_fn(inputs_flat)
    embedded = embedded_pts
    
    if views is not None:
        inputs_views = views[:, None].expand(inputs.shape)
        inputs_views_flat = torch.reshape(inputs_views, [-1, inputs_views.shape[-1]])
        embedded_views = embed_views_fn(inputs_views_flat)
        embedded = torch.cat([embedded_pts, embedded_views], dim=-1)
    
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def D_run_network(inputs, viewdirs, frame_time, fn, embed_fn, embeddirs_fn, embedtime_fn, netchunk=1024*64,
                embd_time_discr=True):
    """Prepares inputs and applies network 'fn'.
    inputs: N_rays x N_points_per_ray x 3
    viewdirs: N_rays x 3
    frame_time: N_rays x 1
    """
    assert len(torch.unique(frame_time)) == 1, "Only accepts all points from same time"
    cur_time = torch.unique(frame_time)[0]

    # embed position
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    # embed time
    if embd_time_discr:
        B, N, _ = inputs.shape
        input_frame_time = frame_time[:, None].expand([B, N, 1])
        input_frame_time_flat = torch.reshape(input_frame_time, [-1, 1])
        embedded_time = embedtime_fn(input_frame_time_flat)
        embedded_times = [embedded_time, embedded_time]

    else:
        assert NotImplementedError

    # embed views
    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    # 实际上是调用的 fn
    outputs_flat, position_delta_flat = D_batchify(fn, netchunk)(embedded, embedded_times)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    position_delta = torch.reshape(position_delta_flat, list(inputs.shape[:-1]) + [position_delta_flat.shape[-1]])
    return outputs, position_delta

# 将模型输出raw转化为语义上有意义的值
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=True, pytest=False):
    # raw : 模型的原始输出, 大小为[num_rays, num_samples per ray, 4]
    # z_vals : 分段积分标记, 大小为[num_rays, num_samples per ray]
    # rays_d : 每条光线的方向, 大小为[num_rays, 3]
    # pytest : ???
    
    # raw_noise_std : std dev of noise added to regularize sigma_a output(存疑)
    # white_bkgd : 是否应用白色背景
    
    # 预定义求解alpha的公式
    raw2alpha = lambda raw, dists, active_fn=F.relu: 1. - torch.exp(-active_fn(raw) * dists)
    
    # dists : 分段积分中, 每一小段的长度
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # 将dists大小调整为[N_rays, N_samples](最后补一维)
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], dim=-1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # rgb : [N_rays, N_samples, 3]
    rgb = torch.sigmoid(raw[..., :3])
    
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    
    # 计算alpha
    # alpha shape : [N_rays, N_samples]
    alpha = raw2alpha(raw[..., 3] + noise, dists)
    
    # 使用alpha计算weights
    # torch.cumprod() 沿dim进行连乘操作
    # Question : 为什么要进行torch.cat()操作, 为什么不直接进行连乘?
    # weights shape : [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]

    # rgb_map shape : [N_rays, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    
    # depth_map shape : [N_rays]
    depth_map = torch.sum(weights * z_vals, dim=-1)
    
    # disparity_map shape : [N_rays]
    disparity_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, dim=-1))
    
    # acc_map shape : [N_rays]
    acc_map = torch.sum(weights, dim=-1)
    
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
    
    # rgb_map : [N_rays, 3] 每条光线的RGB颜色
    # disparity_map : [N_rays] depth map的倒数
    # acc_map : [N_rays] 每条光线上weights的和
    # weights : [N_rays, N_samples] 每条光线上每个采样点的权重
    # depth_map : [N_rays] 每条光线到达物体的估计距离
    return rgb_map, disparity_map, acc_map, weights, depth_map

# 渲染一组光线(体素渲染 Volumetric Rendering 实现)
def render_rays(ray_batch, network_fn,
                network_fn_chunky, N_samples=64,
                retraw=False, lindisp=False,
                perturb=0., N_importance=0,
                white_bkgd=True,
                raw_noise_std=0.,
                verbose=False, network_fine=None, pytest=False,
                z_vals=None, use_two_models_for_fine=False
                ):
    # ray_batch : 用于渲染的一组光线，包含渲染需要的所有信息
    # ray_batch 中每一条光线数据结构如下：
    # [rays_o, rays_d, near, far, viewdirs]
    # network_fn : NeRF模型函数
    # network_fn_chunky : 小批量处理inputs的NeRF模型函数
    # retraw : return中是否含有模型输出raw数据
    # net_work_fine : 与NeRF网络相同的精细NeRF网络
    # pytest : ???

    # N_samples : 每条光线上的采样次数
    # N_importance : 每条光线需要额外采样的次数。这些采样仅会输入给network_fine, 即精细NeRF网络
    # perturb : 0. or 1. 若非零，则分层随机抽样
    # white_bkgd : bool, True代表白色背景
    # raw_noise_std : std dev of noise added to regularize sigma_a output(存疑)
    # lindisp : 是否在disparity上均匀采样(否则在depth上均匀采样)

    # ray_batch中光线数量
    N_rays = ray_batch.shape[0]

    # 从ray_batch中读取所需数据
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1, 1]

    if z_vals == None:
        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp:
            # 在depth上均匀采样
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            # 在disparity上均匀采样
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        # 增加随机扰动
        if perturb > 0.:
            # 计算采样区域的插值
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)

            # (0, 1)随机采样点生成
            t_rand = torch.rand(z_vals.shape)
            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

    # rays_o : [N_rays, 3]
    # rays_d : [N_rays, 3]
    # z_vals : [N_rays, N_samples]
    # pts : [N_rays, N_samples, 3]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    raw = network_fn_chunky(pts, viewdirs, network_fn)
    rgb_map, disparity_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest)

    if N_importance > 0:
        rgb_map_0, disparity_map_0, acc_map_0 = rgb_map, disparity_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = Hierarchical_Sampling(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        # pts shape : [N_rays, N_samples + N_importance, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        run_fn = network_fn if network_fine is None else network_fine

        raw = network_fn_chunky(pts, viewdirs, run_fn)

        rgb_map, disparity_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest)

    ret = {
        'rgb_map': rgb_map,
        'disparity_map': disparity_map,
        'acc_map': acc_map
    }

    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disparity_map0'] = disparity_map_0
        ret['acc_map0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    return ret

def D_render_rays(ray_batch, network_fn,
                network_query_fn, N_samples=64,
                retraw=False, lindisp=False,
                perturb=0., N_importance=0,
                network_fine=None, white_bkgd=False,
                raw_noise_std=0.,
                verbose=False, pytest=False, z_vals=None,
                use_two_models_for_fine=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None
    bounds = torch.reshape(ray_batch[...,6:9], [-1,1,3])
    near, far, frame_time = bounds[...,0], bounds[...,1], bounds[...,2] # [-1,1]
    z_samples = None
    rgb_map_0, disp_map_0, acc_map_0, position_delta_0 = None, None, None, None

    if z_vals is None:
        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


        if N_importance <= 0:
            raw, position_delta = network_query_fn(pts, viewdirs, frame_time, network_fn)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        else:
            if use_two_models_for_fine:
                raw, position_delta_0 = network_query_fn(pts, viewdirs, frame_time, network_fn)
                rgb_map_0, disp_map_0, acc_map_0, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            else:
                with torch.no_grad():
                    raw, _ = network_query_fn(pts, viewdirs, frame_time, network_fn)
                    _, _, _, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = Hierarchical_Sampling(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    run_fn = network_fn if network_fine is None else network_fine
    raw, position_delta = network_query_fn(pts, viewdirs, frame_time, run_fn)
    rgb_map, disp_map, acc_map, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'z_vals' : z_vals,
           'position_delta' : position_delta}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        if rgb_map_0 is not None:
            ret['rgb0'] = rgb_map_0
        if disp_map_0 is not None:
            ret['disp0'] = disp_map_0
        if acc_map_0 is not None:
            ret['acc0'] = acc_map_0
        if position_delta_0 is not None:
            ret['position_delta_0'] = position_delta_0
        if z_samples is not None:
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

# 分小批次渲染一整组光线
def batchify_rays(rays_flat, retraw=False, chunk=1024*32, **kwargs):
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], retraw=retraw, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k : torch.cat(all_ret[k], dim=0) for k in all_ret}
    return all_ret

def D_batchify_rays(rays_flat, retraw=False, chunk=1024*32, **kwargs):
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = D_render_rays(rays_flat[i:i + chunk], retraw=retraw, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k : torch.cat(all_ret[k], dim=0) for k in all_ret}
    return all_ret

# 渲染一张图片
def render( H, W, K, chunk=1024*32,
            rays=None, c2w=None, ndc=True,
            retraw=False,
            near=0., far=1.,
            use_viewdirs=True,
            c2w_staticCam=None,
            **kwargs):
    if c2w is not None:
        # 渲染整张图片所有的光线
        rays_o, rays_d = get_rays_torch(H, W, K, c2w)
    else:
        # 使用输入的光线
        rays_o, rays_d = rays

    if use_viewdirs:
        views = rays_d
        # 特殊情况(存疑)
        if c2w_staticCam is not None:
            rays_o, rays_d = get_rays_torch(H, W, K, c2w_staticCam)
        views = views / torch.norm(views, dim=-1, keepdim=True)
        views = torch.reshape(views, [-1, 3]).float()

    shape = rays_d.shape

    # llff
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # 构建ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)

    if use_viewdirs:
        rays = torch.cat([rays, views], -1)

    # 渲染 并 调整大小
    all_ret = batchify_rays(rays, chunk=chunk,**kwargs)
    for k in all_ret:
        k_shape = list(shape[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_shape)

    # 构建特殊的return数据结构
    k_extrct = ['rgb_map', 'disparity_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extrct]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extrct}
    return ret_list + [ret_dict]

def D_render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., frame_time=None,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays_torch(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays_torch(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    frame_time = frame_time * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far, frame_time], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = D_batchify_rays(rays, chunk=chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


# 渲染相机运动路径上的所有图片
def render_path(render_poses, hwf, K, chunk,
                render_dict,
                gt_imgs=None,
                savedir=None,
                render_factor=0):
    H, W, focal = hwf
    # 是否降低采样率以提速
    if render_factor != 0:
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []  # rgb pictures
    disps = []  # disparity maps

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, 'time: {:.2f}s'.format(time.time() - t))
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk, c2w=c2w[:3, :4], **render_dict)

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if i == 0:
            print('rgb_shape : ', rgb.shape, 'disp_shape : ', disp.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

def D_render_path(render_poses, render_times, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None,
                render_factor=0, save_also_gt=False, i_offset=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    if savedir is not None:
        save_dir_estim = os.path.join(savedir, "estim")
        save_dir_gt = os.path.join(savedir, "gt")
        if not os.path.exists(save_dir_estim):
            os.makedirs(save_dir_estim)
        if save_also_gt and not os.path.exists(save_dir_gt):
            os.makedirs(save_dir_gt)

    rgbs = []
    disps = []

    for i, (c2w, frame_time) in enumerate(zip(tqdm(render_poses), render_times)):
        rgb, disp, acc, _ = D_render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], frame_time=frame_time, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if savedir is not None:
            rgb8_estim = to8b(rgbs[-1])
            filename = os.path.join(save_dir_estim, '{:03d}.png'.format(i + i_offset))
            imageio.imwrite(filename, rgb8_estim)
            if save_also_gt:
                rgb8_gt = to8b(gt_imgs[i])
                filename = os.path.join(save_dir_gt, '{:03d}.png'.format(i + i_offset))
                imageio.imwrite(filename, rgb8_gt)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

# Create A NeRF
def create_NeRF(args):
    # Positional Encoding Functions
    type = args.nerf_type
    if type == "original":
        print("ORIGINAL")
        embed_fn, input_ch = get_embedder(L=args.L_positions, input_dims=args.input_dims)
        embed_fn_views, input_ch_views = get_embedder(L=args.L_views, input_dims=args.input_dims_view)

        output_ch = 5 if args.N_importance > 0 else 4
        skips = [4]

        # 初始化NeRF模型
        model = NeRF.get_by_name(type, D=args.netdepth,
                     W=args.netwidth,
                     input_ch=input_ch,
                     output_ch=output_ch,
                     skips=skips,
                     input_ch_views=input_ch_views,
                     use_viewdirs=args.use_viewdirs).to(device)
        grad_vars = list(model.parameters())

        # 初始化精细NeRF模型
        model_fine = None
        if args.N_importance > 0:
            model_fine = NeRF.get_by_name(type, D=args.netdepth_fine, W=args.netwidth_fine,
                              input_ch=input_ch, output_ch=output_ch,
                              skips=skips,
                              input_ch_views=input_ch_views,
                              use_viewdirs=args.use_viewdirs).to(device)
            grad_vars += list(model_fine.parameters())

        # 由network抽象定义的函数(小批量处理的版本)
        network_fn_chunky = lambda inputs, views, network_fn: run_network(inputs, views, network_fn,
                                                                          embed_pts_fn=embed_fn,
                                                                          embed_views_fn=embed_fn_views,
                                                                          netchunk=args.netchunk)

        # 初始化optimizer
        optimizer = torch.optim.AdamW(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        start = 0  # 全局训练次数标记
        basedir = args.basedir
        expname = args.expname
        # 加载checkpoints
        if args.ft_path is not None and args.ft_path != 'None':
            checkpoints = [args.ft_path]
        else:
            checkpoints = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname)))
                           if 'tar' in f]
        print('Found checkpoints', checkpoints)
        if len(checkpoints) > 0 and not args.no_reload:
            checkpoint_path = checkpoints[-1]
            print('Reloading from', checkpoint_path)
            checkpoint = torch.load(checkpoint_path)

            start = checkpoint['global_step']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.load_state_dict(checkpoint['network_fn_state_dict'])
            if model_fine is not None:
                model_fine.load_state_dict(checkpoint['network_fine_state_dict'])

        render_train = {
            'network_fn_chunky': network_fn_chunky,
            'network_fn': model,
            'network_fine': model_fine,
            'perturb': args.perturb,
            'raw_noise_std': args.raw_noise_std,
            'N_importance': args.N_importance,
            'N_samples': args.N_samples,
            'use_viewdirs': args.use_viewdirs,
            'white_bkgd': args.white_bkgd,
            'use_two_models_for_fine': args.use_two_models_for_fine,
        }

        # for llff forward-facing data ndc
        if args.dataset_type != 'llff' or args.no_ndc:
            print('NOT ndc')
            render_train['ndc'] = False
            render_train['lindisp'] = args.lindisp

        render_test = {k: render_train[k] for k in render_train}
        render_test['perturb'] = False
        render_test['raw_noise_std'] = 0.

        return render_train, render_test, start, grad_vars, optimizer

    elif type =="direct_temporal":
        embed_fn, input_ch = get_embedder(L=args.L_positions, input_dims=args.input_dims)
        embedtime_fn, input_ch_time = get_embedder(L=args.L_positions, input_dims=args.input_dims_view)

        input_ch_views = 0
        embeddirs_fn = None
        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(L=args.L_views, input_dims=args.input_dims)

        output_ch = 5 if args.N_importance > 0 else 4
        skips = [4]
        model = NeRF.get_by_name(args.nerf_type, D=args.netdepth, W=args.netwidth,
                                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                                 input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                                 use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                                 zero_canonical=not args.not_zero_canonical).to(device)
        grad_vars = list(model.parameters())

        model_fine = None
        if args.use_two_models_for_fine:
            model_fine = NeRF.get_by_name(args.nerf_type, D=args.netdepth_fine, W=args.netwidth_fine,
                                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                                          input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                                          use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                                          zero_canonical=not args.not_zero_canonical).to(device)
            grad_vars += list(model_fine.parameters())
        network_query_fn = lambda inputs, viewdirs, ts, network_fn: D_run_network(inputs, viewdirs, ts, network_fn,
                                                                                embed_fn=embed_fn,
                                                                                embeddirs_fn=embeddirs_fn,
                                                                                embedtime_fn=embedtime_fn,
                                                                                netchunk=args.netchunk,
                                                                                embd_time_discr=args.nerf_type != "temporal")

        # Create optimizer
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        if args.do_half_precision:
            print("Run model at half precision")
            if model_fine is not None:
                [model, model_fine], optimizers = amp.initialize([model, model_fine], optimizer, opt_level='O1')
            else:
                model, optimizers = amp.initialize(model, optimizer, opt_level='O1')

        start = 0
        basedir = args.basedir
        expname = args.expname
        ##########################
        # Load checkpoints
        if args.ft_path is not None and args.ft_path != 'None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                     'tar' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)

            start = ckpt['global_step']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Load model
            model.load_state_dict(ckpt['network_fn_state_dict'])
            if model_fine is not None:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            if args.do_half_precision:
                amp.load_state_dict(ckpt['amp'])

        ##########################

        render_kwargs_train = {
            'network_query_fn': network_query_fn,
            'perturb': args.perturb,
            'N_importance': args.N_importance,
            'network_fine': model_fine,
            'N_samples': args.N_samples,
            'network_fn': model,
            'use_viewdirs': args.use_viewdirs,
            'white_bkgd': args.white_bkgd,
            'raw_noise_std': args.raw_noise_std,
            'use_two_models_for_fine': args.use_two_models_for_fine,
        }

        # NDC only good for LLFF-style forward facing data
        if args.dataset_type != 'llff' or args.no_ndc:
            render_kwargs_train['ndc'] = False
            render_kwargs_train['lindisp'] = args.lindisp

        render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
        render_kwargs_test['perturb'] = False
        render_kwargs_test['raw_noise_std'] = 0.

        return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def train():
    parser = config_parser()
    args = parser.parse_args()
    is_straightforward = args.is_straightforward
    # 加载数据
    if args.nerf_type == "direct_temporal":
        images, poses, times, render_poses, render_times, hwf, i_split = D_load_blender_data(args.datadir,
                                                                                             args.half_res,
                                                                                             args.testskip)
        print('LOADED blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]
        min_time, max_time = times[i_train[0]], times[i_train[-1]]
        assert min_time == 0., "time must start at 0"
        assert max_time == 1., "max time must be 1"
    elif args.nerf_type == "original":
        if args.dataset_type == 'blender':
            images, poses, render_poses, hwf, i_split = load_blender_data(database_dir=args.datadir,
                                                                        half_res=args.half_res,
                                                                        testskip=args.testskip)

            print('LOADED blender', images.shape, render_poses.shape, hwf, args.datadir)
            i_train, i_val, i_test = i_split

            near = 2.
            far = 6.

            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3]

        elif args.dataset_type == 'llff':
            images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                    recenter=True, bd_factor=.75,
                                                                    spherify=args.spherify)
            hwf = poses[0,:3,-1]
            poses = poses[:,:3,:4]
            print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
            if not isinstance(i_test, list):
                i_test = [i_test]

            if args.llffhold > 0:
                print('Auto LLFF holdout,', args.llffhold)
                i_test = np.arange(images.shape[0])[::args.llffhold]

            i_val = i_test
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

            print('DEFINING BOUNDS')
            if args.no_ndc:
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.

            else:
                near = 0.
                far = 1.
            print('NEAR FAR', near, far)

    K = None  # 相机内参矩阵
    # 将相机内参调整为正确的数据类型
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal,     0, 0.5 * W],
            [    0, focal, 0.5 * H],
            [    0,     0,       1]
        ])
    # 获取render_poses 和 render_times
    if args.render_test:
        render_poses = np.array(poses[i_test])
        if args.nerf_type == "direct_temporal":
            render_times = np.array(times[i_test])

    basedir = args.basedir
    expname = args.expname
    print("basedir = ",basedir,'\n',"expname = ", expname)
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    # 初始化NeRF模型
    render_train, render_test, start, grad_vars, optimizer = create_NeRF(args)
    global_step = start
    
    # 边界 : 存储近平面和远平面
    boundarys_dict = {
        'near' : near,
        'far' : far
    }
    
    render_train.update(boundarys_dict)
    render_test.update(boundarys_dict)
    
    # 将需渲染的poses-data转移至GPU
    render_poses = torch.Tensor(render_poses).to(device)
    if args.nerf_type == "direct_temporal":
        render_times = torch.Tensor(render_times).to(device)

    # 若只需渲染，则直接渲染
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_poses 已被替换为 test poses
                images = images[i_test]
            else:
                images = None
            
            test_save_dir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(test_save_dir, exist_ok=True)
            print('test poses shape : ', render_poses.shape)
            if args.nerf_type == "original":
                rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_test, gt_imgs=images, savedir=test_save_dir, render_factor=args.render_factor)
            elif args.nerf_type == "direct_temporal":
                rgbs, _ = D_render_path(render_poses, render_times, hwf, K, chunk=args.chunk, render_kwargs=render_test, gt_imgs=images, savedir=test_save_dir,
                                      render_factor=args.render_factor)
            print('Done rendering', test_save_dir)
            imageio.mimwrite(os.path.join(test_save_dir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            
            return
    
    # 初始化raybatch tensor, 若需要batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        print('Batching Rays')
        # [N_poses, ro + rd, H, W, 3]
        rays = np.stack([get_rays_numpy(H, W, K, p) for p in poses[:, :3, :4]], axis=0)
        print('Done batching Rays, Concats')
        # [N_poses, ro + rd + rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], axis=1)
        # [N_poses, H, W, ro + rd + rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        # only train images
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], axis=0)
        # [N_train_poses * H * W, ro + rd + rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('Shuffle Rays')
        np.random.shuffle(rays_rgb)
        print('Done')
        i_batch = 0
    
    # 将训练所用的data转移至GPU
    # 若不使用batching, 则每次训练整张图片
    if args.no_batching:
        images = torch.Tensor(images).to(device)
    # 若使用batching, 则每次随机训练一组光线
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)
    if args.nerf_type == "direct_temporal":
        times = torch.Tensor(times).to(device)
    poses = torch.Tensor(poses).to(device)
    
    # 全局最大训练步数
    N_iters = args.N_iter + 1
    print('BEGIN TRAINING')
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()
        # 使用batching, 则随机抽样一组光线
        if use_batching:
            batch = rays_rgb[i_batch:i_batch + args.N_rand]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
        
            i_batch += args.N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle Batched Rays after an Epoch")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        # 不使用batching, 每次随机抽样一张图片
        else:
            if i >= args.precrop_iters_time:
                img_i = np.random.choice(i_train)
            else:
                skip_factor = i / float(args.precrop_iters_time) * len(i_train)
                max_sample = max(int(skip_factor), 3)
                img_i = np.random.choice(i_train[:max_sample])
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if args.nerf_type == "direct_temporal":
                frame_time = times[img_i]
            # 是否需要随机采样光线
            if N_rand is not None:
                rays_o, rays_d = get_rays_torch(H, W, K, torch.Tensor(pose))

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                        ), dim=-1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(0, H - 1, H),
                            torch.linspace(0, W - 1, W)
                        ), dim=-1)
                    
                coords = torch.reshape(coords, [-1, 2]) # [H * W, 2]
                select_indices = np.random.choice(coords.shape[0], size=[args.N_rand], replace=False) # [N_rand, ]
                selected_coords = coords[select_indices].long() # [N_rand, 2]
                rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]] # [N_rand, 3]
                rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]] # [N_rand, 3]
                batch_rays = torch.stack([rays_o, rays_d], dim=0)
                target_s = target[selected_coords[:, 0], selected_coords[:, 1]] # [N_rand, 3]
                
        # 优化(optimization)循环
        if args.nerf_type == "original":
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays, retraw=True, **render_train)
        elif args.nerf_type == "direct_temporal":
            rgb, disp, acc, extras = D_render(H, W, K, chunk=args.chunk, rays=batch_rays, frame_time=frame_time,
                                            verbose=i < 10, retraw=True,
                                            **render_train)
        # 如果启用了tv_loss(总变差）则计算下一帧
        if args.add_tv_loss and args.nerf_type == "direct_temporal":
            frame_time_prev = times[img_i - 1] if img_i > 0 else None
            frame_time_next = times[img_i + 1] if img_i < times.shape[0] - 1 else None

            if frame_time_prev is not None and frame_time_next is not None:
                if np.random.rand() > .5:
                    frame_time_prev = None
                else:
                    frame_time_next = None

            if frame_time_prev is not None:
                rand_time_prev = frame_time_prev + (frame_time - frame_time_prev) * torch.rand(1)[0]
                _, _, _, extras_prev = D_render(H, W, K, chunk=args.chunk, rays=batch_rays, frame_time=rand_time_prev,
                                                verbose=i < 10, retraw=True, z_vals=extras['z_vals'].detach(),
                                                **render_train)

            if frame_time_next is not None:
                rand_time_next = frame_time + (frame_time_next - frame_time) * torch.rand(1)[0]
                _, _, _, extras_next = D_render(H, W, K, chunk=args.chunk, rays=batch_rays, frame_time=rand_time_next,
                                                verbose=i < 10, retraw=True, z_vals=extras['z_vals'].detach(),
                                                **render_train)

        optimizer.zero_grad()
        img_loss = mse(rgb, target_s)

        tv_loss = 0
        # 计算tv_loss
        if args.add_tv_loss:
            if frame_time_prev is not None:
                tv_loss += ((extras['position_delta'] - extras_prev['position_delta']).pow(2)).sum()
                if 'position_delta_0' in extras:
                    tv_loss += ((extras['position_delta_0'] - extras_prev['position_delta_0']).pow(2)).sum()
            if frame_time_next is not None:
                tv_loss += ((extras['position_delta'] - extras_next['position_delta']).pow(2)).sum()
                if 'position_delta_0' in extras:
                    tv_loss += ((extras['position_delta_0'] - extras_next['position_delta_0']).pow(2)).sum()
            tv_loss = tv_loss * args.tv_loss_weight

        loss = img_loss + tv_loss
        PSNR = mse2psnr(img_loss)
        
        if 'rgb0' in extras:
            img_loss0 = mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            PSNR0 = mse2psnr(img_loss0)

        if args.do_half_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        
        # 更新学习率
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        
        dt = time.time() - time0
        
        # ABOUT Logging
        # # Save Model State Data
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if render_train['network_fine'] is not None:
                save_dict['network_fine_state_dict'] = render_train['network_fine'].state_dict()
            if args.do_half_precision:
                save_dict['amp'] = amp.state_dict()
            torch.save(save_dict, path)
            print('Saved checkpoints at', path)
        # 打印训练信息
        if i % args.i_print == 0:
            tqdm_txt = f"[TRAIN] Iter: {i} Loss_fine: {img_loss.item()} PSNR: {PSNR.item()}"
            if args.add_tv_loss:
                tqdm_txt += f" TV: {tv_loss.item()}"
            tqdm.write(tqdm_txt)

            writer.add_scalar('loss', img_loss.item(), i)
            writer.add_scalar('psnr', PSNR.item(), i)
            if 'rgb0' in extras:
                writer.add_scalar('loss0', img_loss0.item(), i)
                writer.add_scalar('psnr0', PSNR0.item(), i)
            if args.add_tv_loss:
                writer.add_scalar('tv', tv_loss.item(), i)

        del loss, img_loss, PSNR, target_s
        if 'rgb0' in extras:
            del img_loss0, PSNR0
        if args.add_tv_loss:
            del tv_loss
        del rgb, disp, acc, extras

        # 创建图片计算PSNR
        if i%args.i_img==0:
            torch.cuda.empty_cache()
            # Log a rendered validation view to Tensorboard
            img_i=np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            if args.nerf_type == "direct_temporal":
                frame_time = times[img_i]
            with torch.no_grad():
                if args.nerf_type == "direct_temporal":
                    rgb, disp, acc, extras = D_render(H, W, K, chunk=args.chunk, c2w=pose, frame_time=frame_time,
                                                    **render_test)
                elif args.nerf_type == "original":
                    rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, c2w=pose, **render_test)
            PSNR = mse2psnr(mse(rgb, target))
            writer.add_image('gt', to8b(target.cpu().numpy()), i, dataformats='HWC')
            writer.add_image('rgb', to8b(rgb.cpu().numpy()), i, dataformats='HWC')
            writer.add_image('disp', disp.cpu().numpy(), i, dataformats='HW')
            writer.add_image('acc', acc.cpu().numpy(), i, dataformats='HW')

            if 'rgb0' in extras:
                writer.add_image('rgb_rough', to8b(extras['rgb0'].cpu().numpy()), i, dataformats='HWC')
            if 'disp0' in extras:
                writer.add_image('disp_rough', extras['disp0'].cpu().numpy(), i, dataformats='HW')
            if 'z_std' in extras:
                writer.add_image('acc_rough', extras['z_std'].cpu().numpy(), i, dataformats='HW')

            print("finish summary")
            writer.flush()
            
        # # Generate 360° Video
        if i % args.i_video == 0 and i > 0:
            with torch.no_grad():
                savedir = os.path.join(basedir, expname, 'frames_{}_spiral_{:06d}_time/'.format(expname, i))
                if savedir is not None:  # 确保目录存在
                    os.makedirs(savedir, exist_ok=True)  # exist_ok=True 允许即使目录已存在也不抛出错误
                if args.nerf_type == "direct_temporal":
                    rgbs, disps = D_render_path(render_poses, render_times, hwf, K,chunk=args.chunk, render_kwargs=render_test, save_also_gt=False,
                                                savedir=savedir)
                elif args.nerf_type == "original":
                    rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_test,
                                                savedir=savedir)
                    print("video generated")
            print('Done Generate Video Frames', rgbs.shape, disps.shape)
            videobase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(videobase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(videobase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            print('Done Saving Video')
        # # print loss psnr iter LOG
        if i % args.i_testset == 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            if testsavedir is not None:  # 确保目录存在
                os.makedirs(testsavedir, exist_ok=True)  # exist_ok=True 允许即使目录已存在也不抛出错误
            print('Testing poses shape...', poses[i_test].shape)
            with torch.no_grad():
                if args.nerf_type == "direct_temporal":
                    D_render_path(torch.Tensor(poses[i_test]).to(device), torch.Tensor(times[i_test]).to(device),
                            hwf, K, args.chunk, render_test, gt_imgs=images[i_test], savedir=testsavedir)
                elif args.nerf_type == "original":
                    render_path(torch.Tensor(poses[i_test]).to(device),
                                hwf, K, args.chunk, render_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')
        global_step += 1

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()