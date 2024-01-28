import torch
import numpy as np

def Hierarchical_Sampling(bins, weights, N_samples, det=False, pytest=False):
    # bins : [N_rays, N_samples + 1] 采样区间的边界
    # weights : [N_rays, N_samples] 每个采样区间的权重值
    # N_importance : 采样数量
    # det : 是否进行确定性采样
    
    # PDF-概率密度
    # CDF-累计分布(即概率密度PDF的积分)

    # 避免NAN
    weights = weights + 1e-5

    # 首先利用coarse网络中的weights归一化计算出PDF
    PDF = weights / torch.sum(weights, dim=-1, keepdim=True)
    
    # 对PDF进行概率连加, 得到累计分布CDF
    CDF = torch.cumsum(PDF, dim=-1)
    # CDF shape : [N_rays, len(bins)]
    CDF = torch.cat([torch.zeros_like(CDF[..., :1]), CDF], dim=-1)
    
    # 生成均匀分布的随机数u, 便于后续对每条光线进行重要性采样
    # 根据det决定进行确定性采样或随机性采样
    if det:
        # Uniform samples
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(CDF.shape[:-1]) + [N_samples])
    else:
        # Random samples
        u = torch.rand(list(CDF.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(CDF.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # 比较u和CDF, 确定生成的随机数u在累积分布函数CDF中的位置
    u = u.contiguous()
    inds = torch.searchsorted(CDF, u, right=True)
    # 确定采样点所在的区间，根据上述得到的索引，计算得到采样区间的下界和上界，便于后续线性插值
    lower = torch.max(torch.zeros_like(inds - 1), inds - 1)
    upper = torch.min((CDF.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([lower, upper], dim=-1) # (N_rays, N_samples, 2)
    
    matched_shape = [inds_g.shape[0], inds_g.shape[1], CDF.shape[-1]]

    # 根据计算出的采样区间上界索引和下界索引, 对CDF和bins进行重写
    CDF_g = torch.gather(CDF.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    
    denom = (CDF_g[..., 1] - CDF_g[..., 0])
    
    # 若denom为0, 表示某个区间的权重为0，故可将其设为任何值
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - CDF_g[..., 0]) / denom
    
    # 线性插值采样
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
    return samples