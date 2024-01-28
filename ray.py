import torch
import numpy as np

def get_rays_torch(H, W, K, c2w):
    # torch.meshgrid输出以第一个参数为列的矩阵和以第二个参数为行的矩阵，恰恰与np.meshgrid相反
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    # 下两行代码即是为了与np.meshgrid统一
    i = i.t()
    j = j.t()
    # 下方即应用公式计算 : d = ((i - cx) / f, (j - cy) / f, 1)
    # OpenCV / Colmap 的相机坐标系中相机的Y-UP朝下，相机光心朝向+Z轴
    # NeRF   / OpenGL 的相机坐标系中相机的Y-UP朝上，相机光心朝向-Z轴
    # 故Y轴项，Z轴项相对公式需要多乘一个负号
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], dim=-1)
    # 将ray directions由相机坐标系转化为世界坐标系
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], dim=-1)
    # 将ray origin由相机坐标系转化为世界坐标系，所有光线起点为同一点
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_numpy(H, W, K, c2w):
    # 仿照上述get_rays_torch, 写出numpy版本的函数代码即可，需要注意torch与numpy的细节不同
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], axis=-1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return  rays_o, rays_d

# for llff forward-facing data
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d