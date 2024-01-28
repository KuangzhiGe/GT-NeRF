import os
import imageio
import numpy as np


# A LLFF Dataset Should Be Like
'''
Database_root
    images
    poses_bounds.npy
'''

# 可能需要先手动执行的一些命令

# 下述命令实现了原NeRF实现代码中_minify()函数的功能
# 若需要N倍缩放图片(原图大小的(100/N)%), 或设置宽高比为[W * H]
# 首先，在数据根目录下新建文件夹名为'images_N'或'images_WxH'
# 在数据根目录下运行命令：
# # copy -Path ./images/* -Destination ./images_N(or ./images_WxH)
# 在新文件夹'images_N'或'images_WxH'下执行重设图像大小的命令与删除原图的命令：
# # magick mogrify -resize (100/N)%(or WxH) -format png *.JPG(or *.png, *.jpg...根据数据原图的格式来)
# # rm ./*.JPG(or .png根据原图的格式来)

def load_data(database_dir, factor=None, width=None, height=None, load_imgs=True):
    
    # poses_bounds.npy : 内含一个shape为[N, 17]的数组
    # 前15维是3x5的poses矩阵，最后2维是光线的near边界与far边界
    
    poses_arr = np.load(os.path.join(database_dir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0]) # [3, 5, N]
    bds = poses_arr[:, -2:].transpose([1, 0])
    
    # 获取原图数据的宽高比shape
    img0 = [os.path.join(database_dir, 'images', f) for f in sorted(os.listdir(os.path.join(database_dir, 'images'))) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    shape = imageio.imread(img0).shape
    
    # 由输入的factor计算得到应使用的加工过的数据集
    sfx = ''
    if factor is not None:
        sfx = '_{}'.format(factor)
    elif height is not None:
        factor = shape[0] / float(height)
        width = int(shape[1] / factor)
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = shape[1] / float(width)
        height = int(shape[0] / factor)
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    # 文件夹不存在-报错
    imgdir = os.path.join(database_dir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'Does Not Exist, Return')
        return
    # pose矩阵数量与图片数量不匹配-报错
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} & poses {}'.format(len(imgfiles), poses.shape[-1]))
    
    # 更新为实际使用的数据集图片的shape
    shape = imageio.imread(imgfiles[0]).shape
    # 修改pose矩阵中的图像宽高
    poses[:2, 4, :] = np.array(shape[:2]).reshape([2, 1])
    # 修改pose矩阵中的焦距
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f)
        else:
            return imageio.imread(f)
    
    imgs = [imread(f)[...,:3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, axis=-1)
    
    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs

# 单位化函数
def normalize(x):
    return x / np.linalg.norm(x)

# 构造相机矩阵地函数
def view_matrix(z, up, pos):
    # z : z轴平均单位方向向量
    # up : y轴主方向向量
    # pos : 平均平移矩阵, 即平均相机位置
    vec2 = normalize(z)
    vec1_avg = up
    # 计算出x轴的单位方向向量
    vec0 = normalize(np.cross(vec1_avg, vec2))
    # 计算出y轴的单位方向向量
    vec1 = normalize(np.cross(vec2, vec0))
    # 拼接构成相机矩阵
    m = np.stack([vec0, vec1, vec2, pos], 1) # [3, 4]
    return m

# 计算所有相机位姿的均值, 计算包括位置和朝向
def poses_avg(poses):
    # 获取图像的宽高与焦距
    hwf = poses[0, :3, -1:] # [3, 1]
    # 计算所有平移矩阵t的均值
    center = poses[:, :3, 3].mean(0) # [3, 1]
    # 全部旋转矩阵R的第三列(z轴)求和并归一化
    # 求出的vec2为z轴的平均单位向量
    vec2 = normalize(poses[:, :3, 2].sum(0)) # [3, 1]
    # 全部旋转矩阵R的第二列(y轴)求和
    # 求出的up为y轴的主方向向量
    up = poses[:, :3, 1].sum(0) # [3, 1]
    # 平均相机矩阵
    c2w = np.concatenate([view_matrix(vec2, up, center), hwf], 1) # [3, 5]
    return c2w

# 中心化相机位姿, 包括位置与朝向
# # 相机的平均位姿的逆左乘所有相机位姿, 从而完成中心化
def recenter_poses(poses):
    tmp_poses = poses + 0
    
    # 平均相机位姿: [3, 4]->[4, 4]
    # bottom : [1, 4], [c2w + bottom]=[4, 4]
    bottom = np.reshape([0, 0, 0, 1.], [1, 4]) # [1, 4]
    c2w = poses_avg(poses) # [3, 4]
    c2w = np.concatenate([c2w[:3, :4], bottom], -2) # [4, 4]
    
    # 所有相机位姿: [3, 4]->[4, 4]
    # bottom : [1, 4]->[1, 1, 4]->[N, 1, 4]
    # np.tile() : 对bottom对应维度复制拓展[1, 1, 4]*[N, 1, 1]->[N, 1, 4]
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1]) # [N, 1, 4]
    # poses shape : [N, 3, 4]->[N, 4, 4]
    poses = np.concatenate([poses[:, :3, :4], bottom], -2) # [N, 4, 4]
    
    # c2w的逆左乘poses, 完成中心化(归一化)
    poses = np.linalg.inv(c2w) @ poses # [N, 4, 4]
    
    # 更新位姿矩阵
    tmp_poses[:, :3, :4] = poses[:, :3, :4] # [N, 4, 4]
    poses = tmp_poses
    return poses

def spherify_poses(poses, bds):
    # 位姿矩阵[3, 4]->[4, 4]的函数
    pose_3x4_to_4x4 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)
    
    # 位姿矩阵中旋转矩阵的第三列(z轴)-方向向量
    rays_d = poses[:, :3, 2:3]
    # 位姿矩阵中的平移矩阵t-相机光心
    rays_o = poses[:, :3, 3:4]
    
    # 计算出与空间中所有相机中心射线的距离之和最小的点
    def min_line_dist(rays_o, rays_d):
        # [N, 3, 3]
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        # [N, 3, 1]
        b_i = -A_i @ rays_o
        # [3]
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist
    
    # 并未找到如此计算的原理解释, 可将其理解为
    # # 把空间中与相机中心射线的距离之和最小的点视为场景的中心点
    pt_mindist = min_line_dist(rays_o, rays_d)
    center = pt_mindist
    
    # 下述过程是在中心化相机位姿, 与recenter_poses()函数的方法基本一致
    # 该过程使得所有相机的中心射线都将经过上述的center点
    ################################################
    # 所有 相机光心到场景中心点的方向向量 的平均距离向量
    up = (poses[:, :3, 3] - center).mean(0) # [3]
    # 归一化为平均单位向量
    vec0 = normalize(up) # [3]
    # 计算得到两两垂直的单位方向向量
    vec1 = normalize(np.cross([.1, .2, .3], vec0)) # [3]
    vec2 = normalize(np.cross(vec0, vec1)) # [3]
    pos = center
    # 构建坐标系
    c2w = np.stack([vec1, vec2, vec0, pos], 1) # [3, 4]
    # c2w的逆左乘poses，完成所有相机位姿的归一化
    poses_reset = np.linalg.inv(pose_3x4_to_4x4(c2w[None])) @ pose_3x4_to_4x4(poses[:, :3, :4])
    ################################################
    
    # 归一化后的所有光心的平均
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
    # 计算缩放因子
    sc = 1. / rad
    # 缩放光心
    poses_reset[:, :3, 3] *= sc
    # 缩放边界
    bds *= sc
    # 归一化
    rad *= sc
    
    # 下述过程用于生成环绕球面一圈的新视角
    # 相机始终在场景中物体的正面
    # 新视角的相机光心位置camera_origin:
    # # 其z的大小是固定的zh(物体前方一段距离);
    # # x, y在半径为rad_circle的圆上.
    #####################################
    # 平均光心位置
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    # 平均光心z轴的距离
    zh = centroid[2]
    # 运动圆半径
    rad_circle = np.sqrt(rad**2 - zh**2)
    # 初始化新视角
    new_poses = []
    # 在运动圆上均匀采样
    for th in np.linspace(0, 2. * np.pi, 120):
        camera_origin = np.array([rad_circle * np.cos(th), rad_circle * np.sin(th), zh])
        up = np.array([0, 0, -1.])
        vec2 = normalize(camera_origin)
        # 构建坐标系
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camera_origin
        p = np.stack([vec0, vec1, vec2, pos], 1) # [3, 4]
        new_poses.append(p)
    # 将新视角拼接在一起
    new_poses = np.stack(new_poses, 0) # [N, 3, 4]
    #################################################
    # 新位姿拼接上原始位姿的hwf
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    # 更新(归一化)的新位姿拼接上原始位姿的hwf
    poses_reset = np.concatenate([poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)
    
    return poses_reset, new_poses, bds    

# 生成用于渲染的螺旋路径的位姿(List)
def render_path_spiral(c2w, up, rads, focal, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([view_matrix(z, up, c), hwf], 1))
    return render_poses

def load_llff_data(database_dir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    # poses : [3, 5, N]
    # # N-数据集的图片个数
    # # 3x3(前3列)旋转矩阵
    # # 3x1(第4列)平移矩阵
    # # 3x1(第5列)[H, W, f]
    # bds : [2, N] 采样的far, near边界(深度值范围)
    # imgs : [H, W, C, N]
    poses, bds, imgs = load_data(database_dir, factor=factor)
    print('Loaded', database_dir, bds.min(), bds.max())
    
    # 列变换与坐标轴更新，COLMAP的坐标轴 与 NeRF(OpenGL)的坐标轴差异
    # # [x, y, z]->[y, -x, z]
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], axis=1)
    
    # 维度调换, 第-1轴(最后轴)到第0轴
    # [N, 3, 5]
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    # [N, H, W, C]
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    # [N, 2]
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # 计算缩放因子, 以bds.min为基准, 与归一化操作类似(?)
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    # 缩放位姿矩阵的平移矩阵
    poses[:, :3, 3] *= sc
    # 缩放边界
    bds *= sc
    
    # 中心化(归一化所有相机位姿矩阵)
    if recenter:
        poses = recenter_poses(poses)
    
    # 将相机分布限制于固定球体内, 并返回一个环绕物体的相机位姿轨迹用于新视角的合成
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)
    
    else:
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3, :4])
        
        # 旋转矩阵R的第2轴(y轴)平均单位向量
        up = normalize(poses[:, :3, 1].sum(0))
        
        # 选择一个"合理"的"焦距深度"
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / ((1. - dt) / close_depth + dt / inf_depth)
        # 计算新的focal
        focal = mean_dz
        
        # 下方代码存疑
        ############################################
        zdelta = close_depth * .2
        tt = poses[:,:3,3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2
        ############################################
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zrate=.5, rots=N_rots, N=N_views)
    
    render_poses = np.array(render_poses).astype(np.float32)
    
    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test