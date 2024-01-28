import os
import cv2
import numpy as np
import imageio
import json
import torch

# 沿z轴平移t-平移矩阵
trans_t = lambda t : torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]
]).float()

# 绕x轴旋转phi-旋转矩阵
rot_phi = lambda phi : torch.Tensor([
    [1,           0,            0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi),  np.cos(phi), 0],
    [0,           0,            0, 1]
]).float()

# 绕y轴旋转theta-旋转矩阵
rot_theta = lambda theta : torch.Tensor([
    [np.cos(theta), 0, -np.sin(theta), 0],
    [            0, 1,              0, 0],
    [np.sin(theta), 0,  np.cos(theta), 0],
    [            0, 0,              0, 1] 
]).float()

def pose_spherical(theta, phi, t):
    # 函数返回三个参数决定的世界坐标系下camera的pose矩阵
    # 函数参数含义同上述三个lambda函数
    # c2w : camera coordinates TO world coordinates
    c2w = trans_t(t)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w

def load_blender_data(database_dir, half_res=False, testskip=1):
    # database_dir : 数据库的根地址，例如data_lego的根地址为名为lego的文件夹
    # half_res :
    # testskip : 取数据时的切片步长，取test集数据时，隔testskip个data取一个data
    
    # branches : (List)数据库的分支名，如下述train集，test集，val集
    branches = ['train', 'val', 'test']
    
    # data_s : (Dict)保存数据的Dict，key值为branches的元素
    data_s = {}
    
    # 读取json文件，data存入data_s中
    for branch in branches:
        with open(os.path.join(database_dir, 'transforms_{}.json'.format(branch)), mode='r') as file:
            data_s[branch] = json.load(file)
    
    all_imgs = []
    all_poses = []
    counts = [0]
    
    for branch in branches:
        
        data = data_s[branch]
        imgs = []
        poses = []
        
        # 按照testskip参数，设置切片步长skip
        if branch == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip
        
        # 按照json文件中的路径读取图片 并 读取json文件中的相机矩阵
        for frame in data['frames'][::skip]:
            img_file_name = os.path.join(database_dir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(img_file_name))
            poses.append(np.array(frame['transform_matrix']))
        
        # 保存RGBA四个通道的值，并归一化，统一数据类型
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        
        # 相机矩阵统一数据类型
        poses = np.array(poses).astype(np.float32)
        
        # counts[0] : 0
        # counts[1] : num_train_imgs
        # counts[2] : num_train_imgs + num_val_imgs
        # counts[3] : num_train_imgs + num_val_imgs + num_test_imgs
        counts.append(counts[-1] + imgs.shape[0])
        
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    # i_split[0] : indices for train_imgs in imgs(last)
    # i_split[1] : indices for val_imgs   in imgs(last)
    # i_split[2] : indices for test_imgs  in imgs(last)
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    # 拼接imgs和poses
    # imgs的shape为 (n_train_imgs + n_val_imgs + n_test_imgs) * H * W * 4
    imgs = np.concatenate(all_imgs, axis=0)
    poses = np.concatenate(all_poses, axis=0)
    
    H, W = imgs[0].shape[:2]
    
    # camera_angle_x : Field of View in x dimension
    # 计算焦距的公式为 focal = (W / 2) / tan(FoV / 2)
    camera_angle_x = float(data['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    # 生成环绕模型360度的相机pose矩阵torch列表
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], dim=0)
    
    # 图像数据是否缩小为原来的一半
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    
    # imgs : image数据，包含train_imgs, val_imgs, test_imgs
    # poses : 与上述image数据大小相同，一一对应
    # render_poses: 渲染所用的poses，在三维中连续(即相机录像是一个环)
    # [H, W, focal] : 图像与相机焦距参数
    # i_split: 
    # # i_split[0] : indices for train_imgs in imgs
    # # i_split[1] : indices for val_imgs   in imgs
    # # i_split[2] : indices for test_imgs  in imgs
    return imgs, poses, render_poses, [H, W, focal], i_split


def D_load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_times = []
    counts = [0]
    for s in splits:
        meta = metas[s]

        imgs = []
        poses = []
        times = []
        # if s=='train' or testskip==0:
        #     skip = 2  # if you remove/change this 2, also change the /2 in the times vector
        # else:
        skip = testskip

        for t, frame in enumerate(meta['frames'][::skip]):
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            cur_time = frame['time'] if 'time' in frame else float(t) / (len(meta['frames'][::skip]) - 1)
            times.append(cur_time)

        assert times[0] == 0, "Time must start at 0"

        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        times = np.array(times).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_times.append(times)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    times = np.concatenate(all_times, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format('render'))):
        with open(os.path.join(basedir, 'transforms_{}.json'.format('render')), 'r') as fp:
            meta = json.load(fp)
        render_poses = []
        for frame in meta['frames']:
            render_poses.append(np.array(frame['transform_matrix']))
        render_poses = np.array(render_poses).astype(np.float32)
    else:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]],
                                   0)
    render_times = torch.linspace(0., 1., render_poses.shape[0])

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, times, render_poses, render_times, [H, W, focal], i_split