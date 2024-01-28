import os
import imageio

# llff数据集根地址
database_dir = './data/nerf_llff_data/fern'

img0 = [os.path.join(database_dir, 'images', f) for f in sorted(os.listdir(os.path.join(database_dir, 'images'))) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
# 原图的shape大小
shape = imageio.imread(img0).shape

# 计算所用的数据
factor = 16
height = None
width  = None

sfx = ''
if factor is not None:
    percent = 100. / factor
    sfx = '{}'.format(percent)
elif height is not None:
    factor = shape[0] / float(height)
    width = int(shape[1] / factor)
    sfx = '{}x{}'.format(width, height)
elif width is not None:
    factor = shape[1] / float(width)
    height = int(shape[0] / factor)
    sfx = '{}x{}'.format(width, height)

print(sfx)