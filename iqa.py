import os
os.environ['HF_HUB_CACHE'] = '/home/gujh/sdsr/.cache/huggingface/hub'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import pyiqa
import torch
import time
# list all available metrics
import re
import time
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create metric with default setting
# name_list = ['lpips', 'dists', 'NIQE', 'MANIQA-pipal', 'MUSIQ', 'CLIPIQA']
name_list = ['NIQE', 'MANIQA-pipal', 'MUSIQ', 'CLIPIQA']
# name_list = ['PSNR', 'SSIM', 'LPIPS', 'DISTS']

name_list = [n.lower() for n in name_list]

metric_list = []
for name in name_list:
    if name == 'psnr' or name == 'ssim' or name == 'niqe':
        iqa_metric = pyiqa.create_metric(name, device=device, crop_border=4)
    else:
        iqa_metric = pyiqa.create_metric(name, device=device)
    metric_list.append(iqa_metric)

    # check if lower better or higher better
    print(name + ' lower_better: ' + str(iqa_metric.lower_better))


def rename_files(filename):
    pattern = r"_(\d)\.png$"
    if re.search(pattern, filename):
        filename = re.sub(pattern, r".png", filename)
    return filename

def rename_dirs(dirname):
    pattern = r"/\_(\d)/$"
    if re.search(pattern, dirname):
        dirname = re.sub(pattern, r"/", dirname)
    return dirname

def iqa(img_dirs, ref_dir):
    score_list = []
    for i, d in enumerate(img_dirs):
        if '.png' not in os.listdir(d)[0]:
            raise NotImplementedError
        else:
            score = iqa_forward(d, ref_dir)

        print(f'{d} finished!')
        score_list.append(score)

    score_list = np.stack(score_list)

    print(score_list)
    with open('result_pyiqa.txt', 'a') as f:
        date_str = time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime())
        lines = [date_str]
        for i, metric_name in enumerate(name_list):
            for j, dir_name in enumerate(img_dirs):
                info = f'{dir_name}: '
                info += f'{metric_name}: {score_list[j, i]:.4f}'
                print(info)
                lines.append(info)
        
        lines.append('\n')
        f.writelines('\n'.join(lines))

def iqa_forward(img_dir, ref_dir):

    files = os.listdir(img_dir)
    score_list = np.zeros((len(metric_list), len(files)))

    for j, iqa_metric in enumerate(metric_list):
        # if 'DSC_1603_x1.png' in img_path:
        for i, f in enumerate(files):
            img = Image.open(os.path.join(img_dir, f))
            if img.size[0] < 512:
                img = img.resize((512, 512))
            score_list[j, i] = iqa_metric(img, os.path.join(ref_dir, f))
    scores = np.mean(score_list, axis=1)

    return scores

        
if __name__ == "__main__":

    img_dirs = [
        # '/home/gujh/sdsr/RealSR/test_HR', 
        '/home/gujh/sdsr/RealSR/test_LR', 
    ]
    # img_dirs = 'results/DrealSR_cfg/uh_lq_cpip1_ImageNet_150k_wocfg4.0'
    # img_dirs = 'results/RealSR_cfg'
    # img_dirs = 'results/DIV2K_cfg'
    if not isinstance(img_dirs, list):
        img_dirs = [os.path.join(img_dirs, i) for i in os.listdir(img_dirs)]

    img_dirs = sorted(img_dirs)

    # ref_dir = None
    # ref_dir = 'StableSR_testsets/RealSRVal_crop128/test_HR'
    ref_dir = '/home/gujh/sdsr/RealSR/test_HR'
    # ref_dir = 'StableSR_testsets/DIV2K_V2_val/gt'

    iqa(img_dirs, ref_dir)
