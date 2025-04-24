from typing import Dict, Sequence
import math
import random
import time

import numpy as np
import torch
from torch.utils import data
from PIL import Image
import cv2

from utils.degradation import circular_lowpass_kernel, random_mixed_kernels
from utils.image import augment, center_crop_arr
from utils.file import load_file_list
import torch.nn.functional as F
import pyiqa
import torchvision.transforms.functional as TF

def random_crop_arr(pil_image, image_size):
    h, w = pil_image.size
    
    # if min(h, w) >= image_size * 2:
    if h == 1024 and w == 1024:
        scale = random.random() + 1  # uniform from 1 to 2
        image_size_crop = int(image_size * scale)
        crop_x = random.randrange(h - image_size_crop + 1)
        crop_y = random.randrange(w - image_size_crop + 1)
        pil_image = pil_image.crop((crop_x, crop_y, crop_x + image_size_crop, crop_y + image_size_crop))
        pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)
        return np.array(pil_image)
    
    else:
        arr = np.array(pil_image)
        if min(h, w) < image_size:
            pad_h = max(0, image_size - h)
            pad_w = max(0, image_size - w)
            arr = cv2.copyMakeBorder(arr, 0, pad_w, 0, pad_h, cv2.BORDER_REFLECT_101)

        crop_y = random.randrange(arr.shape[0] - image_size + 1)
        crop_x = random.randrange(arr.shape[1] - image_size + 1)
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
        

def random_resize_img(pil_image, image_size=1024):
    h, w = pil_image.size
    if min(h, w) > image_size * 1.5:
        short_s = random.randint(image_size, image_size * 1.5)
        h, w = (short_s, int(w/h*short_s)) if h < w else (int(h/w*short_s), short_s)
        pil_image = pil_image.resize((h, w), resample=Image.BILINEAR)

    return pil_image


class RealESRGANDataset(data.Dataset):
    """
    # TODO: add comment
    """

    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        use_rot: bool,
        # blur kernel settings of the first degradation stage
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        betag_range: Sequence[float],
        betap_range: Sequence[float],
        sinc_prob: float,
        # blur kernel settings of the second degradation stage
        blur_kernel_size2: int,
        kernel_list2: Sequence[str],
        kernel_prob2: Sequence[float],
        blur_sigma2: Sequence[float],
        betag_range2: Sequence[float],
        betap_range2: Sequence[float],
        sinc_prob2: float,
        final_sinc_prob: float
    ) -> "RealESRGANDataset":
        super(RealESRGANDataset, self).__init__()
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["center", "random", "none"], f"invalid crop type: {self.crop_type}"

        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        # a list for each kernel probability
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        # betag used in generalized Gaussian blur kernels
        self.betag_range = betag_range
        # betap used in plateau blur kernels
        self.betap_range = betap_range
        # the probability for sinc filters
        self.sinc_prob = sinc_prob

        self.blur_kernel_size2 = blur_kernel_size2
        self.kernel_list2 = kernel_list2
        self.kernel_prob2 = kernel_prob2
        self.blur_sigma2 = blur_sigma2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2
        self.sinc_prob2 = sinc_prob2
        
        # a final sinc filter
        self.final_sinc_prob = final_sinc_prob
        
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        
        # kernel size ranges from 7 to 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        # TODO: kernel range is now hard-coded, should be in the configure file
        # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

    @torch.no_grad()
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # -------------------------------- Load hq images -------------------------------- #

        hq_path = self.paths[index]
        success = False
        for _ in range(3):
            try:
                pil_img = Image.open(hq_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {hq_path}"
        
        # pil_img = random_resize_img(pil_img, self.out_size * 2)
        if self.crop_type == "random":
            pil_img = random_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "center":
            pil_img = center_crop_arr(pil_img, self.out_size)
        # self.crop_type is "none"
        else:
            pil_img = np.array(pil_img)
            assert pil_img.shape[:2] == (self.out_size, self.out_size)
        # hwc, rgb to bgr, [0, 255] to [0, 1], float32
            
        img_hq = (pil_img[..., ::-1] / 255.0).astype(np.float32)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_hq = augment(img_hq, self.use_hflip, self.use_rot)
        
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # [0, 1], BGR to RGB, HWC to CHW
        img_hq = torch.from_numpy(
            img_hq[..., ::-1].transpose(2, 0, 1).copy()
        ).float()
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        p_p = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
        n_p = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'

        return {
            "hq": img_hq, "kernel1": kernel, "kernel2": kernel2,
            "sinc_kernel": sinc_kernel, "txt": ''
        }
    
        if random.random() > 0.2:
            txt = p_p
        else:
            txt = n_p
            downsampled = F.interpolate(img_hq.unsqueeze(0), scale_factor=0.25, mode='bilinear')
            img_hq = F.interpolate(downsampled, scale_factor=4, mode='bilinear').squeeze()
            
        return {
            "hq": img_hq, "kernel1": kernel, "kernel2": kernel2,
            "sinc_kernel": sinc_kernel, "txt": txt
        }

    def __len__(self) -> int:
        return len(self.paths)


class RealESRGANFDataset(data.Dataset):
    """
    # TODO: add comment
    """

    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        use_rot: bool,
        # blur kernel settings of the first degradation stage
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        betag_range: Sequence[float],
        betap_range: Sequence[float],
        sinc_prob: float,
        # blur kernel settings of the second degradation stage
        blur_kernel_size2: int,
        kernel_list2: Sequence[str],
        kernel_prob2: Sequence[float],
        blur_sigma2: Sequence[float],
        betag_range2: Sequence[float],
        betap_range2: Sequence[float],
        sinc_prob2: float,
        final_sinc_prob: float
    ) -> "RealESRGANDataset":
        super(RealESRGANFDataset, self).__init__()
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["center", "random", "none"], f"invalid crop type: {self.crop_type}"

        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        # a list for each kernel probability
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        # betag used in generalized Gaussian blur kernels
        self.betag_range = betag_range
        # betap used in plateau blur kernels
        self.betap_range = betap_range
        # the probability for sinc filters
        self.sinc_prob = sinc_prob

        self.blur_kernel_size2 = blur_kernel_size2
        self.kernel_list2 = kernel_list2
        self.kernel_prob2 = kernel_prob2
        self.blur_sigma2 = blur_sigma2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2
        self.sinc_prob2 = sinc_prob2
        
        # a final sinc filter
        self.final_sinc_prob = final_sinc_prob
        
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        
        # kernel size ranges from 7 to 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        # TODO: kernel range is now hard-coded, should be in the configure file
        # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

        name_list = ['clipiqa', 'maniqa-pipal', 'musiq']
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.thre = [0.4, 0.5, 45]
        self.metric_list = []
        for name in name_list:
            iqa_metric = pyiqa.create_metric(name, device=device)
            for p in iqa_metric.net.parameters():
                p.requires_grad = False
            self.metric_list.append(iqa_metric)
            print(name + ' lower_better: ' + str(iqa_metric.lower_better))


    @torch.no_grad()
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # -------------------------------- Load hq images -------------------------------- #
        flag = True
        while flag:
            hq_path = self.paths[index]
            success = False
            for _ in range(3):
                try:
                    pil_img = Image.open(hq_path).convert("RGB")
                    success = True
                    break
                except:
                    time.sleep(1)
            assert success, f"failed to load image {hq_path}"
            
            # pil_img = random_resize_img(pil_img, self.out_size * 2)
            if self.crop_type == "random":
                pil_img = random_crop_arr(pil_img, self.out_size)
            elif self.crop_type == "center":
                pil_img = center_crop_arr(pil_img, self.out_size)
            # self.crop_type is "none"
            else:
                pil_img = np.array(pil_img)
                assert pil_img.shape[:2] == (self.out_size, self.out_size)
            # hwc, rgb to bgr, [0, 255] to [0, 1], float32
            
            tmp_img = Image.fromarray(pil_img)
            tmp_tensor = TF.to_tensor(tmp_img).clone().detach()
            cnt = 0
            for i, iqa_metric in enumerate(self.metric_list):
                score = iqa_metric(tmp_tensor)
                if score < self.thre[i]:
                    cnt += 1
                elif i == 0:
                    flag = False
                    break
            if cnt <= 1:
                flag = False
            else:
                # tmp_img.save(f'/my_project_path/DiffBIR-main/fail/{index}.png')
                index = random.randint(0, len(self.paths) - 1)
            
        img_hq = (pil_img[..., ::-1] / 255.0).astype(np.float32)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_hq = augment(img_hq, self.use_hflip, self.use_rot)
        
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # [0, 1], BGR to RGB, HWC to CHW
        img_hq = torch.from_numpy(
            img_hq[..., ::-1].transpose(2, 0, 1).copy()
        ).float()
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        p_p = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
        n_p = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'

        return {
            "hq": img_hq, "kernel1": kernel, "kernel2": kernel2,
            "sinc_kernel": sinc_kernel, "txt": ''
        }
    
        if random.random() > 0.2:
            txt = p_p
        else:
            txt = n_p
            downsampled = F.interpolate(img_hq.unsqueeze(0), scale_factor=0.25, mode='bilinear')
            img_hq = F.interpolate(downsampled, scale_factor=4, mode='bilinear').squeeze()
            
        return {
            "hq": img_hq, "kernel1": kernel, "kernel2": kernel2,
            "sinc_kernel": sinc_kernel, "txt": txt
        }

    def __len__(self) -> int:
        return len(self.paths)
