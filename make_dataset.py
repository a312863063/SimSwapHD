import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap import video_swap
import os
import glob
import tqdm
import argparse
# Note: /home/xxx/anaconda3/lib/python3.x/site-packages/insightface/utils/face_align.py should modified below
# line28: src_all = np.array([src1, src2, src3, src4, src5])
# line53: src = src_all * image_size / 112

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('parse parameters')
    parser.add_argument('--dataroot', type=str, default='../dataset/CelebA')
    parser.add_argument('--extract_size', type=int, default=512)
    parser.add_argument('--output_img_dir', type=str, default='./dataset/CelebA/imgs')
    parser.add_argument('--output_latent_dir', type=str, default='./dataset/CelebA/latents')
    parser.add_argument('--detect_checkpoint_dir', type=str, default='./insightface_func/models')
    parser.add_argument('--netArc_checkpoint', type=str, default='./arcface_model/arcface_checkpoint.tar')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    opt = parser.parse_args()

    torch.nn.Module.dump_patches = True
    app = Face_detect_crop(name='antelope', root=opt.detect_checkpoint_dir)
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
    netArc_checkpoint = torch.load(opt.netArc_checkpoint)
    netArc = netArc_checkpoint['model'].module
    netArc = netArc.to(opt.device)
    netArc.eval()

    images_list = glob.glob(opt.dataroot+'/*/*.jpg') + glob.glob(opt.dataroot+'/*/*.jpeg') + \
                  glob.glob(opt.dataroot+'/*/*.bmp') + glob.glob(opt.dataroot+'/*/*.png')
    os.makedirs(opt.output_img_dir, exist_ok=True)
    os.makedirs(opt.output_latent_dir, exist_ok=True)
    i = 0
    
    for image_path in tqdm.tqdm(images_list):
        try:
            with torch.no_grad():
                img_a_whole = cv2.imread(image_path)
                if img_a_whole is None:
                    print('wrong picture:', image_path)
                    continue
                img_a_align_crop, _ = app.get(img_a_whole, opt.extract_size)
                if not img_a_align_crop:
                    continue
                img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
                img_a = transformer_Arcface(img_a_align_crop_pil)
                img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
                img_id = img_id.to(opt.device)
                img_id_downsample = F.interpolate(img_id, scale_factor=112/opt.extract_size)
                latend_id = netArc(img_id_downsample)
                latend_id = F.normalize(latend_id, p=2, dim=1).cpu().numpy()
                img_a_align_crop_pil.save(os.path.join(opt.output_img_dir, str(i).zfill(7)+'.png'))
                np.save(os.path.join(opt.output_latent_dir, str(i).zfill(7)+'.npy'), latend_id)
                i += 1
        except:
            print('fail when process picture:', image_path)
            continue


