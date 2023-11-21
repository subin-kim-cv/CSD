import os
import torch
import argparse
import PIL
import numpy as np
import glob
import mediapy as media
from csdedit import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_prompt', type=str, default='')
    parser.add_argument('--src_prompt', type=str, default='')
    parser.add_argument('--neg_prompt', type=str, default='')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--device', type=int ,default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--min_step", type=int, default=20)
    parser.add_argument("--max_step", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=5.)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--decay_iter", type=int, default=20)
    parser.add_argument("--decay_rate", type=float, default=0.9)
    parser.add_argument("--svgd", action="store_true")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--stride', default=16, type=int)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--stride_vae', action='store_true')
    parser.add_argument('--consistency_decoder', action='store_true')
    parser.add_argument("--rows", default=2, type=int)
    parser.add_argument("--cols", default=12, type=int)
    opt = parser.parse_args()


    seed_everything(opt.seed)
    os.makedirs(os.path.join(opt.save_path), exist_ok=True)
    
    # prepare datasets
    files = sorted(glob.glob(os.path.join(opt.data_path, "*")))
    images = []
    for file in files:
        image = PIL.Image.open(file).convert('RGB')
        opt.w, opt.h = image.size
        images.append(image.resize((512, 512)))
    
    # collaborative score distillation 
    sd = CSDEdit(opt)
    opt.device = torch.device(f'cuda:{opt.device}')
    opt.save_name = f'prompt_{opt.tgt_prompt}_gs_{opt.guidance_scale}_igs_{opt.image_guidance_scale}_batch_{opt.batch}_lr_{opt.lr}_dec_{opt.decay_rate}_stco_{opt.stride_vae}_svgd_{opt.svgd}.png'
    img = sd.edit_video(images, opt.guidance_scale, opt.image_guidance_scale)
    
    # save image
    img = [image.resize((opt.w, opt.h)) for image in img]
    img_grid = image_grid(img, rows=opt.rows, cols=opt.cols)
    img_grid.save(os.path.join(opt.save_path, opt.save_name))
    
    # save video
    vid = [np.array(image)[None, :] for image in img]
    vid = np.concatenate(vid, axis=0).astype(np.float32) / 255.0
    media.write_video(os.path.join(opt.save_path, 'outputs.mp4'), vid, fps=10)

