import os
import torch
import argparse
import PIL
from csdedit import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_prompt', nargs="+", type=str, required=True)
    parser.add_argument('--src_prompt', type=str, default='')
    parser.add_argument('--neg_prompt', type=str, default='')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--device', type=int ,default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--min_step", type=int, default=20)
    parser.add_argument("--max_step", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--guidance_scale", type=float, default=15)
    parser.add_argument("--image_guidance_scale", type=float, default=5.)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--decay_iter", type=int, default=20)
    parser.add_argument("--decay_rate", type=float, default=0.9)
    parser.add_argument("--svgd", action="store_true")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--stride', default=16, type=int)
    parser.add_argument('--batch', type=int, default=-1)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--stride_vae', action='store_true')
    parser.add_argument('--consistency_decoder', action='store_true')
    opt = parser.parse_args()
   
   
    seed_everything(opt.seed)
    os.makedirs(opt.save_path, exist_ok=True)
    
    # prepare datasets
    image = PIL.Image.open(opt.data_path).convert('RGB')
    w, h = image.size
    new_w, new_h = 1920, 512
    image = image.resize((new_w, new_h), resample=PIL.Image.Resampling.LANCZOS)
    
    opt.n_weights = len(opt.tgt_prompt)
    
    # collaborative score distillation 
    sd = CSDEdit(opt)
    opt.device = torch.device(f'cuda:{opt.device}') 
    opt.save_name = f'prompt_{opt.tgt_prompt}_gs_{opt.guidance_scale}_igs_{opt.image_guidance_scale}.png'
    img = sd.edit_image_region(image, new_h, new_w, opt.guidance_scale, opt.image_guidance_scale)
    os.makedirs(opt.save_path, exist_ok=True)
        
    # save image
    img = img.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
    img.save(os.path.join(opt.save_path, opt.save_name))