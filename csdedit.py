from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionInstructPix2PixPipeline
from tqdm import tqdm
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import PIL


##################################################################

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def PIL2Tensor(image):
    w, h = image[0].size
    w, h = (x - x % 512 for x in (w, h))  # resize to integer multiple of 8
    image = [np.array(i.resize((w, h), resample=PIL.Image.Resampling.LANCZOS))[None, :] for i in image]
    # image = [np.array(i.resize((512, 512)))[None, :] for i in image]
    image = np.concatenate(image, axis=0)
    image = np.array(image).astype(np.float32) / 255.0
    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        image = [np.array(i.resize((w, h), resample=PIL.Image.Resampling.LANCZOS))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

##################################################################

CONST_SCALE = 0.18215

WEIGHTS_4 = [
    [0.9375, 0.0625, 0.,     0.    ],
    [0.6875, 0.3125, 0.,     0.    ],
    [0.4375, 0.5625, 0.,     0.    ],
    [0.1875, 0.8125, 0.,     0.    ],
    [0.    , 0.875 , 0.125 , 0.    ],
    [0.    , 0.625 , 0.375 , 0.    ],
    [0.    , 0.375 , 0.625 , 0.    ],
    [0.    , 0.125 , 0.875 , 0.    ],
    [0.    , 0.    , 0.8125, 0.1875],
    [0.    , 0.    , 0.5625, 0.4375],
    [0.    , 0.    , 0.3125, 0.6875],
    [0.    , 0.    , 0.0625, 0.9375],
]
WEIGHTS_3 = [
    [1.,   0.,   0.],
    [1.,   0.,   0.],
    [0.75, 0.25, 0.],
    [0.5,  0.5,  0.],
    [0.25, 0.75, 0.],
    [0.,   1.,   0.],
    [0.,   1.,   0.],
    [0.,   0.75, 0.25],
    [0.,   0.5,  0.5],
    [0.,   0.25, 0.75],
    [0.,   0.,   1.],
    [0.,   0.,   1.],
]
WEIGHTS_2 = [
    [1.   , 0.   ],
    [1.   , 0.   ],
    [1.   , 0.   ],
    [1.   , 0.   ],
    [0.875, 0.125],
    [0.625, 0.375],
    [0.375, 0.625],
    [0.125, 0.875],
    [0.   , 1.   ],
    [0.   , 1.   ],
    [0.   , 1.   ],
    [0.   , 1.   ],
]


class CSDEdit(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = opt.device
        model_key = 'timbrooks/instruct-pix2pix'
        self.fp16 = opt.fp16
        self.precision_t = torch.float16 if self.fp16 else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_key, 
                                                                           torch_dtype=self.precision_t, 
                                                                           safety_checker=None)
        self.pipe.to(self.device)
        print(f'[INFO] loading InstructPix2Pix...')
        
        # improve memory performance trading compute time
        self.pipe.enable_attention_slicing()
        
        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        
        if self.fp16:
            self.unet.half()
            self.vae.half()

        self.scheduler = DDIMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder="scheduler")
        self.scheduler.set_timesteps(100)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore
        
        if opt.consistency_decoder:
            from consistencydecoder import ConsistencyDecoder
            self.decoder_consistency = ConsistencyDecoder(device=opt.device)
        print(f'[INFO] loaded InstructPix2Pix!')
        
    def rbf_kernel(self, X, Y, gamma=-1, ad=1):
        # X and Y should be tensors with shape (batch_size, num_channels, height, width)
        # gamma is a hyperparameter controlling the width of the RBF kernel

        # Reshape X and Y to have shape (batch_size, num_channels*height*width)
        X_flat = X.view(X.size(0), -1)
        Y_flat = Y.view(Y.size(0), -1)

        # Compute the pairwise squared Euclidean distances between the samples
        with torch.cuda.amp.autocast():
            dists = torch.cdist(X_flat, Y_flat, p=2)**2

        if gamma <0: # use median trick
            gamma = torch.median(dists)
            gamma = torch.sqrt(0.5 * gamma / np.log(dists.size(0) + 1))
            gamma = 1 / (2 * gamma**2)
            # print(gamma)

        gamma = gamma * ad 
        # gamma = torch.max(gamma, torch.tensor(1e-3))
        # Compute the RBF kernel using the squared distances and gamma
        K = torch.exp(-gamma * dists)
        dK = -2 * gamma * K.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (X.unsqueeze(1) - Y.unsqueeze(0))
        dK_dX = torch.sum(dK, dim=1)

        return K, dK_dX

    
    def get_views(self, panorama_height, panorama_width, window_size=64, stride=32):
        panorama_height /= 8
        panorama_width /= 8
        num_blocks_height = (panorama_height - window_size) // stride + 1
        num_blocks_width = (panorama_width - window_size) // stride + 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            views.append((h_start, h_end, w_start, w_end))
        return np.array(views)


    
    def get_views_region(self, panorama_height, panorama_width, window_size=64, stride=32, n_weights=4): # brute force
        panorama_height /= 8
        panorama_width /= 8
        num_blocks_height = (panorama_height - window_size) // stride + 1
        num_blocks_width = (panorama_width - window_size) // stride + 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        if n_weights ==4:
            weights = WEIGHTS_4
        elif n_weights == 3:
            weights = WEIGHTS_3
        elif n_weights == 2:
            weights = WEIGHTS_2
            
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            views.append((h_start, h_end, w_start, w_end))
                        
        return np.array(views), np.array(weights)

    @torch.no_grad()
    def encode_latents(self, images):
        images = preprocess(images).to(self.device)
        if self.fp16:
            images = images.half()
        posterior = self.vae.encode(images).latent_dist.sample()
        latent = posterior * CONST_SCALE
        return latent
           
    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / CONST_SCALE * latents
        if self.opt.consistency_decoder:
            imgs = self.decoder_consistency(latents)
        else:
            imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
    
    @torch.no_grad()
    def encode_latents_stride(self, images):
        images = preprocess(images).to(self.device)
        if self.fp16:
            images = images.half()
        width = images.size(3) // 8
        height = images.size(2) // 8
        n_blocks_width  = (width -  64) // self.opt.stride + 1
        n_blocks_height = (height - 64) // self.opt.stride + 1
        
        count = torch.zeros([1, 4, height, width], 
                            dtype=self.precision_t, device=images.device)
        value = torch.zeros([1, 4, height, width], 
                            dtype=self.precision_t, device=images.device)
        for h in range(n_blocks_height):
            for w in range(n_blocks_width):
                h_start = h * self.opt.stride
                img_h_start = 8 * h_start
                h_end = h_start + 64
                img_h_end = img_h_start + 512
                w_start = w * self.opt.stride
                img_w_start = 8 * w_start
                w_end = w_start + 64
                img_w_end = img_w_start + 512
                latents = self.vae.encode(images[:,:,img_h_start:img_h_end, img_w_start:img_w_end]).latent_dist.sample()
                value[:, :, h_start:h_end, w_start:w_end] += latents
                count[:, :, h_start:h_end, w_start:w_end] += 1
        latents = torch.where(count > 0, value / count, value)
        latent = latents * CONST_SCALE
        return latent
    
    @torch.no_grad()
    def decode_latents_stride(self, latents):
        latents = 1 / CONST_SCALE * latents
        width  = latents.size(3) * 8
        height = latents.size(2) * 8
        n_blocks_width  = (latents.size(3) - 64) // self.opt.stride + 1
        n_blocks_height = (latents.size(2) - 64) // self.opt.stride + 1
        count = torch.zeros([1, 3, height, width], 
                            dtype=self.precision_t, device=latents.device)
        value = torch.zeros([1, 3, height, width], 
                            dtype=self.precision_t, device=latents.device)
        for h in range(n_blocks_height):
            for w in range(n_blocks_width):
                h_start = h * self.opt.stride 
                img_h_start = 8 * h_start
                h_end = h_start + 64
                img_h_end = img_h_start + 512
                w_start = w * self.opt.stride
                img_w_start = 8 * w_start
                w_end = w_start + 64
                img_w_end = img_w_start + 512
                if self.opt.consistency_decoder:
                    imgs = self.decoder_consistency(latents[:, :, h_start:h_end, w_start:w_end])
                else:
                    imgs = self.vae.decode(latents[:, :, h_start:h_end, w_start:w_end]).sample
                value[:, :, img_h_start:img_h_end, img_w_start:img_w_end] += imgs
                count[:, :, img_h_start:img_h_end, img_w_start:img_w_end] += 1
        imgs = torch.where(count > 0, value / count, value)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def edit_image(self, image, height=512, width=2048,
                      guidance_scale=7.5, image_guidance_scale=1.5):

        if isinstance(self.opt.tgt_prompt, str):
            tgt_prompt = [self.opt.tgt_prompt]

        if isinstance(self.opt.neg_prompt, str):
            neg_prompt = [self.opt.neg_prompt]
        
        if isinstance(self.opt.src_prompt, str):
            src_prompt = [self.opt.src_prompt]

        with torch.no_grad():
            src_text_embeds = self.pipe._encode_prompt(
                src_prompt, device=self.device, num_images_per_prompt=1, 
                do_classifier_free_guidance=True, negative_prompt=neg_prompt
            )
            tgt_text_embeds = self.pipe._encode_prompt(
                tgt_prompt, device=self.device, num_images_per_prompt=1, 
                do_classifier_free_guidance=True, negative_prompt=neg_prompt
            )
            if self.opt.stride_vae:
                src_latent = self.encode_latents_stride([image])
            else:
                src_latent = self.encode_latents([image])
                
        tgt_latent = src_latent.clone().detach().to(self.device)
        tgt_latent.requires_grad = True 
        views = self.get_views(height, width, stride=self.opt.stride)

        count = torch.zeros_like(tgt_latent)
        value = torch.zeros_like(tgt_latent)
                
        optimizer = torch.optim.SGD([tgt_latent], lr=self.opt.lr, weight_decay=self.opt.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=self.opt.decay_iter, 
                                                    gamma=self.opt.decay_rate)

        for step in tqdm(range(self.opt.num_steps)):
            optimizer.zero_grad()
            t = torch.randint(self.opt.min_step, self.opt.max_step + 1,
                              [1], dtype=torch.long, device=self.device)
            
            # initialize 
            count.zero_()
            value.zero_()
            if self.opt.batch > 0:
                indices = np.random.permutation(len(views))
                indices = indices[:self.opt.batch]
                train_views = views[indices]
            else:
                train_views = views
            
            batch_size = len(train_views)
            latent_views = torch.empty([batch_size, 4, 64, 64], dtype=self.precision_t).to(self.device)
            scores = torch.empty([batch_size, 4, 64, 64], dtype=self.precision_t).to(self.device)
            for i, (h_start, h_end, w_start, w_end) in enumerate(train_views):
                with torch.no_grad():
                    src_latent_view = src_latent[:, :, h_start:h_end, w_start:w_end]
                    tgt_latent_view = tgt_latent[:, :, h_start:h_end, w_start:w_end]
                    uncond_image_latents_view = torch.zeros_like(src_latent_view)
                    image_latents = torch.cat([src_latent_view, src_latent_view, uncond_image_latents_view], dim=0)    
                
                    noise = torch.randn_like(src_latent_view)
                    src_latent_noisy = self.scheduler.add_noise(src_latent_view, noise, t)
                    tgt_latent_noisy = self.scheduler.add_noise(tgt_latent_view, noise, t) 
                    
                    src_model_input = torch.cat([src_latent_noisy] * 3)
                    src_model_input = torch.cat([src_model_input, image_latents], dim=1)
                    src_noise_pred = self.unet(
                        src_model_input, t, encoder_hidden_states=src_text_embeds).sample
                    
                    tgt_model_input = torch.cat([tgt_latent_noisy] * 3)
                    tgt_model_input = torch.cat([tgt_model_input, image_latents], dim=1)
                    tgt_noise_pred = self.unet(
                        tgt_model_input, t, encoder_hidden_states=tgt_text_embeds).sample
                
                # perform guidance
                src_noise_pred_text, src_noise_pred_image, src_noise_pred_uncond = src_noise_pred.chunk(3)
                src_noise_pred = (
                    src_noise_pred_uncond
                    + guidance_scale * (src_noise_pred_text - src_noise_pred_image)
                    + image_guidance_scale * (src_noise_pred_image - src_noise_pred_uncond)
                )
                tgt_noise_pred_text, tgt_noise_pred_image, tgt_noise_pred_uncond = tgt_noise_pred.chunk(3)
                tgt_noise_pred = (
                    tgt_noise_pred_uncond
                    + guidance_scale * (tgt_noise_pred_text - tgt_noise_pred_image)
                    + image_guidance_scale * (tgt_noise_pred_image - tgt_noise_pred_uncond)
                )
                
                noise = tgt_noise_pred - src_noise_pred
                # latent_views[i] = tgt_latent_view
                latent_views[i] = tgt_latent_noisy
                scores[i] = noise
                            
            w_t = (1-self.alphas[t])
            if self.opt.svgd:
                with torch.cuda.amp.autocast():
                    K, dK_dX = self.rbf_kernel(latent_views, latent_views, gamma=-1, ad=1)
                    scores = w_t * (torch.matmul(scores.transpose(0,3), K).transpose(0,3) + dK_dX) / K.size(0)
                
            for j, (h_start, h_end, w_start, w_end) in enumerate(train_views):
                value[:, :, h_start:h_end, w_start:w_end] += scores[j]
                count[:, :, h_start:h_end, w_start:w_end] += 1
                grad_all = torch.where(count > 0, value / count, value)
                                                
            tgt_latent.backward(gradient=grad_all, retain_graph=True)
            
            optimizer.step()
            scheduler.step()

        # Img latents -> imgs
        with torch.no_grad():
            if self.opt.stride_vae:
                imgs = self.decode_latents_stride(tgt_latent)  # [1, 3, 512, 512]
            else:
                imgs = self.decode_latents(tgt_latent)
            img = T.ToPILImage()(imgs[0].cpu())
        return img
    
    
    @torch.no_grad()
    def edit_image_region(self, image, height=512, width=2048,
                      guidance_scale=7.5, image_guidance_scale=1.5):

        if isinstance(self.opt.tgt_prompt, str):
            tgt_prompt = [self.opt.tgt_prompt]
        else:
            tgt_prompt = self.opt.tgt_prompt

        n_prompt = len(tgt_prompt)
        if isinstance(self.opt.neg_prompt, str):
            neg_prompt = [self.opt.neg_prompt] * n_prompt
        
        if isinstance(self.opt.src_prompt, str):
            src_prompt = [self.opt.src_prompt] * n_prompt

        with torch.no_grad():
            src_text_embeds = self.pipe._encode_prompt(
                src_prompt, device=self.device, num_images_per_prompt=1, 
                do_classifier_free_guidance=True, negative_prompt=neg_prompt
            )
            tgt_text_embeds = self.pipe._encode_prompt(
                tgt_prompt, device=self.device, num_images_per_prompt=1, 
                do_classifier_free_guidance=True, negative_prompt=neg_prompt
            )
        
            if self.opt.stride_vae:
                src_latent = self.encode_latents_stride([image])
            else:
                src_latent = self.encode_latents([image])

        tgt_latent = src_latent.clone().detach().to(self.device)
        tgt_latent.requires_grad = True 
        views, weights = self.get_views_region(height, width, stride=self.opt.stride, n_weights=self.opt.n_weights)
        
        count = torch.zeros_like(tgt_latent)
        value = torch.zeros_like(tgt_latent)
        optimizer = torch.optim.SGD([tgt_latent], lr=self.opt.lr, weight_decay=self.opt.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=self.opt.decay_iter, 
                                                    gamma=self.opt.decay_rate)

        for step in tqdm(range(self.opt.num_steps)):
            optimizer.zero_grad()
            t = torch.randint(self.opt.min_step, self.opt.max_step + 1,
                              [1], dtype=torch.long, device=self.device)
            # initialize 
            count.zero_()
            value.zero_()
            if self.opt.batch > 0:
                indices = np.random.permutation(len(views))
                indices = indices[:self.opt.batch]
                train_views = views[indices]
                train_weights = weights[indices]
            else:
                train_views = views
                train_weights = weights
            batch_size = len(train_views)
            latent_views = torch.empty([batch_size, 4, 64, 64], dtype=self.precision_t).to(self.device)
            scores = torch.zeros([batch_size, 4, 64, 64], dtype=self.precision_t).to(self.device)
            for i, (h_start, h_end, w_start, w_end) in enumerate(train_views):
                with torch.no_grad():
                    src_latent_view = src_latent[:, :, h_start:h_end, w_start:w_end]
                    src_latent_view_batch = src_latent_view.repeat(n_prompt, 1,1,1)
                    tgt_latent_view = tgt_latent[:, :, h_start:h_end, w_start:w_end]
                    tgt_latent_view_batch = tgt_latent_view.repeat(n_prompt, 1,1,1)
                    uncond_image_latents_view = torch.zeros_like(src_latent_view_batch)
                    image_latents = torch.cat([src_latent_view_batch, src_latent_view_batch, uncond_image_latents_view], dim=0)    
                
                    noise = torch.randn_like(src_latent_view_batch)
                    src_latent_noisy = self.scheduler.add_noise(src_latent_view_batch, noise, t)
                    tgt_latent_noisy = self.scheduler.add_noise(tgt_latent_view_batch, noise, t) 
                    
                    src_model_input = torch.cat([src_latent_noisy] * 3)
                    src_model_input = torch.cat([src_model_input, image_latents], dim=1)
                    src_noise_pred = self.unet(
                        src_model_input, t, encoder_hidden_states=src_text_embeds).sample
                    
                    tgt_model_input = torch.cat([tgt_latent_noisy] * 3)
                    tgt_model_input = torch.cat([tgt_model_input, image_latents], dim=1)
                    tgt_noise_pred = self.unet(
                        tgt_model_input, t, encoder_hidden_states=tgt_text_embeds).sample
                
                # perform guidance
                src_noise_pred_text, src_noise_pred_image, src_noise_pred_uncond = src_noise_pred.chunk(3)
                src_noise_pred = (
                    src_noise_pred_uncond
                    + guidance_scale * (src_noise_pred_text - src_noise_pred_image)
                    + image_guidance_scale * (src_noise_pred_image - src_noise_pred_uncond)
                )
                tgt_noise_pred_text, tgt_noise_pred_image, tgt_noise_pred_uncond = tgt_noise_pred.chunk(3)
                tgt_noise_pred = (
                    tgt_noise_pred_uncond
                    + guidance_scale * (tgt_noise_pred_text - tgt_noise_pred_image)
                    + image_guidance_scale * (tgt_noise_pred_image - tgt_noise_pred_uncond)
                )
                
                noise = tgt_noise_pred - src_noise_pred
                
                for j,w in enumerate(train_weights[i]):
                    scores[i] += w * noise[j]
                # latent_views[i] = tgt_latent_view
                latent_views[i] = tgt_latent_noisy[0]
            
            w_t = (1-self.alphas[t])
            if self.opt.svgd:
                with torch.cuda.amp.autocast():
                    K, dK_dX = self.rbf_kernel(latent_views, latent_views, gamma=-1, ad=1)
                    scores = w_t * (torch.matmul(scores.transpose(0,3), K).transpose(0,3) + dK_dX) / K.size(0)
                
            for j, (h_start, h_end, w_start, w_end) in enumerate(train_views):
                value[:, :, h_start:h_end, w_start:w_end] += scores[j]
                count[:, :, h_start:h_end, w_start:w_end] += 1
                grad_all = torch.where(count > 0, value / count, value)
                                                
            tgt_latent.backward(gradient=grad_all, retain_graph=True)
            
            optimizer.step()
            scheduler.step()
        
        with torch.no_grad():
            if self.opt.stride_vae:
                imgs = self.decode_latents_stride(tgt_latent)
            else:
                imgs = self.decode_latents(tgt_latent)  # [1, 3, 512, 512]
            img = T.ToPILImage()(imgs[0].cpu())
        
        return img


    @torch.no_grad()
    def edit_video(self, video, guidance_scale=7.5, image_guidance_scale=1.5):
        
        batch_size = len(video)
        
        if isinstance(self.opt.tgt_prompt, str):
            tgt_prompts = [self.opt.tgt_prompt] * batch_size

        if isinstance(self.opt.neg_prompt, str):
            neg_prompts = [self.opt.neg_prompt] * batch_size
        
        if isinstance(self.opt.src_prompt, str):
            src_prompts = [self.opt.src_prompt] * batch_size

        with torch.no_grad():
            src_text_embeds = self.pipe._encode_prompt(
                src_prompts, device=self.device, num_images_per_prompt=1, 
                do_classifier_free_guidance=True, negative_prompt=neg_prompts
            )
            tgt_text_embeds = self.pipe._encode_prompt(
                tgt_prompts, device=self.device, num_images_per_prompt=1, 
                do_classifier_free_guidance=True, negative_prompt=neg_prompts
            )
            src_latents = self.encode_latents(video)
        
        src_latents.requires_grad = False       
        tgt_latents = src_latents.clone().detach().to(self.device)
        tgt_latents.requires_grad = True 
        
        uncond_image_latents = torch.zeros_like(tgt_latents)
        image_cond_latents = torch.cat([src_latents, src_latents, uncond_image_latents], dim=0)
    
                
        optimizer = torch.optim.SGD([tgt_latents], lr=self.opt.lr, weight_decay=self.opt.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=self.opt.decay_iter, 
                                                    gamma=self.opt.decay_rate)
        
        for step in tqdm(range(self.opt.num_steps)):
            optimizer.zero_grad()
            t = torch.randint(self.opt.min_step, self.opt.max_step + 1,
                              [1], dtype=torch.long, device=self.device)

            with torch.no_grad():
                # add noise
                noise = torch.randn_like(src_latents)
                src_latents_noisy = self.scheduler.add_noise(src_latents, noise, t)
                tgt_latents_noisy = self.scheduler.add_noise(tgt_latents, noise, t)
            
                src_model_input = torch.cat([src_latents_noisy] * 3)
                src_model_input = torch.cat([src_model_input, image_cond_latents], dim=1)
                
                src_noise_pred = self.unet(
                    src_model_input, t, encoder_hidden_states=src_text_embeds).sample
                
                tgt_model_input = torch.cat([tgt_latents_noisy] * 3)
                tgt_model_input = torch.cat([tgt_model_input, image_cond_latents], dim=1)
                tgt_noise_pred = self.unet(
                    tgt_model_input, t, encoder_hidden_states=tgt_text_embeds).sample

                
            # perform guidance
            src_noise_pred_text, src_noise_pred_image, src_noise_pred_uncond = src_noise_pred.chunk(3)
            src_noise_pred = (
                    src_noise_pred_uncond
                    + guidance_scale * (src_noise_pred_text - src_noise_pred_image)
                    + image_guidance_scale * (src_noise_pred_image - src_noise_pred_uncond)
                )
            tgt_noise_pred_text, tgt_noise_pred_image, tgt_noise_pred_uncond = tgt_noise_pred.chunk(3)
            tgt_noise_pred = (
                    tgt_noise_pred_uncond
                    + guidance_scale * (tgt_noise_pred_text - tgt_noise_pred_image)
                    + image_guidance_scale * (tgt_noise_pred_image - tgt_noise_pred_uncond)
                )
                
            noise = tgt_noise_pred - src_noise_pred

            w_t = (1-self.alphas[t])
            if self.opt.svgd:
                with torch.cuda.amp.autocast():
                    K, dK_dX = self.rbf_kernel(tgt_latents_noisy, tgt_latents_noisy, gamma=-1, ad=1)
                    scores = torch.matmul(noise.transpose(0,3), K).transpose(0,3) + dK_dX
                    grad = w_t * scores / K.size(0)
            else:
                grad = w_t * noise
                                                
            tgt_latents.backward(gradient=grad, retain_graph=True)
            
            optimizer.step()
            scheduler.step()
            
        # Img latents -> imgs
        with torch.no_grad():
            imgs = self.decode_latents(tgt_latents)
            imgs = [T.ToPILImage()(image.cpu()) for image in imgs]
        return imgs

