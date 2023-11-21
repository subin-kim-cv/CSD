# CSD
Official implementation of the paper **[Collaborative Score Distillation for Consistent Visual Editing](https://subin-kim-cv.github.io/CSD/)** (NeurIPS 2023).

[Subin Kim*](https://subin-kim-cv.github.io/)<sup>1</sup>, 
[Kyungmin Lee*](https://kyungmnlee.github.io/)<sup>1</sup>, 
[June Suk Choi](https://github.com/choi403)<sup>1</sup>, 
[Jongheon Jeong](https://jh-jeong.github.io/)<sup>1</sup>,
[Kihyuk Sohn](https://sites.google.com/site/kihyuksml)<sup>2</sup>,
[Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html)<sup>1</sup>.  
<sup>1</sup>KAIST, <sup>2</sup>Google Research  
[paper](https://subin-kim-cv.github.io/CSD/resources/kim2023csd.pdf) | [project page](https://subin-kim-cv.github.io/CSD/)
| [arXiv](https://arxiv.org/abs/2307.04787)

**TL;DR**: Consistent zero-shot visual synthesis across various and complex visual modalities

<p align="center">
    <img src=assets/concept_figure.png>
</p>

## Requirements
### Environments
Required packages you should install are listed below:
```
conda create -n csd python=3.8
conda activate csd
pip install torch==2.0.1 torchvision==0.15.2
pip install diffusers==0.20.0 transformers accelerate mediapy
# for consistency decoder
pip install git+https://github.com/openai/consistencydecoder.git
```

## Image Editing 
Run the following script with a single GPU.
```
python csdedit_image.py --device=0 --svgd --fp16 --stride=16 \
--save_path='output/' --data_path='data/river.jpg' \
--batch=4 --tgt_prompt='turn into van gogh style painting' \
--guidance_scale=7.5 --image_guidance_scale=5
```
<p align="center">
    <img src=assets/river_vangogh.png>
</p>

```
python csdedit_image.py --device=0 --svgd --fp16 --stride=16 \
--save_path='output/' --data_path='data/sheeps.jpg' \
--batch=4 --tgt_prompt='turn the sheeps into wolves' \
--guidance_scale=7.5 --image_guidance_scale=5 
```
<p align="center">
    <img src=assets/sheep_wolves.png>
</p>

To edit image of high resolution, we encode and decode in patch-wise. To do that, add '--stride_vae': 

```
python csdedit_image.py --device=0 --svgd --fp16 --stride=16 \
--save_path='output/' --data_path='data/michelangelo.jpeg' \
--batch=8 --tgt_prompt='Re-imagine people are in galaxy' \
--guidance_scale=15 --image_guidance_scale=5 --stride_vae --lr=4.0
```
<p align="center">
    <img src=assets/michelangelo_galaxy.png>
</p>

## Compositional Image Editing
To edit the image with region-wise prompts while ensuring smooth transitions between patches with different instructions, do the following:
```
python csdedit_image_region.py --device 0 --svgd --fp16 \
--save_path 'output/' --data_path 'data/vienna.jpg' \
--tgt_prompt 'turn into sunny weather' 'turn into cloudy weather' 'turn into rainy weather' 'turn into snowy weather' \
--stride 16 --batch 4 --guidance_scale 15 --image_guidance_scale 5
```
<p align="center">
    <img src=assets/region_vienna.png>
</p>


## Video Editing
```
python csdedit_video.py --device 0 --svgd --fp16 \
--save_path 'output/break/' --data_path 'data/break' \
--tgt_prompt="Change the color of his T-shirt to yellow" \
--guidance_scale=7.5 --image_guidance_scale=1.5 --lr=0.5 \
--rows 2 --cols 12 --svgd --num_steps 100 
```
<p align="center">
    <img src=assets/break/outputs.gif width="500"> 
</p>



## Citation
```
@inproceedings{
    kim2023collaborative,
    title={Collaborative score distillation for consistent visual editing},
    author={Kim, Subin and Lee, Kyungmin and Choi, June Suk and Jeong, Jongheon and Sohn, Kihyuk and Shin, Jinwoo},
    booktitle={Advances in Neural Information Processing Systems},
    year={2023},
}
```
