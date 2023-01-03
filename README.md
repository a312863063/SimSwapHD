# SimSwapHD
Reimplement of SimSwap training code<br />
- 20210919 这份代码原本是中秋节的时候写的；<br />
- 20211005 256px的模型分享给大家，细节上会比官方224px的好一点点；<br />
- 20221122 512px的模型已优化完成，但是属于客户定制的没法分享（效果大概是下面这样）：<br />
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/test1.jpg)
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/test2.jpg)
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/test_compare.jpg)
- 有兴趣做些新内容的朋友可以来联系我，不是换脸，参看[这个网站](http://www.seeprettyface.com/)。<br />
<br /><br /><br />

# Instructions
## 1. Environment preparation
### Step 1.Install packages for python
1) Refer to the [SIMSWAP preparation](https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md) to install the python packages.<br />
2) Refer to the [SIMSWAP preparation](https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md) to download the 224-pix pretrained model (for finetune) or none and other necessary pretrained weights.<br />
3) Or you can download my 256px-pretrained-weight:[Baidu Netdisk](https://pan.baidu.com/s/1FyuAtL208dXCA8OxSJRpXg)（Code：i497）<br /><br />
### Step 2.Modify the ```insightface``` package to support arbitrary-resolution training
- If you use CONDA and conda environment name is ```simswap```, then find the code in place: <br />
 `C://Anaconda/envs/simswap/Lib/site-packages/insightface/utils/face_align.py`<br /><br />
change <b>#line 28 & 29</b>:<br />
`src = np.array([src1, src2, src3, src4, src5])`<br />
`src_map = {112: src, 224: src * 2}`<br />
into<br />
`src_all = np.array([src1, src2, src3, src4, src5])`<br />
`#src_map = {112: src, 224: src * 2}`<br /><br />
change <b>#line 53</b>:<br />
`src = src_map[image_size]`<br />
into<br />
`src = src_all * image_size / 112`<br /><br />
After modifying code, we can extract faces of any resolution and pass them to the model for training. <br />
- If you don't use CONDA, just find the location of the package and change the code in the same way as above.<br /><br /><br /><br />



## 2. Preparing training data
### Preparing image files
- Put all the image files in your datapath (eg. `./dataset/CelebA`)<br />
- We recommend you with [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset which contains clear and diverse face images.<br /><br />
### Pre-Processing image files
- Run the commad with :<br />
`CUDA_VISIBLE_DEVICES=0 python make_dataset.py \`<br />
&emsp;&emsp;`--dataroot ./dataset/CelebA \`<br />
&emsp;&emsp;`--extract_size 512 \`<br />
&emsp;&emsp;`--output_img_dir ./dataset/CelebA/imgs \`<br />
&emsp;&emsp;`--output_latent_dir ./dataset/CelebA/latents`<br /><br />
### Getting extracted images and latents
- When data-processing is done, two folders will be created in `./dataset/CelebA/`:<br />
`./dataset/CelebA/imgs/`: extracted 512-pix images<br />
`./dataset/CelebA/latents/`: extracted image face latents embedded from ArcNet network<br /><br /><br /><br />

## 3. Start Training
### Finetuning
- Run the command with:<br />
`CUDA_VISIBLE_DEVICES=0 python train.py \`<br />
&emsp;&emsp;`--name CelebA_512_finetune \`<br />
&emsp;&emsp;`--which_epoch latest \`<br />
&emsp;&emsp;`--dataroot ./dataset/CelebA \`<br />
&emsp;&emsp;`--image_size 512 \`<br />
&emsp;&emsp;`--display_winsize 512 \`<br />
&emsp;&emsp;`--continue_train`<br /><br />
NOTICE:<br />
&emsp;&emsp;If `chekpoints/CelebA_512_finetune` is an un-existed folder, it will first copy the official model from `chekpoints/people/latest_net_*.pth` to `chekpoints/CelebA_512_finetune/`.<br /><br />

### New training
- Run the command with:<br />
`CUDA_VISIBLE_DEVICES=0 python train.py \`<br />
&emsp;&emsp;`--name CelebA_512 \`<br />
&emsp;&emsp;`--which_epoch latest \`<br />
&emsp;&emsp;`--dataroot ./dataset/CelebA \`<br />
&emsp;&emsp;`--image_size 512 \`<br />
&emsp;&emsp;`--display_winsize 512`<br /><br />

- When training is done, several files will be created in `chekpoints/CelebA_512_finetune` folder:<br />
`web/`: training-process visualization files<br />
`latest_net_G.pth`: Latest checkpoint of G network<br />
`latest_net_D1.pth`: Latest checkpoint of D1 network<br />
`latest_net_D2.pth`: Latest checkpoint of D2 network<br />
`loss_log.txt`: Doc to record loss during whole training process<br />
`iter.txt`: Doc to record iter information<br />
`opt.txt`: Doc to record options for the training<br />
<br /><br /><br /><br />


## 4.Training Result
### （1）CelebA with 224x224 res
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/train_celeba_224.png)

### （2）CelebA with 512x512 res
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/train_celeba_512_1.png)
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/train_celeba_512_2.png)
<br /><br /><br /><br />

## 5.Inference
### Face swapping for video with 1 face
- Run the command with:<br />
`python test_video_swapsingle.py \`<br />
&emsp;&emsp;`--image_size 512 \`<br />
&emsp;&emsp;`--use_mask \`<br />
&emsp;&emsp;`--name CelebA_512_finetune \`<br />
&emsp;&emsp;`--Arc_path arcface_model/arcface_checkpoint.tar \`<br />
&emsp;&emsp;`--pic_a_path ./demo_file/Iron_man.jpg \`<br />
&emsp;&emsp;`--video_path ./demo_file/multi_people_1080p.mp4 \`<br />
&emsp;&emsp;`--output_path ./output/multi_test_swapsingle.mp4 \`<br />
&emsp;&emsp;`--temp_path ./temp_results `<br /><br />

### Face swapping for video/images with more faces
- We inherited almost the same usage from [SIMSWAP guidance](https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/usage.md)
<br /><br />

### Differences from our codes and official codes
- param `crop_size` -> `image_size` <br />
- I applied spNorm to the high-resolution image during training, which is conducive to the the model learning.<br />
- This code can be compatible with the official SimSwap pretrained-weight.<br />
<br /><br /><br />
