# SimSwap-train
Reimplement of SimSwap training code<br />
- 20210919 这份代码原本是中秋节的时候写的；<br />
- 20211130 后来我们团队有换脸相关需求了，做了很多改进与优化，不过应该没法分享出来；<br />
- 20211211 我把这份代码的使用文档更新了一版，512pix是可以训的，希望能帮助到各位。<br /><br /><br />

# Instructions
## 1. Environment preparation
### Step 1.Install packages for python
1) Refer to the [SIMSWAP preparation](https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md) to install the python packages.<br />
2) Refer to the [SIMSWAP preparation](https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md) to download the 224-pix pretrained model (for finetune) or none and other necessary pretrained weights.<br /><br />
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
- We reconmend you with [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)) dataset which contains clear and diverse face images.<br /><br />
### Pre-Processing image files
- Run the commad with :<br />
`CUDA_VISIBLE_DEVICES=0 python make_dataset.py /`<br />
&emsp;&emsp;`--dataroot ./dataset/CelebA /`<br />
&emsp;&emsp;`--extract_size 512 /`<br />
&emsp;&emsp;`--output_img_dir ./dataset/CelebA/imgs /`<br />
&emsp;&emsp;`--output_latent_dir ./dataset/CelebA/latents`<br /><br />
### Getting extracte images and latents
- When data-processing is done, two folders will be created in `./dataset/CelebA/`:<br />
`./dataset/CelebA/imgs/`: extracted 512-pix images<br />
`./dataset/CelebA/latents/`: extracted image face latents embedded from ArcNet network<br /><br /><br /><br />

## 3. Start Training
### Finetuning
- Run the command with:<br />
`CUDA_VISIBLE_DEVICES=0 python train.py /`<br />
&emsp;&emsp;`--name CelebA_512_finetune /`<br />
&emsp;&emsp;`--which_epoch latest /`<br />
&emsp;&emsp;`--dataroot ./dataset/CelebA /`<br />
&emsp;&emsp;`--image_size 512 /`<br />
&emsp;&emsp;`--display_winsize 512 /`<br />
&emsp;&emsp;`--continue_train`<br /><br />
NOTICE:<br />
&emsp;&emsp;If `chekpoints/CelebA_512_finetune` is an un-existed folder, it will first copy the official model from `chekpoints/people/latest_net_*.pth` to `chekpoints/CelebA_512_finetune/`.<br /><br />

### New training
- Run the command with:<br />
`CUDA_VISIBLE_DEVICES=0 python train.py /`<br />
&emsp;&emsp;`--name CelebA_512 /`<br />
&emsp;&emsp;`--which_epoch latest /`<br />
&emsp;&emsp;`--dataroot ./dataset/CelebA /`<br />
&emsp;&emsp;`--image_size 512 /`<br />
&emsp;&emsp;`--display_winsize 512 /`<br /><br />

- When training is done, several files will be created in `chekpoints/CelebA_512_finetune` folder:<br />
`web/`: training-process visualization files<br />
`latest_net_G.pth`: Latest checkpoint of G network<br />
`latest_net_D1.pth`: Latest checkpoint of D1 network<br />
`latest_net_D2.pth`: Latest checkpoint of D2 network<br />
`loss_log.txt`: Doc to record loss during whole training process<br />
`iter.txt`: Doc to record iter information<br />
`iter.txt`: Doc to record options for the training<br />
<br /><br /><br /><br />


## 4.Training Result
### （1）CelebA with 224x224 res
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/train_celeba_224.png)

### （2）CelebA with 512x512 res
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/train_celeba_512_1.png)
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/train_celeba_512_2.png)


## 5.Inference
- I applied spNorm to the high-resolution image during training, which is conducive to the the model learning. Therefore, our Inference codes are different from official codes.<br />
- In order to be compatible with the official model, I modify all the places including
`swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]`<br />
to <br />
`swap_result = swap_model(None, spNorm(frame_align_crop_tenor), id_vetor, None, True)[0]` <br />

# Our work
&emsp;&emsp;I share with you the effect of improved version SimSwapHD, which has made changes in both structure and training-processing from our group: ByteDance AILab SA-TTS.<br />
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/apply_example.jpg)
&emsp;&emsp;Watch video here:<br />
&emsp;&emsp;Video file is here: ```docs/apply_example.mp4```<br /><br />


