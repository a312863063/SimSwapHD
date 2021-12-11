# SimSwap-train
Reimplement of SimSwap training code<br />
这份代码原本是中秋节的时候写的；<br />
后来我们团队有换脸相关需求了，又做了很多优化，不过没法分享出来；<br />
我把这份代码的使用文档更新了一版，512pix是可以训的，希望能帮助到各位。<br /><br /><br />

# Instructions
## 1.Environment preparation
### Step 1.Install packages for python
&emsp;&emsp;1) Refer to the [SIMSWAP preparation](https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md) to install the python packages.<br />
&emsp;&emsp;2) Refer to the [SIMSWAP preparation](https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md) to download the 224-pix pretrained model (for finetune) or none and other necessary pretrained weights.<br /><br />
### Step 2.Modify the ```insightface``` package to support arbitrary-resolution training
#### &emsp;&emsp;CONDA environment (recommend)
&emsp;&emsp;If your use CONDA and conda environment name is ```simswap```, then find the code in place: <br />
&emsp;&emsp;&emsp;&emsp; `C://Anaconda/envs/```simswap```/Lib/site-packages/insightface/utils/face_align.py`<br /><br />
&emsp;&emsp;change #line 28 & 29:<br />
&emsp;&emsp;&emsp;&emsp;`src = np.array([src1, src2, src3, src4, src5])`<br />
&emsp;&emsp;&emsp;&emsp;`src_map = {112: src, 224: src * 2}`<br />
&emsp;&emsp;into<br />
&emsp;&emsp;&emsp;&emsp;`src_all = np.array([src1, src2, src3, src4, src5])`<br />
&emsp;&emsp;&emsp;&emsp;`#src_map = {112: src, 224: src * 2}`<br /><br />
&emsp;&emsp;change #line 53:<br />
&emsp;&emsp;&emsp;&emsp;`src = src_map[image_size]`<br />
&emsp;&emsp;into<br />
&emsp;&emsp;&emsp;&emsp;`src = src_all * image_size / 112`<br /><br />
#### &emsp;&emsp;None-CONDA environment
&emsp;&emsp;&emsp;&emsp;Just find the location of the package and change the code in the same way as above.<br /><br /><br /><br />



## 2.Preparing training data
&emsp;&emsp;Put all the image files (`.jpg`, `.jpeg`, `.png`, `.bmp` are supported) in your datapath (eg. `./dataset/CelebA`), and run the commad (512 pix for example):<br />
&emsp;&emsp;`python make_dataset.py --dataroot ./dataset/CelebA --extract_size 512 --output_img_dir ./dataset/CelebA/imgs --output_latent_dir ./dataset/CelebA/latents`<br /><br />
&emsp;&emsp;After processing, the `cropped face images` and ` net_Arc embedded latents` will be recored in the `output_img_dir` and `output_latent_dir` directories.<br /><br /><br /><br />

## 3.Start Training
### （1）Finetuning
&emsp;&emsp;Run command:
`CUDA_VISIBLE_DEVICES=0 python train.py --name CelebA_512_finetune --dataroot ./dataset/CelebA --image_size 512 --display_winsize 512 --continue_train`<br /><br />
&emsp;&emsp; NOTICE: If chekpoints/`name` is an un-existed folder, it will first copy the official model from chekpoints/people to chekpoints/`name`; then finetuning.<br /><br />

### （2）New Training
&emsp;&emsp;Run command:
`CUDA_VISIBLE_DEVICES=0 python train.py --name CelebA_512 --dataroot ./dataset/CelebA --image_size 512 --display_winsize 512`<br /><br />
&emsp;&emsp;<br />

&emsp;&emsp;When training is finished, training-process visualization, loss log-files and model weights will be stored in chekpoints/`name` folder.<br /><br /><br /><br />


## 4.Training Result
### （1）CelebA with 224x224 res
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/train_celeba_224.png)

### （2）CelebA with 512x512 res
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/train_celeba_512_1.png)
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/train_celeba_512_2.png)


## 5.Inference
&emsp;&emsp;I applied spNorm to the high-resolution image during training, which is conducive to the the model learning. Therefore, during Inference you need to modify<br />
&emsp;&emsp;`swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]`<br />
&emsp;&emsp;to <br />
&emsp;&emsp;`swap_result = swap_model(None, spNorm(frame_align_crop_tenor), id_vetor, None, True)[0]` <br />

# Apply example
&emsp;&emsp;The demo presented below uses modified architecture and with many training optimizations, and I'm sorry I cannot share that for the business issues.<br />
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/apply_example.jpg)
&emsp;&emsp;Watch video here:<br />
&emsp;&emsp;Video file here: ```docs/apply_example.mp4```<br /><br />


