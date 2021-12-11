# SimSwap-train
Reimplement of SimSwap training code<br />
这份代码原本是中秋节的时候写的；<br />
后来我们团队有换脸相关需求了，又做了很多优化，不过没法分享出来；<br />
我把这份代码的使用文档更新了一版，512pix是可以训的，希望能帮助到各位。<br /><br /><br />

# Instructions
## 1.Environment Preparation
### Step 1.Install Packages for python
&emsp;&emsp;Refer to the README document of [SIMSWAP preparation](https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md) to install the python packages and download the pretrained model;<br />
### Step 2.Modify the ```insightface``` package to support arbitrary-resolution training
（2）In order to support custom resolution, you need to modify two places in `/*your envs*/site-packages/insightface/utils/face_align.py`:<br />
&emsp;&emsp;line28: `src_all = np.array([src1, src2, src3, src4, src5])`<br />
&emsp;&emsp;line53: `src = src_all * image_size / 112`

## 2.Making Training Data
`python make_dataset.py --dataroot ./dataset/CelebA --extract_size 512 --output_img_dir ./dataset/CelebA/imgs --output_latent_dir ./dataset/CelebA/latents`<br /><br />
The face images and latents will be recored in the `output_img_dir` and `output_latent_dir` directories.

## 3.Start Training
### （1）New Training
`CUDA_VISIBLE_DEVICES=0 python train.py --name CelebA_512 --dataroot ./dataset/CelebA --image_size 512 --display_winsize 512`<br /><br />
Training visualization, loss log-files and model weights will be stored in chekpoints/`name` folder.

### （2）Finetuning
`CUDA_VISIBLE_DEVICES=0 python train.py --name CelebA_512_finetune --dataroot ./dataset/CelebA --image_size 512 --display_winsize 512 --continue_train`<br /><br />
If chekpoints/`name` is an un-existed folder, it will first copy the official model from chekpoints/people to chekpoints/`name`; then finetuning.

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
&emsp;&emsp;给大家分享一段我们组（Bytedance AI-Lab SA-TTS）做了大量优化之后的效果：<br />
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/apply_example.jpg)
&emsp;&emsp;Watch video here：```docs/apply_example.mp4```, or watch online:<br /><br />


