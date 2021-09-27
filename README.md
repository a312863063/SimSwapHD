# SimSwap-train
Reimplement of SimSwap training code<br />

## Instructions
### 1.Environment Preparation
（1）Refer to the README document of [SIMSWAP](https://github.com/neuralchen/SimSwap) to configure the environment and download the pretrained model;<br />
（2）In order to support custom resolution, you need to modify two places in `/*your envs*/site-packages/insightface/utils/face_align.py`:<br />
&emsp;&emsp;line28: `src_all = np.array([src1, src2, src3, src4, src5])`<br />
&emsp;&emsp;line53: `src = src_all * image_size / 112`

### 2.Making Training Data
`python make_dataset.py --dataroot ./dataset/CelebA --extract_size 512 --output_img_dir ./dataset/CelebA/imgs --output_latent_dir ./dataset/CelebA/latents`<br /><br />
The face images and latents will be recored in the `output_img_dir` and `output_latent_dir` directories.

### 3.Start Training
#### （1）New Training
`CUDA_VISIBLE_DEVICES=0 python train.py --name CelebA_512 --dataroot ./dataset/CelebA --image_size 512 --display_winsize 512`<br /><br />
Training visualization, loss log-files and model weights will be stored in chekpoints/`name` folder.

#### （2）Finetuning
`CUDA_VISIBLE_DEVICES=0 python train.py --name CelebA_512_finetune --dataroot ./dataset/CelebA --image_size 512 --display_winsize 512 --continue_train`<br /><br />
If chekpoints/`name` is an un-existed folder, it will first copy the official model from chekpoints/people to chekpoints/`name`; then finetuning.

### 4.Training Result
#### （1）CelebA with 224x224 res
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/train_celeba_224.png)

#### （2）CelebA with 512x512 res
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/train_celeba_512_1.png)
![Image text](https://github.com/a312863063/SimSwap-train/blob/main/docs/img/train_celeba_512_2.png)
