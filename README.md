# LaPDA: Latent-space Point Cloud Denoising with Adaptivity (TVCG).

> PyTorch implementation of the paper "LaPDA: Latent-space Point Cloud Denoising with Adaptivity".

## 📦 Installation

```
-conda create -n myenv python=3.9
-conda activate myenv
-conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
-conda install -c fvcore -c iopath -c conda-forge fvcore iopath
-conda install -c bottler nvidiacub
-conda install pytorch3d -c pytorch3d
-pip install "git+https://github.com/facebookresearch/pytorch3d.git"
-conda install pyg -c pyg
-pip install point-cloud-utils==0.29.6
-pip install plyfile
-pip install pandas
-pip install tensorboard
-pip install torchsummary
-conda install pytorch-cluster -c pyg
```

---

## 📂 Dataset Preparation
put your test data in the directory "./data"

---

## 🧪 Test

```bash
python test.py --niters=1 --dataset='PUNet' --resolution='10000_poisson' --noise='0.01';
python test.py --niters=1 --dataset='PUNet' --resolution='10000_poisson' --noise='0.02';
python test.py --niters=2 --dataset='PUNet' --resolution='10000_poisson' --noise='0.03';
```

---


## 📎 Acknowledgement and citation
Our code is partially based on *"Score-Based Point Cloud Denoising"* and *"StraightPCF: Straight Point Cloud Filtering"*.
If you find our paper interesting and our code useful, please cite our paper with the following BibTex citation:
```bibtex
@ARTICLE{LaPDA2025,
  author={Du, Peng and Wang, Xingce and Wu, Zhongke and Ru, Xudong and Granier, Xavier and He, Ying},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={LaPDA: Latent-Space Point Cloud Denoising With Adaptivity}, 
  year={2025},
  pages={1-15},
  doi={10.1109/TVCG.2025.3635138}
}

```
