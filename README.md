Unofficial PyTorch implement of [Video cloze procedure for self-supervised spatio-temporal learning](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-LuoD.2527.pdf) [AAAI'20]

Codes are mainly based on [VCOP](https://github.com/xudejing/video-clip-order-prediction) [CVPR'19] 

### Requirements
> This is my experimental environment

PyTorch 1.3.0   
python 3.7.4

### Supported features
- Dataset: UCF101
- Tasks: Spatial rotation, temporal shuffling, spatial permutation, temporal remote shuffling, tempoal ajacent shuffling
- Modality: RGB, Res

### Scripts
#### Dataset preparation
You can follow [VCOP](https://github.com/xudejing/video-clip-order-prediction) [CVPR'19] to prepare dataset.

If you have decoded frames from videos, you can edit `framefolder = os.path.join('/path/to/your/frame/folders', videoname[:-4])` in `ucf101.py` and directly use our provided list.

#### Train self-supervised part
```python
python train_vcp.py
```

#### Retrieve video clips
```python
python retrieve_clips.py --ckpt=/path/to/self-supervised_model
```

#### Fine-tune models for video recognition
```python
python ft_classify --ckpt=/path/to/self-supervised_model
```
If you want to train models from scratch, use
```python
python train_classify mode=train
```

#### Test models for video recognition
```python
python train_classify --ckpt=/path/to/fine-tuned_model
```

### Results
#### Retrieval results
Tag | Modality | top1 | top5 | top10 | top20 | top50
---|---|---|---|---|---|---
R3D (VCP, paper) | RGB | 18.6 | 33.6 | 42.5| 53.5 | 68.1
R3D (VCP, reimplemented) | RGB | 24.2 | 41.2 | 50.3 | 60.2 | 74.8 
R3D (VCP, reimplemented) | Res | **26.3** | **44.8** | **55.0** | **65.4** | **78.7** 

#### Recognition results
Dataset | Tag | Modality | Acc
---|---|---|---
UCF101 | R3D (VCP, paper) | RGB | 68.1
UCF101 | R3D (VCP, reimplemented) | RGB | 67.4
UCF101 | R3D (VCP, reimplemented) | Res | **78.4**

**Residual clips + 3D CNN** The residual clips with 3D CNNs are effective. More information about this part can be found in [Rethinking Motion Representation: Residual Frames with 3D ConvNets for Better Action Recognition](https://arxiv.org/abs/2001.05661) (previous but more detailed version) and [Motion Representation Using Residual Frames with 3D CNN](https://arxiv.org/abs/2006.13017) (short version with better results).

The key code for this part is 
```
shift_x = torch.roll(x,1,2)
x = ((shift_x -x) + 1)/2
```
Which is slightly different from that in papers.

### Citation
VCP
```
@article{luo2020video,
  title={Video cloze procedure for self-supervised spatio-temporal learning},
  author={Luo, Dezhao and Liu, Chang and Zhou, Yu and Yang, Dongbao and Ma, Can and Ye, Qixiang and Wang, Weiping},
  journal={arXiv preprint arXiv:2001.00294},
  year={2020}
}
```
Residual clips + 3D CNN
```
@article{tao2020rethinking,
  title={Rethinking Motion Representation: Residual Frames with 3D ConvNets for Better Action Recognition},
  author={Tao, Li and Wang, Xueting and Yamasaki, Toshihiko},
  journal={arXiv preprint arXiv:2001.05661},
  year={2020}
}

@article{tao2020motion,
  title={Motion Representation Using Residual Frames with 3D CNN},
  author={Tao, Li and Wang, Xueting and Yamasaki, Toshihiko},
  journal={arXiv preprint arXiv:2006.13017},
  year={2020}
}
```
