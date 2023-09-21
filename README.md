# LE2usion

This is the code of the paper titled as "LE2Fusion: A novel local edge enhancement module for infrared and visible image fusion". 

The paper is accepted by ICIG2023 and can be found above.

# Framework

![framework](data_loader/framework.png)

# Environment

- Python 3.9.13
- torch 1.12.1
- torchvision 0.13.1
- tqdm 4.64.1

# To Train

We train our network using [MSRS](https://github.com/Linfeng-Tang/MSRS).

You can run the following prompt:

```python
python train.py
```

# To Test

Put your image pairs in the "test_data" directory and run the following prompt: 

```python
python test.py
```

# Models

The model for our network is "fusion_model_epoch_4.pth".

# Acknowledgement

- For calculating the image quality assessments, please refer to this [Metric](https://github.com/Linfeng-Tang/Evaluation-for-Image-Fusion).

# Contact Informaiton

If you have any questions, please contact me at <yongbiao_xiao_jnu@163.com>.

# Citation

If this work is helpful to you, please cite it as (BibTeX):

```
@article{xiao2023le2fusion,
  title={LE2Fusion: A novel local edge enhancement module for infrared and visible image fusion},
  author={Xiao, Yongbiao and Li, Hui and Cheng, Chunyang and Song, Xiaoning},
  journal={arXiv preprint arXiv:2305.17374},
  year={2023}
}
```

