## Enhancing Multi-Label Classifcation via Dynamic Label-Order Learning

Created by Jiangnan Li, Yice Zhang, Shiwei Chen, and Ruifeng Xu, Harbin Insitute of Technology, Shenzhen.

This repository contains the official PyTorch implementation of our AAAI 2024 paper [Enhancing Multi-Label Classifcation via Dynamic Label-Order Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29814). Our code is modified from [Text2Event](https://github.com/luyaojie/text2event).

### Environment Setup and Data Preprocessing
General

- Python (verified on 3.10)
- CUDA (verified on 11.7)

Python Packages

- see requirements.txt

```bash
conda create -n DLOL python=3.10
conda activate DLOL
pip install -r requirements.txt
```

### Training and Evaluation
Traing
```text
bash run_reuters.sh
```
Evaluation
```text
bash run_eval.bash
```
### Matters Need Attention
Currently, our code only supports training and evaluation on the Reuters dataset. We will reproduce our results on a new environment and update our GitHub repo.

Our code differs from the original paper in some ways. We reimplement our loss of multiple orders based on [Set Learning](https://github.com/KagamiBaka/Set-Learning) to avoid hyperparameter tuning, making our code easier to use.

### Citation
If you find this code helpful for your research, please consider citing
```text
@inproceedings{li2024enhancing,
  title={Enhancing Multi-Label Classification via Dynamic Label-Order Learning},
  author={Li, Jiangnan and Zhang, Yice and Chen, Shiwei and Xu, Ruifeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={17},
  pages={18527--18535},
  year={2024}
}
```