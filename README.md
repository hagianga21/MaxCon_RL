# Demo -- Unsupervised Learning for Robust Fitting
This is demo of the following CVPR 2021 paper:
Unsupervised Learning for Robust Fitting: A Reinforcement Learning Approach

Paper can be accessed at: https://arxiv.org/abs/2103.03501

![](Data/results/demo.gif)
# Pre-requisite
```
- Pytorch
- Torch geometric
- cvxpy: pip install cvxpy
```
# Usage
- Run the inference using pre-trained model:
```
python inference.py
```
- Download minimal data and train the model
```
python downloadData.py
python train.py
```
# Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{MaxCon_RL,
  title={Unsupervised Learning for Robust Fitting: A ReinforcementLearning Approach},
  author={Giang Truong and Huu Le and David Suter and Erchuan Zhang and Syed Zulqarnain Gilani},
  booktitle={CVPR 2021},
  year={2021}
}
```
