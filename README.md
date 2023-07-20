# MMNR

This repository contains Pytorch implementation of [MMNR](https://dl.acm.org/doi/10.1145/3539618.3591738):

> Multi-view Multi-aspect Neural Networks for Next-basket Recommendation.
> Zhiying Deng, Jianjun Li, Zhiqiang Guo, Wei Liu, Li Zou, Guohui Li.
> The 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2023).

MMNR employs normalization to balance the number of interactions from user and item views, ensuring adequate representation and eliminating the bias caused by differences in interactions. Additionally, MMNR considers the fine-grained context of the item, enabling the modeling of its diverse features across multiple aspects and facilitating comprehensive analysis.
## Environments

- torch 1.10.1+cuda 11.2
- python 3.6.13
- numpy 1.19.5
- scipy 1.5.4
- scikit-learn 0.23.2

## Running the code
create folder ./src/his/seq/his_test/TaFeng
```python
$ cd src
$ python main.py --dataset TaFeng --lr 0.01 --l2 0.001 --asp 11 --ctx 3 --decay 0.6 --h1 5 --h2 5 --batch_size 100 --dim 32 --isTrain 0
$ python main.py --dataset Dunnhumby --lr 0.001 --l2 0.01 --asp 14 --ctx 3 --decay 0.6 --h1 5 --h2 5 --batch_size 100 --dim 32 --isTrain 0
$ python main.py --dataset ValuedShopper --lr 0.01 l2 0.01 --asp 13 --ctx 3 --decay 0.6 --h1 5 --h2 5 --batch_size 100 --dim 32 --isTrain 0
$ python main.py --dataset RetailRocket --lr 0.01 l2 0.01 --asp 15 --ctx 3 --decay 0.6 --h1 5 --h2 5 --batch_size 100 --dim 32 --isTrain 0
```

## Citation

```
@inproceedings{DBLP:conf/sigir/DengLGLZL23,
  author       = {Zhiying Deng and
                  Jianjun Li and
                  Zhiqiang Guo and
                  Wei Liu and
                  Li Zou and
                  Guohui Li},
  title        = {Multi-view Multi-aspect Neural Networks for Next-basket Recommendation},
  booktitle    = {Proceedings of the 46th International {ACM} {SIGIR} Conference on
                  Research and Development in Information Retrieval, {SIGIR} 2023, Taipei,
                  Taiwan, July 23-27, 2023},
  pages        = {1283--1292},
  year         = {2023},
}
```

