# Kaggle-SIIM-ISIC-Melanoma-Classification

[Link to competition](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview)

## Libraries used:

- [Albumentations](https://github.com/albu/albumentations)
- [Pytorch Image Models](https://github.com/rwightman/pytorch-image-models)
- Numpy
- Open-CV
- Pytorch
- SciKit-Learn

Regarding the libraries, all rights belong to their respective owners.

## Score

Best Private Score achieved: 

2x EfficientNet B0: 0.9200 AUC with 512x512 imagesize
No Metadata, TTA and Pseudolabelling was used,
~Top 58% Rank

*TPU code has some issues where model trained with TPU, loaded on CPU for inference gave really bad public LB scores but decent CV score.
