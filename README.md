# Hyperspectral Image Analysis using ML/DL

A comparative analysis project of machine learning methods for Indian Pines dataset classification.

Contains:
* 460 Fully Connected Neural Network (FNN) models
* 14 Support Vector Machine (SVM) models
* 6 Random Forest models

## Results

| Method         | Best Accuracy | Best F1-score
|---------------|--------------:|--------------:
| FNN           | 0.940         | 0.939         
| Random Forest | 0.862         | 0.860
| SVM           | 0.914         | 0.914

---

## Dataset: [Indian Pines Hyperspectral Dataset](https://www.kaggle.com/datasets/abhijeetgo/indian-pines-hyperspectral-dataset/data)

Preprocessing:
1. Removal of unlabeled pixels
2. Normalization

---

## Project Structure

> For convenient navigation in file explorer, sort by **modification date (ascending)**

* results/ 
    * FNN/
        * results.csv
        * default/
            * metrics.png
            * results.csv
            * train_val_loss/
                * .png
                * ...
                * ...
        * Lasso/
        * ...
        * ...
    * SVM/
        * metrics.png
        * results.csv
    * RF/
        * metrics.png
        * results.csv
* architectures/
    * models_default.txt
    * ...
    * ...
* code/
    * FNN.ipynb
    * kFold_cv.ipynb
    * Random_Forest.ipynb
    * SVM.ipynb
    * utils.py
    * classFNN.py
* requirements.txt
* README.md

---

### Trained models are available on [Yandex Disk](https://disk.yandex.ru/d/Ubh_uis85WRC3g) in the following format:

* runs/
    * FNN/
        * default/
            * [model architecture in '{l1}-{l2}-...-{l_n}' format, where {l_k} is number of neurons in k-th layer]/
                * args.yaml
                * model.pth
                * results.csv
                * training.csv
            * ...
            * ...
        * Lasso/ 
            * ...
        * Ridge/
            * ...
        * ...
        * ...
    * SVM/
        * SVM_.../
            * args.yaml
            * results.csv
        * ...
        * ...
    * RF/
        * RF_.../
            * args.yaml
            * results.csv
        * ...
        * ...