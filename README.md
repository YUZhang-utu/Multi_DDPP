# Multi_DDPP
In this work, we firstly trained a big dataset including differet cell assays but they represent similar task and diverse molecules. Then we transferred the knowledge from the big dataset into our specific-task and high-fidelity dataset, which comprehensively improve our model. And we constructed models with physiological parameters not just chemical information.
![图片](https://github.com/user-attachments/assets/4c3a1859-a053-47d8-bfcb-71033336f58c)
# Overview
* features : construction of molecular graph<br>
* data: a big dataset, a high-fidelity dataset for macrocycles and a dataset for regression models<br>
* Multi_DDPP :the pre-trained model on the big dataset and the model based on knowledge distillation<br>
* regression_model :regression models<br>
# Requirement
  * python 3.8<br>
  * pytorch 2.3.0<br>
  * rdkit 2021.09.2<br>
  * scikit-learn 1.3.2<br>
  * dgl 2.2.1<br>
# Reference
