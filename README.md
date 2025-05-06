# Multi_DDPP
In this work, we firstly trained a big dataset including differet cell assays such as Caco-2, RRCK, MDCK, PAMPA. Then we transferred the knowledge from the big dataset into our specific-task and high-fidelity dataset, which comprehensively improve our model.
![图片](https://github.com/user-attachments/assets/4c3a1859-a053-47d8-bfcb-71033336f58c)
# Overview
* feature/ : construction of molecular graph<br>
* data/: a big dataset including Caco-2, RRCK,MDCK,PAMPA and a high-fidelity PAMPA dataset for macrocycles<br>
* model/ :the pre-trained model on the big dataset and the model based on knowledge distillation<br>
# Requirement
  * python 3.8<br>
  * pytorch 2.3.0<br>
  * rdkit 2021.09.2<br>
  * scikit-learn 1.3.2<br>
  * dgl 2.2.1<br>
	# Reference
