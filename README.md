Diabetic Retinopathy detection and severity classification using Deep Learning

Models Weights Download Link:
https://drive.google.com/drive/folders/1xsZurnRnyQKNypgNhtGO6EUv6Bqp3j3d?usp=sharing
Dataset Download Link:
https://drive.google.com/drive/folders/1dpCSU65wjcUyFZQTasStmDocvAhFH27Y?usp=sharing

We created an ensemble learning model to detect and classify Diabetic Retinopathy
Dataset source was Kaggle DR Comptetition 
We balanced the dataset according to the minimal class. 

Classes are: \
0: No DR\
1: Mild\
2: Moderate\
3: Severe\
4: Proliferative

The trained models are:\
InceptionResNet-v2\
SWIN Transformers\
Modified VGG16 with attention\
Basic CNN

Each model contributes to the prediction with a specified weight according to its validation accuracy and its degree of certainty.

Sample Output:\
![image](https://user-images.githubusercontent.com/101192969/210878371-31193ba3-e7da-407d-b910-e48bea84659a.png)

