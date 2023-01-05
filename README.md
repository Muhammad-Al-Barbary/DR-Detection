Diabetic Retinopathy detection and severity classification using Deep Learning

Download Models Weights from here:
https://drive.google.com/drive/folders/1xsZurnRnyQKNypgNhtGO6EUv6Bqp3j3d?usp=sharing

We created an ensemble learning model to detect and classify Diabetic Retinopathy
Dataset was balanced according to the minimal class. 

Classes are: 
0: No DR
1: Mild
2: Moderate
3: Severe
4: Proliferative

The trained models are:
1. InceptionResNet-v2
2. SWIN Transformers
3. Modified VGG16 with attention
4. Basic CNN

Each model contributes to the prediction with a specified weight according to its validation accuracy and its degree of certainty.

