from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import BatchNormalization, Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D, multiply, LocallyConnected2D, Lambda
from tensorflow.keras.optimizers import SGD, Adam
from keras.applications.vgg16 import VGG16
from keras.metrics import top_k_categorical_accuracy
import cv2
import numpy as np
import transformers
from transformers import SwinForImageClassification, SwinConfig
import torch
from torchvision.transforms import (
    Compose, 
    Normalize, 
    Resize, 
    ToTensor
)
# CLASSES=['No DR', 'Mild','Moderate','Severe','Proliferative']
# id2label = {id:label for id, label in enumerate(CLASSES)}
# label2id = {label:id for id,label in id2label.items()}

class DREnsembleDetector():
    def __init__(
        self,
        inception_resnet_weights,
        modified_vgg16_weights,
        cnn_weights,
        transformers_weights
        ):

        self.inception_resnet=InceptionResNetV2(
            input_shape = (299,299,3),
            weights =None,
            include_top = True,
            pooling='max',
            classes=5
            )
        self.inception_resnet.load_weights(inception_resnet_weights)

        self.cnn= self._create_cnn(cnn_weights)
        self.modvgg16= self._create_modified_vgg16(modified_vgg16_weights)
        self.transformers= torch.load(transformers_weights,map_location=torch.device('cpu'))

    # def _create_transformer():
      # config = SwinConfig.from_pretrained(
      #   "microsoft/swin-tiny-patch4-window7-224",
      #   num_labels=len(label2id),
      #   label2id=label2id,
      #   id2label=id2label,
      #   finetuning_task="image-classification"
      # )

      # model = SwinForImageClassification.from_pretrained(
      #   "microsoft/swin-tiny-patch4-window7-224",
      #   config=config,
      #   ignore_mismatched_sizes=True)

    #basic cnn
    def _create_cnn(self,weights):
        EPOCHS = 30
        INIT_LR = 1e-3
        model = Sequential()
        # first set of CONV => RELU => MAX POOL layers
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(299,299,3)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='softmax'))
        # returns our fully constructed deep learning + Keras image classifier 
        opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
        # use binary_crossentropy if there are two classes
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        model.load_weights(weights)
        return model

    def _create_modified_vgg16(self,weights):
        in_lay = Input((256,256,3))
        base_pretrained_model = VGG16(input_shape =  (256,256,3), include_top = False, weights = None)
        pt_features = base_pretrained_model(in_lay)
        pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
        pt_features = base_pretrained_model(in_lay)
        bn_features = BatchNormalization()(pt_features)
        # here we do an attention mechanism to turn pixels in the GAP on an off
        attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(bn_features))
        attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
        attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
        attn_layer = Conv2D(1, kernel_size = (1,1), padding = 'valid', activation = 'sigmoid')(attn_layer)
        # fan it out to all of the channels
        up_c2_w = np.ones((1, 1, 1, pt_depth))
        up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
                    activation = 'linear', use_bias = False, weights = [up_c2_w])
        attn_layer = up_c2(attn_layer)
        mask_features = multiply([attn_layer, bn_features])
        gap_features = GlobalAveragePooling2D()(mask_features)
        gap_mask = GlobalAveragePooling2D()(attn_layer)
        # to account for missing values from the attention model
        gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
        gap_dr = Dropout(0.25)(gap)
        dr_steps = Dropout(0.25)(Dense(128, activation = 'relu')(gap_dr))
        out_layer = Dense(5, activation = 'softmax')(dr_steps)
        retina_model = Model(inputs = [in_lay], outputs = [out_layer])
        def top_2_accuracy(in_gt, in_pred):
            return top_k_categorical_accuracy(in_gt, in_pred, k=2)
        retina_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                                metrics = ['categorical_accuracy', top_2_accuracy])
        retina_model.load_weights(weights)
        return retina_model

    def inception_resnet_preprocessing(self,image):
        image=np.asarray([cv2.resize(image,(299,299))])
        return image
    def cnn_preprocessing(self,image):
        image=np.asarray([cv2.resize(image,(299,299))])
        return image
    def modvgg16_preprocessing(self,image):
        image=np.asarray([cv2.resize(image,(256,256))])
        return image
    def transformers_preprocessing(self,image):
        _val_transforms = Compose([
                  ToTensor(),
                  Resize([224,224]),
                  Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),#from imagenet
                  ])
        image=_val_transforms(image)
        image=torch.unsqueeze(image,dim=0)
        return image

    def predict(self, image):
        inception_resnet_image=self.inception_resnet_preprocessing(image)
        inception_resnet_prediction= self.inception_resnet.predict(inception_resnet_image,verbose=0)
        modvgg16_image= self.modvgg16_preprocessing(image)
        modvgg16_prediction= self.modvgg16.predict(modvgg16_image,verbose=0)
        cnn_image= self.cnn_preprocessing(image)
        cnn_prediction= self.cnn.predict(cnn_image,verbose=0)
        transformers_image= self.transformers_preprocessing(image)
        transformers_prediction= self.transformers(transformers_image).logits.detach()
        transformers_prediction=torch.nn.Softmax(1)(transformers_prediction)
        transformers_prediction=np.asarray(transformers_prediction)
        prediction= (inception_resnet_prediction+modvgg16_prediction+cnn_prediction+transformers_prediction)/4
        return prediction

