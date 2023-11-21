from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from loguru import logger
import numpy as np
import os

class ResNet():
    def __init__(self, input_shape, num_classes, pretrained, weights, include_top):
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = Sequential()
        self.input_shape = input_shape
        self.weights = weights
        self.include_top = include_top

    def __call__(self):
        self.get_model()
        return self.model

    def get_model(self):
        resnet_model= ResNet50(
            include_top=self.include_top,
            input_shape=self.input_shape,
            pooling='avg',
            classes=self.num_classes,
            weights=self.weights)
        
        for layer in resnet_model.layers:
                layer.trainable=self.pretrained

        self.model.add(resnet_model)
        self.model.add(Dense(self.num_classes, activation = "softmax"))

        self.model.compile(loss = "categorical_crossentropy",
                    optimizer = "rmsprop",
                    metrics = ["accuracy"])
        
        return self.model
    
class Inception():
    def __init__(self, input_shape, num_classes, pretrained, weights, include_top):
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = Sequential()
        self.input_shape = input_shape
        self.weights = weights
        self.include_top = include_top

    def __call__(self):
        self.get_model()
        return self.model
    
    def get_model(self):

        inception_model=InceptionV3(
            include_top=self.include_top,
            weights=self.weights,
            input_shape=self.input_shape,
            pooling='avg')
        
        for layer in inception_model.layers:
            layer.trainable = self.pretrained
        
        self.model.add(inception_model)
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        
        self.model.compile(optimizer = Adam(), 
            loss = 'categorical_crossentropy', 
            metrics = ['acc'])
        
        return self.model
    
class DenseNet():
    def __init__(self, input_shape, num_classes, pretrained, weights, include_top):
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = Sequential()
        self.initializer = initializers.he_normal(seed=32)
        self.input_shape = input_shape
        self.weights = weights
        self.include_top = include_top

    def __call__(self):
        self.get_model()
        return self.model
    
    def get_model(self):
        densenet_model = DenseNet169(
            include_top=self.include_top,
            weights=self.weights,
            input_tensor=None,
            input_shape=self.input_shape,
            pooling=None)
        
        densenet_model.trainable = self.pretrained
        for layer in densenet_model.layers:
            if 'conv5' in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False

        self.model.add(densenet_model)
        self.model.add(Flatten())
        self.model.add(Dense(units=256, activation='relu', kernel_initializer=self.initializer))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(units=128, activation='relu', kernel_initializer=self.initializer))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(units=self.num_classes, activation='softmax', kernel_initializer=self.initializer))

        self.model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])
        
        return self.model

def load_model(model_name, pretrained, weights, include_top, num_classes, input_shape):
    if model_name == "resnet":
        model = ResNet(input_shape, num_classes, pretrained, weights, include_top)
        logger.info("ResNet model is loaded.")
    elif model_name == "inception":
        model = Inception(input_shape, num_classes, pretrained, weights, include_top)
        logger.info("Inception model is loaded.")
    elif model_name == "densenet":
        model = DenseNet(input_shape, num_classes, pretrained, weights, include_top)
        logger.info("DenseNet model is loaded.")
    else:
        raise ValueError("Invalid model name")
    return model()

def save_model(model, path):
    save_path = os.path.join(path,"LAST_MODEL")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save(os.path.join(save_path,"last_model.keras"))
    model.save_weights(os.path.join(save_path,"last_weights.keras"))
    logger.info("Model and weights are saved.")

def save_history(history, path):
    save_path = os.path.join(path,"LAST_MODEL")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path,"last_history.npy"),history.history)
    logger.info("History is saved.")
