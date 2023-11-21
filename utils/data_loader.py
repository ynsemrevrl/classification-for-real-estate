from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Literal
from config import TRAIN_CONFIG, TEST_CONFIG

class DataLoader():
    def __init__(self, kind:Literal["train", "valid", "test"], 
                 path, 
                 target_size, 
                 batch_size, 
                 shuffle=True):
        
        self.kind = kind
        self.path = path
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self):
        return self.get_data()
    
    def get_data(self):
        if self.kind == "train":
            self.train_data = ImageDataGenerator().flow_from_directory(
                self.path,
                target_size=self.target_size,
                batch_size=self.batch_size,
                shuffle=self.shuffle)
            return self.train_data
            
        elif self.kind == "valid":

            self.valid_data = ImageDataGenerator().flow_from_directory(
                self.path,
                target_size=self.target_size,
                batch_size=self.batch_size,
                shuffle=self.shuffle)
            return self.valid_data
        
        else: # test
            self.test_data = ImageDataGenerator().flow_from_directory(
                self.path,
                target_size=self.target_size,
                batch_size=self.batch_size,
                shuffle=self.shuffle)
            return self.test_data


def load_data():
    train_data = DataLoader(kind="train",
                            path=TRAIN_CONFIG.dataloader.train_path,
                            target_size=TRAIN_CONFIG.dataloader.target_size,
                            batch_size=TRAIN_CONFIG.batch_size,
                            shuffle=TRAIN_CONFIG.dataloader.shuffle)()
    
    valid_data = DataLoader(kind="valid",
                            path=TRAIN_CONFIG.dataloader.valid_path,
                            target_size=TRAIN_CONFIG.dataloader.target_size,
                            batch_size=TRAIN_CONFIG.batch_size,
                            shuffle=TRAIN_CONFIG.dataloader.shuffle)()
    
    test_data = DataLoader(kind="test",
                            path=TEST_CONFIG.dataloader.test_path,
                            target_size=TEST_CONFIG.dataloader.target_size,
                            batch_size=TEST_CONFIG.batch_size,
                            shuffle=TEST_CONFIG.dataloader.shuffle)()
    
    return train_data, valid_data, test_data