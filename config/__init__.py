import yaml
from dataclass.classes import *

with open('config/config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

TRAIN_CONFIG = TrainConfig(
    batch_size=config['train_info']['batch_size'],
    epochs=config['train_info']['epochs'],
    learning_rate=config['train_info']['learning_rate'],
    dataloader=TrainDatasetConfig(
        train_path=config['train_info']['dataloader']['train_path'],
        valid_path=config['train_info']['dataloader']['valid_path'],
        target_size=config['train_info']['dataloader']['target_size'],
        shuffle=config['train_info']['dataloader']['shuffle']),
    results_path=config['train_info']['results_path']
    )

TEST_CONFIG = TestConfig(
    batch_size=config['test_info']['batch_size'],
    dataloader=TestDatasetConfig(
        test_path=config['test_info']['dataloader']['test_path'],
        target_size=config['test_info']['dataloader']['target_size'],
        shuffle=config['test_info']['dataloader']['shuffle'])
    )

MODEL_CONFIG = ModelConfig(
    model_name=config['model_info']['model_name'],
    pretrained=config['model_info']['pretrained'],
    weights=config['model_info']['weights'],
    include_top=config['model_info']['include_top'],
    num_classes=config['model_info']['num_classes'],
    input_shape=config['model_info']['input_shape'],
    save_path=config['model_info']['save_path']
    )


