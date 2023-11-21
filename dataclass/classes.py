from dataclasses import dataclass

@dataclass
class TrainDatasetConfig:
    train_path: str
    valid_path: str
    target_size: tuple
    shuffle: bool

@dataclass
class TestDatasetConfig:
    test_path: str
    target_size: tuple
    shuffle: bool

@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    dataloader: TrainDatasetConfig
    results_path: str

@dataclass
class TestConfig:
    batch_size: int
    dataloader: TestDatasetConfig

@dataclass
class ModelConfig:
    model_name: str
    pretrained: bool
    weights: str
    include_top: bool
    num_classes: int
    input_shape: tuple
    save_path: str

