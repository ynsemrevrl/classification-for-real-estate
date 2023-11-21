from utils.model_ops import load_model, save_model, save_history
from config import MODEL_CONFIG, TRAIN_CONFIG
from utils.data_loader import load_data
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from loguru import logger

def main():

    # Load dataset
    train_data, valid_data, _ = load_data()

    # Load model with Transfer Learning
    model = load_model(
        model_name=MODEL_CONFIG.model_name,
        pretrained=MODEL_CONFIG.pretrained,
        weights=MODEL_CONFIG.weights,
        include_top=MODEL_CONFIG.include_top,
        num_classes=MODEL_CONFIG.num_classes,
        input_shape=MODEL_CONFIG.input_shape
    )

    # Train model    
    main_path = os.path.join(MODEL_CONFIG.save_path, MODEL_CONFIG.model_name)
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_filename = os.path.join(main_path,"weights_{epoch:04d}.keras")
    checkpoint = ModelCheckpoint(model_filename, monitor = "val_loss", save_best_only=True)
    callbacks_list = [checkpoint]
    
    if tf.test.is_gpu_available(cuda_only=True):
        print("Training on GPU")
        device = tf.test.gpu_device_name()
    else:
        device = '/CPU:0'

    logger.info(f"Training started in {device}")
    with tf.device(device):
        hist = model.fit(
            train_data,
            steps_per_epoch=train_data.samples // TRAIN_CONFIG.batch_size,
            epochs=TRAIN_CONFIG.epochs,
            validation_data=valid_data,
            validation_steps=valid_data.samples // TRAIN_CONFIG.batch_size,
            callbacks=callbacks_list
        )

    # Save last model and history
    save_model(model, main_path)
    save_history(hist, main_path)

if __name__ == "__main__":
    main()