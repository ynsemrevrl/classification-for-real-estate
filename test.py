from utils.data_loader import load_data
from utils.model_ops import load_model
from config import MODEL_CONFIG
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import glob
import os

def main():

    # Load dataset
    _,_,test_data = load_data()

    # Load model with Transfer Learning
    model = load_model(
        model_name=MODEL_CONFIG.model_name,
        pretrained=MODEL_CONFIG.pretrained,
        weights=MODEL_CONFIG.weights,
        include_top=MODEL_CONFIG.include_top,
        num_classes=MODEL_CONFIG.num_classes,
        input_shape=MODEL_CONFIG.input_shape
    )

    main_path = os.path.join(MODEL_CONFIG.save_path, MODEL_CONFIG.model_name)
    model_weights = glob.glob(main_path + "/*.keras")
    model_weights.sort()
    
    # Load best model weights
    model.load_weights(model_weights[-1])

    # Evaluate model
    scores = model.evaluate(test_data)
    print(f"Test loss: {scores[0]}")
    print(f"Test accuracy: {scores[1]}")

    # F1 score, precision, recall
    y_pred = model.predict(test_data)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = test_data.classes
    print(f"F1 score: {f1_score(y_true, y_pred, average='macro')}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro')}")
    print(f"Recall: {recall_score(y_true, y_pred, average='macro')}")
    print(f"Confusion matrix: {confusion_matrix(y_true, y_pred)}")

if __name__ == "__main__":
    main()
