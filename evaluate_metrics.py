import os
import json

def display_metrics():
    metrics_path = r"d:\AI Image detector\metrics.json"

    if not os.path.exists(metrics_path):
        print("\n[!] Error: metrics.json not found. Please train the model first by running: python train_model.py\n")
        return

    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model_type = data.get('model_type', 'custom').upper()
    acc = data.get('accuracy', 0.0) * 100.0
    prec = data.get('precision', 0.0) * 100.0
    rec = data.get('recall', 0.0) * 100.0
    f1 = data.get('f1_score', 0.0) * 100.0
    roc_auc = data.get('roc_auc', 0.0)
    best_loss = data.get('best_val_loss', 0.0)
    test_count = data.get('test_count', 0)
    cm = data.get('confusion_matrix', {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0})
    history = data.get('history', {})

    print("\n" + "="*65)
    print(f"       AI IMAGE DETECTOR - CNN MODEL EVALUATION REPORT")
    print("="*65)
    print(f"  Model Architecture : {model_type} CNN")
    print(f"  Checkpoint File    : d:\\AI Image detector\\best_model.pth")
    print(f"  Test Sample Count  : {test_count} images (10% held-out test split)")
    print("-" * 65)

    print(f"  [+] PERFORMANCE METRICS:")
    print(f"      - Test Accuracy  : {acc:.2f}%")
    print(f"      - Precision      : {prec:.2f}%")
    print(f"      - Recall (Sens.) : {rec:.2f}%")
    print(f"      - F1-Score       : {f1:.2f}%")
    print(f"      - ROC-AUC Score  : {roc_auc:.4f}")
    print(f"      - Val Loss       : {best_loss:.4f}")
    print("-" * 65)

    print(f"  [+] CONFUSION MATRIX TABLE:")
    print(f"                        Predicted Real    Predicted AI")
    print(f"     Actual Real Photo    {cm['tn']:<16} {cm['fp']:<12}")
    print(f"     Actual AI Generated  {cm['fn']:<16} {cm['tp']:<12}")
    print("-" * 65)

    if 'train_loss' in history and 'val_loss' in history:
        print(f"  [+] TRAINING EPOCH HISTORY:")
        print(f"     Epoch | Train Loss | Val Loss | Train Acc % | Val Acc %")
        print(f"     -----------------------------------------------------")
        for idx in range(len(history['train_loss'])):
            tr_l = history['train_loss'][idx]
            va_l = history['val_loss'][idx]
            tr_a = history['train_acc'][idx] * 100.0
            va_a = history['val_acc'][idx] * 100.0
            print(f"       {idx+1:02d}  |   {tr_l:.4f}   |  {va_l:.4f}  |   {tr_a:.2f}%   |  {va_a:.2f}%")
    
    print("="*65 + "\n")

if __name__ == '__main__':
    display_metrics()
