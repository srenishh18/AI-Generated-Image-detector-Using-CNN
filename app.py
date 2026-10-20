import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static', template_folder='templates')

# ---------------------------------------------------------
# CNN Model Architecture Classes
# ---------------------------------------------------------
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze(1)

class EfficientNetB0Detector(nn.Module):
    def __init__(self):
        super(EfficientNetB0Detector, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x.squeeze(1)

# Global variables for model & device
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_trained_model():
    global MODEL
    model_path = r"d:\AI Image detector\best_model.pth"
    if not os.path.exists(model_path):
        print("Warning: best_model.pth not found yet.")
        return False
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model_type = checkpoint.get('model_type', 'custom')
        
        if model_type == 'efficientnet':
            MODEL = EfficientNetB0Detector()
        else:
            MODEL = CustomCNN()
            
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.to(DEVICE)
        MODEL.eval()
        print(f"Model successfully loaded! Type: {model_type}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global MODEL
    if MODEL is None:
        success = load_trained_model()
        if not success:
            return jsonify({'error': 'Model weights not trained yet. Please run model training first.'}), 400

    try:
        image_bytes = None
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            image = Image.open(file.stream).convert('RGB')
        elif 'image_path' in request.json:
            path = request.json['image_path']
            image = Image.open(path).convert('RGB')
        else:
            return jsonify({'error': 'No image file or path provided'}), 400

        # Preprocess & Predict
        input_tensor = PREPROCESS_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logit = MODEL(input_tensor)
            ai_prob = float(torch.sigmoid(logit).item())
        
        real_prob = 1.0 - ai_prob
        is_ai = ai_prob >= 0.5
        
        label = "AI-Generated Image" if is_ai else "Real Image"
        confidence = (ai_prob if is_ai else real_prob) * 100.0

        return jsonify({
            'success': True,
            'label': label,
            'is_ai': is_ai,
            'confidence': round(confidence, 2),
            'ai_probability': round(ai_prob * 100.0, 2),
            'real_probability': round(real_prob * 100.0, 2)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics_path = r"d:\AI Image detector\metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({'error': 'Metrics not found'}), 444

if __name__ == '__main__':
    load_trained_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
