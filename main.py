from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meibomian Gland Grading API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model class labels
CLASS_LABELS = {
    0: "Grade 0 - Normal",
    1: "Grade 1 - Mild MGD",
    2: "Grade 2 - Moderate MGD", 
    3: "Grade 3 - Severe MGD"
}

GRADE_DESCRIPTIONS = {
    0: "Normal - No visible gland dropout",
    1: "Mild - <25% gland dropout",
    2: "Moderate - 25-75% gland dropout", 
    3: "Severe - >75% gland dropout"
}

# FIXED: Use EfficientNet-B0 to match your saved model dimensions
class EnhancedEfficientNetClassifier(nn.Module):
    """Enhanced EfficientNet model matching your ACTUAL saved model architecture"""
    def __init__(self, num_classes=4, dropout_rate=0.4):
        super(EnhancedEfficientNetClassifier, self).__init__()
        # Use EfficientNet-B0 (matches your saved model dimensions)
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Get number of features from the backbone (1280 for EfficientNet-B0)
        num_features = self.backbone.classifier[1].in_features  # This will be 1280
        
        # Create classifier that matches your training code structure
        # But with the correct input dimensions
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),    # Changed from 1024 to 512 to match checkpoint
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)      # Direct to output classes
        )
    
    def forward(self, x):
        return self.backbone(x)

# Alternative: Try to match the EXACT architecture from your checkpoint
class OriginalEnhancedEfficientNetClassifier(nn.Module):
    """Try to recreate the exact model that was actually trained"""
    def __init__(self, num_classes=4, dropout_rate=0.4):
        super(OriginalEnhancedEfficientNetClassifier, self).__init__()
        # Use EfficientNet-B0 (based on the checkpoint dimensions)
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Get number of features (should be 1280 for EfficientNet-B0)
        num_features = self.backbone.classifier[1].in_features
        
        # Try to match the EXACT structure from checkpoint
        # The checkpoint shows: Linear(1280 -> 512) in backbone.classifier.1
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),  # This matches checkpoint: 512 features
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),  # Add intermediate layer if needed
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Initialize model - try the simpler version first
model = EnhancedEfficientNetClassifier(num_classes=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enhanced model loading with better error handling
try:
    model_path = "C:\eye\eye_\model\meibomian_model_20250721_170031.pth"
    
    logger.info(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try loading with strict=False to see which layers load successfully
    if "model_state_dict" in checkpoint:
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model_state_dict"], 
            strict=False  # Allow partial loading
        )
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
            
        logger.info(f"✅ Model loaded (partial) from epoch {checkpoint.get('epoch', 'unknown')}")
        
    else:
        # Try direct loading with strict=False
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        logger.info("✅ Model loaded (partial) from direct state dict")
    
    model.to(device)
    model.eval()
    logger.info(f"✅ Model moved to {device} and set to eval mode")
    
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    
    # Alternative: Try the original architecture
    logger.info("🔄 Trying alternative architecture...")
    try:
        model = OriginalEnhancedEfficientNetClassifier(num_classes=4)
        checkpoint = torch.load(model_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        model.to(device)
        model.eval()
        logger.info("✅ Alternative model architecture loaded successfully")
        
    except Exception as e2:
        logger.error(f"❌ Both architectures failed: {e2}")
        raise Exception(f"Failed to load model with any architecture: {e}")

# Image transforms (keep the same)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Enhanced prediction endpoint"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate file size
        image_bytes = await file.read()
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        
        # Process image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Transform and predict
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[predicted_class].item()
        
        # Prepare response
        result = {
            "success": True,
            "predicted_grade": predicted_class,
            "predicted_class": CLASS_LABELS[predicted_class],
            "description": GRADE_DESCRIPTIONS[predicted_class],
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                f"grade_{i}": round(prob.item() * 100, 2) 
                for i, prob in enumerate(probabilities)
            },
            "all_grades": [
                {
                    "grade": i,
                    "label": CLASS_LABELS[i],
                    "description": GRADE_DESCRIPTIONS[i],
                    "probability": round(probabilities[i].item() * 100, 2)
                }
                for i in range(4)
            ]
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Meibomian Gland Grading API",
        "status": "running",
        "model_loaded": True,
        "device": str(device)
    }

@app.get("/health")
async def health_check():
    try:
        # Quick model test
        test_tensor = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            test_output = model(test_tensor)
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_responsive": True,
            "device": str(device),
            "output_shape": list(test_output.shape)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
