import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps

# 38-class labels for PlantVillage dataset
DISEASE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

# Standard normalization for torchvision inference
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _map_hf_to_torchvision(hf_state_dict: dict) -> dict:
    """Translates a Hugging Face ViT state dict to torchvision ViT format on the fly."""
    tv_state_dict = {}
    
    # 1. Patch embeddings (projection conv)
    tv_state_dict['conv_proj.weight'] = hf_state_dict['vit.embeddings.patch_embeddings.projection.weight']
    tv_state_dict['conv_proj.bias'] = hf_state_dict['vit.embeddings.patch_embeddings.projection.bias']
    
    # 2. Class token and position embeddings
    tv_state_dict['class_token'] = hf_state_dict['vit.embeddings.cls_token']
    tv_state_dict['encoder.pos_embedding'] = hf_state_dict['vit.embeddings.position_embeddings']
    
    # 3. Encoder layers
    for i in range(12):
        hf_prefix = f'vit.encoder.layer.{i}.'
        tv_prefix = f'encoder.layers.encoder_layer_{i}.'
        
        # Self-attention query, key, value concatenation
        q_w = hf_state_dict[hf_prefix + 'attention.attention.query.weight']
        k_w = hf_state_dict[hf_prefix + 'attention.attention.key.weight']
        v_w = hf_state_dict[hf_prefix + 'attention.attention.value.weight']
        tv_state_dict[tv_prefix + 'self_attention.in_proj_weight'] = torch.cat([q_w, k_w, v_w], dim=0)
        
        q_b = hf_state_dict[hf_prefix + 'attention.attention.query.bias']
        k_b = hf_state_dict[hf_prefix + 'attention.attention.key.bias']
        v_b = hf_state_dict[hf_prefix + 'attention.attention.value.bias']
        tv_state_dict[tv_prefix + 'self_attention.in_proj_bias'] = torch.cat([q_b, k_b, v_b], dim=0)
        
        # Self-attention output dense layer
        tv_state_dict[tv_prefix + 'self_attention.out_proj.weight'] = hf_state_dict[hf_prefix + 'attention.output.dense.weight']
        tv_state_dict[tv_prefix + 'self_attention.out_proj.bias'] = hf_state_dict[hf_prefix + 'attention.output.dense.bias']
        
        # LayerNorms
        tv_state_dict[tv_prefix + 'ln_1.weight'] = hf_state_dict[hf_prefix + 'layernorm_before.weight']
        tv_state_dict[tv_prefix + 'ln_1.bias'] = hf_state_dict[hf_prefix + 'layernorm_before.bias']
        tv_state_dict[tv_prefix + 'ln_2.weight'] = hf_state_dict[hf_prefix + 'layernorm_after.weight']
        tv_state_dict[tv_prefix + 'ln_2.bias'] = hf_state_dict[hf_prefix + 'layernorm_after.bias']
        
        # MLP (dense 1 and dense 2)
        tv_state_dict[tv_prefix + 'mlp.linear_1.weight'] = hf_state_dict[hf_prefix + 'intermediate.dense.weight']
        tv_state_dict[tv_prefix + 'mlp.linear_1.bias'] = hf_state_dict[hf_prefix + 'intermediate.dense.bias']
        tv_state_dict[tv_prefix + 'mlp.linear_2.weight'] = hf_state_dict[hf_prefix + 'output.dense.weight']
        tv_state_dict[tv_prefix + 'mlp.linear_2.bias'] = hf_state_dict[hf_prefix + 'output.dense.bias']
        
    # 4. Final layernorm
    tv_state_dict['encoder.ln.weight'] = hf_state_dict['vit.layernorm.weight']
    tv_state_dict['encoder.ln.bias'] = hf_state_dict['vit.layernorm.bias']
    
    # 5. Classifier head
    tv_state_dict['heads.head.weight'] = hf_state_dict['classifier.weight']
    tv_state_dict['heads.head.bias'] = hf_state_dict['classifier.bias']
    
    return tv_state_dict

class DiseaseClassifier:
    """Loads a pre-trained PyTorch model (EfficientNet/ViT) and does top-3 inference."""
    def __init__(self, checkpoint_path: str):
        import gc
        import sys
        # Set single-thread CPU mode to minimize RAM overhead
        torch.set_num_threads(1)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the saved checkpoint directly
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model_name = checkpoint["model_name"]
        self.num_classes = checkpoint["num_classes"]
        self.class_names = checkpoint.get("class_names", DISEASE_CLASSES[:self.num_classes])
        
        # Instantiate a standard, unquantized float32 shell for the model
        float_model = self._create_empty_model(self.model_name, self.num_classes)
        
        # Check if model state dict is present directly (original unquantized weights)
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
            # If the state dict is in Hugging Face format, map it to torchvision format
            is_hf = any(k.startswith("vit.") for k in state_dict.keys())
            if is_hf:
                state_dict = _map_hf_to_torchvision(state_dict)
            float_model.load_state_dict(state_dict)
        elif "model" in checkpoint:
            # Extract the saved (quantized) model object from the checkpoint
            q_model = checkpoint["model"]
            
            # Recursively copy and dequantize weights/biases from q_model to float_model
            q_modules = dict(q_model.named_modules())
            f_modules = dict(float_model.named_modules())
            
            for name, f_mod in f_modules.items():
                if name not in q_modules:
                    continue
                q_mod = q_modules[name]
                
                # If it's a quantized linear module
                if type(q_mod).__name__ == "Linear" and "quantized" in type(q_mod).__module__:
                    w_float = q_mod.weight().dequantize()
                    b_float = q_mod.bias()
                    
                    f_mod.weight.data.copy_(w_float)
                    if b_float is not None:
                        f_mod.bias.data.copy_(b_float)
                # If it has standard weight and bias tensors
                elif hasattr(f_mod, "weight") and hasattr(q_mod, "weight"):
                    if f_mod.weight is not None and q_mod.weight is not None:
                        w = q_mod.weight
                        if isinstance(w, torch.Tensor):
                            w_data = w.dequantize() if w.is_quantized else w.data
                            f_mod.weight.data.copy_(w_data)
                    if hasattr(f_mod, "bias") and hasattr(q_mod, "bias"):
                        if f_mod.bias is not None and q_mod.bias is not None:
                            f_mod.bias.data.copy_(q_mod.bias.data)
        else:
            raise KeyError("Neither 'model_state' nor 'model' key found in the checkpoint.")
            
        self.model = float_model
        self.model.to(self.device)
        self.model.eval()
        
        # Clean up immediately
        del checkpoint
        if "q_model" in locals():
            del q_model
        del float_model
        gc.collect()

        # Force glibc to release cached memory back to the OS on Linux (Railway)
        if sys.platform.startswith("linux"):
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

    def _create_empty_model(self, model_name: str, num_classes: int) -> nn.Module:
        """Helper to instantiate the model architecture."""
        if model_name.startswith("efficientnet"):
            model = models.efficientnet_v2_s(weights=None)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
        elif model_name == "vit_b16" or model_name == "vit_huggingface":
            model = models.vit_b_16(weights=None)
            num_features = model.heads.head.in_features
            model.heads.head = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {model_name}")
        return model

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocesses raw image bytes into a PyTorch batch tensor."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = ImageOps.exif_transpose(image)
        tensor = _TRANSFORM(image)
        return tensor.unsqueeze(0).to(self.device)

    def predict_top_3(self, image_bytes: bytes):
        """Returns the top 3 predictions with disease names and probabilities."""
        tensor = self.preprocess(image_bytes)
        with torch.no_grad():
            outputs = self.model(tensor)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = torch.softmax(logits, dim=1)[0]
        
        # Sort and pick top 3
        top_probs, top_idxs = torch.topk(probs, k=3)
        
        results = []
        for prob, idx in zip(top_probs, top_idxs):
            disease_raw = self.class_names[idx.item()]
            # Simplify label name for English reading
            disease_en = disease_raw.split("___")[-1].replace("_", " ").strip()
            results.append({
                "disease_raw": disease_raw,
                "disease_name_en": disease_en,
                "confidence": round(prob.item(), 4)
            })
        return results
