import io
import base64
import torch
import numpy as np
from PIL import Image, ImageOps

class GradCAM:
    """Computes Grad-CAM activations and overlays them on the input image using pure PyTorch/PIL."""
    def __init__(self, classifier):
        self.classifier = classifier
        self.model = classifier.model
        self.device = classifier.device
        
        # Determine the target layer to hook based on architecture
        if classifier.model_name.startswith("efficientnet"):
            self.target_layer = self.model.features
        elif classifier.model_name in ["vit_b16", "vit_huggingface"]:
            self.target_layer = self.model.encoder.layers[-1]
        else:
            self.target_layer = None
            
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture forward activations and backward gradients."""
        if not self.target_layer:
            return

        def forward_hook(module, input, output):
            # HuggingFace attention hook outputs a tuple: (last_hidden_state, None) or a tensor.
            # We want the tensor activations!
            self.activations = output[0] if isinstance(output, tuple) else output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap_overlay(self, image_bytes: bytes, target_class_idx: int) -> tuple[str, dict]:
        """Generates a Grad-CAM heatmap overlay and returns the base64 string + explainable regions."""
        # 1. Forward and backward passes to capture activations/gradients
        tensor = self.classifier.preprocess(image_bytes)
        self.model.zero_grad()
        
        outputs = self.model(tensor)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        score = logits[0, target_class_idx]
        score.backward()

        # Check if hooks captured features successfully (avoid failure if model doesn't support hook)
        if self.activations is None or self.gradients is None:
            # Fallback to returning original image if Grad-CAM hook fails
            orig_b64 = base64.b64encode(image_bytes).decode("utf-8")
            return orig_b64, {"primary_region": "Undetermined", "activation_level": "Low"}

        # 2. Compute weights using GAP of gradients
        # activations shape: [1, C, H, W] for EfficientNet
        activations = self.activations.detach()
        gradients = self.gradients.detach()
        
        if len(activations.shape) == 4:
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1).squeeze(0)
        else:
            # For ViT (token activations shape: [1, SeqLen, Dim])
            # Average over seq dimension
            weights = torch.mean(gradients, dim=1, keepdim=True)
            cam = torch.sum(weights * activations, dim=2).squeeze(0)
            # Reshape seq tokens to spatial shape (e.g. 14x14 grid)
            side = int(np.sqrt(cam.size(0) - 1)) if cam.size(0) in [197, 257] else 14
            cam = cam[1:].reshape(side, side)  # exclude class token

        # 3. Apply ReLU and normalize
        cam = torch.relu(cam)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        cam = cam.cpu().numpy()

        # 4. Generate PIL Images
        original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = original_img.size
        
        # Resize heatmap to match original image
        heatmap_gray = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        
        # Colorize grayscale heatmap: Blue (low) -> Yellow (mid) -> Red (high)
        heatmap_color = ImageOps.colorize(heatmap_gray, black="#0000ff", white="#ff0000", mid="#ffff00")
        
        # Blend original image and heatmap (65% original, 35% heatmap)
        overlay_img = Image.blend(original_img, heatmap_color, alpha=0.35)
        
        # Convert overlay image to base64
        buffered = io.BytesIO()
        overlay_img.save(buffered, format="JPEG")
        overlay_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 5. Extract explainable regions based on activation intensity
        # Split spatial dimensions to check where high values reside
        # As a simple heuristic: check if top activation is in upper/lower/left/right of the frame
        y_max, x_max = np.unravel_index(np.argmax(cam), cam.shape)
        h_grid, w_grid = cam.shape
        
        region = "Center of the leaf"
        if y_max < h_grid / 3:
            region = "Leaf tip / upper margin"
        elif y_max > 2 * h_grid / 3:
            region = "Base of the leaf / lower margin"
        elif x_max < w_grid / 3:
            region = "Left leaf margin"
        elif x_max > 2 * w_grid / 3:
            region = "Right leaf margin"

        explainable_data = {
            "primary_region": region,
            "feature_detected": "Concentrated necrotic lesions" if cam_max.item() > 0.6 else "Mild spotty activation",
            "activation_level": "High" if cam_max.item() > 0.7 else "Moderate"
        }
        
        return overlay_b64, explainable_data
