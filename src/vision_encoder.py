from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class VisionEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading VisionEncoder model {model_name} on device: {self.device}")
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.model.eval() # Set model to evaluation mode
            logger.info("VisionEncoder model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load VisionEncoder model {model_name}: {e}")
            raise

    def encode(self, images, normalize_embeddings: bool = True):
        """
        Encodes a single image or a list of images into embeddings.
        Args:
            images: A single PIL Image object or a list of PIL Image objects.
            normalize_embeddings: Whether to normalize the embeddings to unit length.
        Returns:
            A numpy array of image embeddings.
        """
        if not isinstance(images, list):
            images = [images]

        if not all(isinstance(img, Image.Image) for img in images):
            raise TypeError("All items in 'images' must be PIL Image objects.")

        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            embeddings = self.model.get_image_features(**inputs)

        if normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1) # L2 normalization

        return embeddings.cpu().numpy()

    def encode_text(self, texts, normalize_embeddings: bool = True):
        """
        Encodes a single text or a list of texts into CLIP text embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("`texts` must be a string or a list of strings.")

        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            embeddings = self.model.get_text_features(**inputs)

        if normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().numpy()
