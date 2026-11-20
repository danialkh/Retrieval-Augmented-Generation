from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embeddings = self.model.encode([image])
        return embeddings[0]


def verify_image_embedding(image_path: str) -> None:
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")