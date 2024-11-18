import logging
from typing import List, Dict, Optional

import numpy as np
import requests
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Set up logging configuration
logging.basicConfig(level=logging.INFO)


class PropertyData:
    """
    Handles property data and embedding generation with proper normalization.
    """

    def __init__(self):
        # Initialize text and image embedding models
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_model = SentenceTransformer('clip-ViT-B-32')

    def preprocess_text(self, text: str) -> str:
        """
        Normalize and clean text data.

        Args:
            text (str): The text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        return text.lower().strip()

    def generate_text_embeddings(self, property_data: Dict) -> Dict[str, np.ndarray]:
        """
        Generate normalized text embeddings for different property attributes.

        Args:
            property_data (Dict): The property data.

        Returns:
            Dict[str, np.ndarray]: A dictionary of normalized embeddings.
        """
        for key, value in property_data.items():
            if isinstance(value, str) and value is None:
                property_data[key] = ''

        # Prepare text data for embeddings
        location_text = self.preprocess_text(
            # f"{property_data.get('location_description', '')} "
            # f"{property_data.get('neighborhood', '')} "
            f"{property_data.get('city', '')} "
            # f"{property_data.get('municipality', '')} "
            f"{property_data.get('county_or_parish', '')} "
            f"{property_data.get('state_or_province', '')} "
            f"{property_data.get('country', '')}"
        )

        property_features_text = self.preprocess_text(" ".join([
            *property_data.get('association_amenities', []),
            *property_data.get('interior_features', []),
            *property_data.get('exterior_features', []),
            *property_data.get('appliances', []),
            *property_data.get('lot_features', []),
            f"property_type: {property_data.get('lp_property_type', '')}",
            f"architectural_style: {property_data.get('architectural_style', '')}",
            f"lp_sale_lease: {property_data.get('lp_sale_lease', '')}",
            # property_data.get('lp_listing_description', 'no description'),
            *property_data.get('accessibility_features', []),
            *property_data.get('building_features', []),
            *property_data.get('fireplace_features', []),
            *property_data.get('laundry_features', []),
            *property_data.get('parking_features', []),
            *property_data.get('pool_features', []),
            *property_data.get('security_features', []),
            *property_data.get('waterfront_features', []),
        ]))

        description_text = self.preprocess_text(
            property_data.get('lp_listing_description', ''),
        )

        # Generate embeddings
        embeddings = {
            "location": self.text_model.encode(location_text),
            "features": self.text_model.encode(property_features_text),
            "description": self.text_model.encode(description_text)
        }

        # Normalize embeddings
        for key in embeddings:
            embeddings[key] = normalize(embeddings[key].reshape(1, -1))[0]

        return embeddings

    def generate_image_embedding(self, image_urls: List[str]) -> Optional[np.ndarray]:
        """
        Generate a normalized aggregated image embedding from property photos.

        Args:
            image_urls (List[str]): A list of image URLs.

        Returns:
            Optional[np.ndarray]: The aggregated image embedding, or None if failed.
        """
        embeddings = []
        for url in image_urls[:5]:  # Limit to first 5 images
            try:
                # Download image
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert('RGB')

                # Generate embedding
                embedding = self.image_model.encode(img)
                embedding = normalize(embedding.reshape(1, -1))[0]
                embeddings.append(embedding)
            except requests.exceptions.RequestException as e:
                logging.warning(f"Error fetching image {url}: {e}")
                continue
            except Exception as e:
                logging.warning(f"Error processing image {url}: {e}")
                continue

        if not embeddings:
            return None

        # Aggregate embeddings by computing the mean
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = normalize(mean_embedding.reshape(1, -1))[0]
        return mean_embedding
