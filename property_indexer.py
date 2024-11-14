import logging
from time import sleep
from typing import Dict

from qdrant_client import QdrantClient, models

from property_data import PropertyData

# Set up logging configuration
logging.basicConfig(level=logging.INFO)


class PropertyIndexer:
    """
    Handles indexing of property data into Qdrant collections.
    """

    def __init__(self, client: QdrantClient):
        self.client = client
        self.property_data = PropertyData()

    def validate_property_data(self, property_data: Dict) -> bool:
        """
        Validate required fields and data formats in property data.

        Args:
            property_data (Dict): The property data to validate.

        Returns:
            bool: True if validation passes, False otherwise.
        """
        required_fields = ["listing_id", "lp_full_address", "association_amenities", "lp_photos"]
        for field in required_fields:
            if field not in property_data:
                logging.error(f"Missing required field: {field}")
                return False
        return True

    def index_property(self, property_data: Dict) -> bool:
        """
        Index a property with normalized vectors into Qdrant collections.

        Args:
            property_data (Dict): The property data to index.

        Returns:
            bool: True if indexing is successful, False otherwise.
        """
        try:
            if not self.validate_property_data(property_data):
                raise ValueError("Property data validation failed")

            # Generate embeddings
            text_embeddings = self.property_data.generate_text_embeddings(property_data)
            image_embedding = self.property_data.generate_image_embedding(property_data["lp_photos"])

            if image_embedding is None:
                raise ValueError("Failed to generate image embeddings")

            # Store normalized vectors in their respective collections
            collections = [
                "location_vectors",
                "features_vectors",
                "visual_vectors"
            ]
            vectors = [
                text_embeddings["location"],
                text_embeddings["features"],
                image_embedding
            ]

            for collection, vector in zip(collections, vectors):
                response = self.client.upsert(
                    collection_name=collection,
                    points=[{
                        "id": property_data["id"],
                        "vector": vector.tolist(),
                        "payload": property_data
                    }]
                )
                logging.info(f"Upsert response for collection {collection}: {response}")

            logging.info(f"Successfully indexed property {property_data['id']}")
            return True

        except Exception as e:
            logging.error(f"Error indexing property {property_data.get('id', 'unknown')}: {e}")
            return False

    def initialize_collections(self, client: QdrantClient):
        """
        Initialize Qdrant collections with proper vector configurations.

        Args:
            client (QdrantClient): The Qdrant client instance.
        """
        collections = {
            "location_vectors": 384,
            "features_vectors": 384,
            "visual_vectors": 512
        }

        for collection in collections.keys():
            try:
                # Check if the collection already exists
                client.get_collection(collection_name=collection)
                logging.info(f"Collection '{collection}' already exists.")
            except Exception:
                # Create the collection if it doesn't exist
                client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(
                        size=collections[collection],  # Adjust based on your model's output dimension
                        distance=models.Distance.COSINE
                    )
                )
                logging.info(f"Created collection '{collection}'.")
                sleep(1)
                collection_info = client.get_collection(collection)
                logging.debug(f"Collection info: {collection_info}")