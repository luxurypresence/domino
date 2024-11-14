import logging
from qdrant_client import QdrantClient, models

from common_class import PropertyFilters, SearchMode
from property_indexer import PropertyIndexer
from property_loader import query_property_records_from_datalake
from property_searcher import PropertySearcher

# Set up logging configuration
logging.basicConfig(level=logging.INFO)


# Example usage
if __name__ == "__main__":
    # Initialize Qdrant client
    client = QdrantClient(url="https://1ae0f535-9ed2-47a3-9955-72bc7a9e8e0f.us-east4-0.gcp.cloud.qdrant.io:6333",
                          api_key="WFjB56G9fZwpga0eVBS_U6joSUqSRm5xHixbjiM0pC1Apj9VODse2Q")
    # client = QdrantClient(url="http://localhost:6333")

    # load property records from datalake
    property_list = query_property_records_from_datalake()

    # Initialize components
    indexer = PropertyIndexer(client)
    searcher = PropertySearcher(client)

    # Initialize collections
    indexer.initialize_collections(client)
    # Index property records
    success = False
    count_num = 0
    for property_record in property_list:
        logging.info(f"indexing property {count_num} ----------------------------------------------------------->>>")
        count_num += 1
        if indexer.index_property(property_record):
            success = True

    if success:
        logging.info(f"property list samples: {property_list[:3]}")
        # Define search filters
        filters = PropertyFilters(
            min_price=8000.0,
            max_price=1000000.0,
            min_bedrooms=2,
            max_bedrooms=4,
            must_have_amenities=["parking"]
        )

        # Find similar properties
        similar_properties = searcher.search_similar_properties(
            property_id=property_list[0]["id"],
            mode=SearchMode.BALANCED,
            filters=filters,
            top_k=5
        )

        # Display results
        for prop in similar_properties:
            logging.info(prop)
    else:
        logging.error("Failed to index the example property.")