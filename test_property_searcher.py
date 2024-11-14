import logging

from qdrant_client import QdrantClient
from common_class import PropertyFilters, SearchMode
from property_searcher import PropertySearcher


logging.basicConfig(level=logging.INFO)
# client = QdrantClient(url="http://localhost:6333")
client = QdrantClient(url="https://1ae0f535-9ed2-47a3-9955-72bc7a9e8e0f.us-east4-0.gcp.cloud.qdrant.io:6333",
                          api_key="WFjB56G9fZwpga0eVBS_U6joSUqSRm5xHixbjiM0pC1Apj9VODse2Q")
searcher = PropertySearcher(client)

# Define search filters
filters = PropertyFilters(
    min_price=1000.0,
    max_price=2500000.0,
    min_bedrooms=2,
    max_bedrooms=5,
    must_have_amenities=["parking"]
)

# Find similar properties
similar_properties = searcher.search_similar_properties(
    property_id=33422879720, # 7510668,7527604,33422879720
    mode=SearchMode.BALANCED,
    filters=filters,
    top_k=5
)

# Display results
for prop in similar_properties:
    logging.info(f"Got similar property: {prop}")