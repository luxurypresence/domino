import csv
import logging
from qdrant_client import QdrantClient
from common_class import PropertyFilters, SearchMode
from property_indexer import PropertyIndexer
from property_searcher import PropertySearcher


# Function to fetch all property IDs from a collection in Qdrant
def get_all_property_ids_from_collection(client, collection_name):
    # Initialize the list for property IDs
    property_data = []
    next_page_offset = None

    # Use `scroll` method to iterate over all points in the collection
    while True:
        # Unpack the returned tuple into records and next_page_offset
        records, next_page_offset = client.scroll(collection_name=collection_name, offset=next_page_offset, limit=100)
        for record in records:
            print(record.id)
            print(record.payload.get("lp_listing_id"))
            data = {
                "property_id": record.id,
                "lp_listing_id": record.payload.get("lp_listing_id"),
            }
            property_data.append(data)
        # Break if there are no more pages
        if next_page_offset is None:
            break
    return property_data

# Function to fetch all property IDs from a collection in Qdrant
# def get_all_property_ids_from_collection(client, collection_name):
#     # Use `scroll` method to iterate over all points in the collection
#     property_ids = []
#     scroll_result = client.scroll(collection_name=collection_name, limit=100)
#     print(scroll_result)
#     property_ids.extend([record.id for record in scroll_result.records])
#     # Continue scrolling until there are no more points
#     while scroll_result.next_page_offset is not None:
#         scroll_result = client.scroll(collection_name=collection_name, offset=scroll_result.next_page_offset, limit=100)
#         property_ids.extend([record.id for record in scroll_result.records])
#
#     return property_ids  # Ensure this returns a list


# Function to search similar properties for all properties in Qdrant or from provided property ID list, and save to CSV
def search_and_save_similar_properties(client, searcher, filters=None, property_data=None, mode=SearchMode.BALANCED,
                                       top_k=5, output_csv="similar_properties.csv"):
    # Step 1: Retrieve property IDs from Qdrant if no list is provided
    if property_data is not None:
        logging.info("Using provided property ID list for similarity search.")
    else:
        collection_name = "location_vectors"  # Replace with your main collection name
        logging.info(f"No property ID list provided, retrieving from collection '{collection_name}'.")
        property_data = get_all_property_ids_from_collection(client, collection_name)

        # Ensure that property_id_list is a list, even if empty
        if property_data is None:
            logging.error(f"Failed to retrieve property IDs from collection '{collection_name}'. Exiting function.")
            return
        logging.info(f"Retrieved {len(property_data)} property IDs from collection '{collection_name}'.")

    # Step 2: Open CSV file to write similar properties for each property_id
    with open(output_csv, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerow(["property_id", "similar_property_ids"])

        # Iterate over each property ID to find similar properties
        for property in property_data:
            # Find similar properties
            similar_properties = searcher.search_similar_properties(
                property_id=property['property_id'],
                mode=mode,
                filters=filters,  # Pass filters (can be None)
                top_k=top_k
            )

            # Extract similar property IDs
            similar_property_ids = [prop["lp_listing_id"] for prop in similar_properties]

            # Write property_id and similar_property_ids to CSV
            csv_writer.writerow([property['lp_listing_id'], similar_property_ids])

            logging.info(f"Similar properties for {property['property_id']} and listing_id {property['lp_listing_id']}: {similar_property_ids}")

    logging.info(f"Similar properties saved to {output_csv}")


# Example usage
if __name__ == "__main__":
    # Initialize Qdrant client
    client = QdrantClient(url="https://1ae0f535-9ed2-47a3-9955-72bc7a9e8e0f.us-east4-0.gcp.cloud.qdrant.io:6333",
                          api_key="WFjB56G9fZwpga0eVBS_U6joSUqSRm5xHixbjiM0pC1Apj9VODse2Q")
    # Initialize Qdrant client and components
    indexer = PropertyIndexer(client)
    searcher = PropertySearcher(client)

    # Define optional search filters (can be None)
    filters = PropertyFilters(
        min_price=8000.0,
        max_price=1000000.0,
        min_bedrooms=2,
        max_bedrooms=4,
        must_have_amenities=["parking"]
    )

    # Example property ID list (optional; will fetch from Qdrant if not provided)
    property_data = None  # Set to None to fetch from Qdrant

    # Run the similarity search for all properties in Qdrant or from provided list, saving results to CSV
    search_and_save_similar_properties(client, searcher, filters=None, property_data=None,
                                       mode=SearchMode.BALANCED, top_k=5, output_csv="similar_properties.csv")
