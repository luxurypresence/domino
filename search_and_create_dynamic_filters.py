import csv
import logging
from qdrant_client import QdrantClient
from common_class import PropertyFilters, SearchMode
from property_indexer import PropertyIndexer
from property_searcher import PropertySearcher


# Function to fetch all property data with calculated filter bounds from a collection in Qdrant
def get_all_property_data_from_collection(client, collection_name):
    # Initialize list for property data
    property_data = []
    next_page_offset = None

    # Use `scroll` method to iterate over all points in the collection
    while True:
        # Unpack the returned tuple into records and next_page_offset
        records, next_page_offset = client.scroll(collection_name=collection_name, offset=next_page_offset, limit=100)

        # Collect each record's ID and filtering criteria from payload
        for record in records:
            # Extract values from the payload
            list_price = record.payload.get("list_price", 0)  # Default to 0 if not available
            bedrooms_total = record.payload.get("bedrooms_total", 0)  # Default to 0 if not available

            # Calculate smart filter bounds for price based on ±5% of list_price
            price_variation = 0.05 * list_price  # 5% of list_price
            min_price = max(0, list_price - price_variation)  # Ensuring no negative value
            max_price = list_price + price_variation

            # Calculate filter bounds for bedrooms with ±2
            min_bedrooms = max(0, bedrooms_total - 2)  # Ensuring no negative value
            max_bedrooms = bedrooms_total + 2
            print(f"payload record {record.payload}")
            print(type(record.payload))
            print(f"lp listing id {record.payload['lp_listing_id']}")

            data = {
                "property_id": record.id,
                "lp_listing_id": record.payload.get("lp_listing_id"),
                "min_price": min_price,
                "max_price": max_price,
                "min_bedrooms": min_bedrooms,
                "max_bedrooms": max_bedrooms
            }
            property_data.append(data)

        # Break if there are no more pages
        if next_page_offset is None:
            break

    return property_data


# Function to search similar properties for all properties in Qdrant or from provided property ID list, and save to CSV
def search_and_save_similar_properties(client, searcher, property_data, mode=SearchMode.BALANCED, top_k=5,
                                       output_csv="search_and_create_dynamic_filters.csv"):
    # Step 1: Open CSV file to write similar properties for each property_id
    with open(output_csv, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerow(["property_id", "similar_property_ids"])

        # Iterate over each property to find similar properties with dynamic filters
        for data in property_data:
            # Create a dynamic filter for each property based on its attributes
            filters = PropertyFilters(
                min_price=data["min_price"],
                max_price=data["max_price"],
                min_bedrooms=data["min_bedrooms"],
                max_bedrooms=data["max_bedrooms"]
            )

            # Find similar properties
            try:
                similar_properties = searcher.search_similar_properties(
                    property_id=data["property_id"],
                    mode=mode,
                    filters=filters,  # Pass dynamic filters
                    top_k=top_k
                )

                print(similar_properties)

                # Extract similar property IDs
                similar_property_ids = [prop["lp_listing_id"] for prop in similar_properties]

                # Write property_id and similar_property_ids to CSV
                csv_writer.writerow([data["lp_listing_id"], similar_property_ids])
                logging.info(f"Similar properties for {data['property_id']} and listing_id is {data['lp_listing_id']}: {similar_property_ids}")

            except Exception as e:
                logging.warning(f"Error applying filters to property {data['property_id']}: {e}")

    logging.info(f"Similar properties saved to {output_csv}")


# Example usage
if __name__ == "__main__":
    # Initialize Qdrant client
    client = QdrantClient(url="https://1ae0f535-9ed2-47a3-9955-72bc7a9e8e0f.us-east4-0.gcp.cloud.qdrant.io:6333",
                          api_key="WFjB56G9fZwpga0eVBS_U6joSUqSRm5xHixbjiM0pC1Apj9VODse2Q")
    # Initialize Qdrant client and components
    indexer = PropertyIndexer(client)
    searcher = PropertySearcher(client)

    # Retrieve property data from Qdrant, including dynamic filter criteria
    collection_name = "location_vectors"  # Replace with your actual collection name
    property_data = get_all_property_data_from_collection(client, collection_name)

    # Run the similarity search for all properties in Qdrant, saving results to CSV
    search_and_save_similar_properties(client, searcher, property_data=property_data, mode=SearchMode.BALANCED, top_k=5,
                                       output_csv="search_and_create_dynamic_filters.csv")
