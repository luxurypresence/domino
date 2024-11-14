import logging
import uuid
from typing import Any

import boto3
import numpy as np
import pandas as pd

from s3_service import S3Service

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

class PropertyLoader:

    def __init__(self, options: dict[str, Any], s3_service: S3Service = None) -> None:
        """Initializes the Unique Property ID workflow
        @param options: Configuration options for the workflow.
        @param s3_service: Service for interacting with AWS S3.
        """
        self.options = options
        self.s3_bucket = options.get('s3_bucket', 'lp-datalakehouse-stage/warehouse')
        self.s3_service = s3_service or S3Service()
        self.start_date = options.get('start_date')
        self.end_date = options.get('end_date')
        self.source_athena_database = options.get('source_athena_database', 'default')
        self.property_table_name = options.get('property_table_name', 'property')
        self.athena_query_path = f'domino-demo/athena-queries'
        self.temp_s3_path = f's3://{self.s3_bucket}/domino-demo/temp/{str(uuid.uuid4())}'
        self.state_column_name = options.get('state_column_name', 'state_or_province')
        self.city_column_name = options.get('city_column_name', 'city')
        self.timestamp_column_name = options.get('timestamp_column_name', 'lp_processed_timestamp')
        self.limit = options.get('limit', 1000)

    def load_property_records(self, states=None) -> pd.DataFrame:
        """Load property records according to province/state, start time and end time.
        It only loads the latest property record filtering out the duplicated records.
        @param states: province/state names
        @return: property data frame
        """
        if self.s3_service.check_db_table_exists(self.source_athena_database, self.property_table_name):
            logging.info(f"Table {self.property_table_name} exists in db {self.source_athena_database}, reading data")
            sql = self.prepare_load_property_sql(states)
            df = self.s3_service.read_athena(sql,
                                             self.source_athena_database,
                                             s3_output=f's3://{self.s3_bucket}/{self.athena_query_path}/{str(uuid.uuid4())}/')
            return df
        logging.error(f"Table {self.property_table_name} does not exist in db {self.source_athena_database}")
        return pd.DataFrame()

    def prepare_load_property_sql(self, states: list=None) -> str:
        """Prepare the sql for loading property records.
        The sql only loads the latest property record filtering out the duplicated records.
        @param states: province or state names
        @return: prepared query sql
        """
        where_clauses = ["lp_full_address IS NOT NULL",
                         "lp_full_address != ''",
                         "lower(lp_full_address) not like '%undisclosed%'",
                         "lp_property_type in ('HOUSE', 'CONDO', 'TOWNHOUSE', 'APARTMENT')",
                         "lp_provider_id = 'trestle-rebny'",
                         "postal_code in ('11205', '11217', '10016', '10025')",
                         "lower(lp_listing_status)='active'"
                         ]
        if states:
            # remove the empty state
            states = [f"'{state}'" for state in states if state]
            if states:
                where_clauses.append(f"state_or_province in ({','.join(states)})")
        if self.start_date:
            where_clauses.append(
                f"{self.timestamp_column_name} >= cast('{self.start_date.replace('T', ' ')}' as timestamp)")
        if self.end_date:
            where_clauses.append(
                f"{self.timestamp_column_name} <= cast('{self.end_date.replace('T', ' ')}' as timestamp)")
        where_clause = " AND ".join(where_clauses)

        query = f"""
            SELECT lp_provider_id,lp_listing_id,listing_id,lp_full_address,lp_formatted_address,lp_property_type,
                   association_amenities,interior_features,appliances,exterior_features,lot_features,architectural_style,
                   community_features,list_price,price_range,lp_photos,bedrooms_total,
                   lp_calculated_bath,lp_listing_description,
                   accessibility_features,building_features,fireplace_features,laundry_features,parking_features,pool_features,security_features,waterfront_features,
                   country,state_or_province,county_or_parish,city,lp_processed_timestamp,event_modification_timestamp
            FROM (
                SELECT lp_provider_id,lp_listing_id, listing_id, lower(lp_full_address) as lp_full_address,
                       association_amenities,interior_features,appliances,exterior_features,lot_features,architectural_style,
                       community_features,list_price,price_range,lp_photos,bedrooms_total,
                       lp_calculated_bath,lp_listing_description,
                       accessibility_features,building_features,fireplace_features,laundry_features,parking_features,pool_features,security_features,waterfront_features,
                       CAST(lp_processed_timestamp AS timestamp(3)) AS lp_processed_timestamp,
                       lp_formatted_address, country, state_or_province,lower(city) as city,county_or_parish,
                       lp_property_type,
                       CAST(event_modification_timestamp AS timestamp(3)) AS event_modification_timestamp,
                       ROW_NUMBER() OVER (PARTITION BY lp_provider_id, lp_listing_id 
                                          ORDER BY event_modification_timestamp DESC, lp_processed_timestamp DESC) AS rn
                FROM {self.source_athena_database}.{self.property_table_name}
                WHERE {where_clause}
            )
            WHERE rn = 1
            limit {self.limit}
        """

        return query


def query_property_records_from_datalake() -> list:
    pd.set_option('display.max_columns', None)
    boto3.setup_default_session(profile_name='data-staging')
    options = {
        'start_date': '2024-11-01T00:00:00',
        'end_date': '2024-11-15T00:00:00',
        'timestamp_column_name': 'lp_processed_timestamp',
        's3_bucket': 'lp-datalakehouse-stage/warehouse',
        'source_athena_database': 'lp_data_model_stage',
        'source_data_table': 'property',
        'limit': 1000,
    }
    loader = PropertyLoader(options)
    try:

        df =  loader.load_property_records()
        df['list_price'] = df['list_price'].astype(float)
        df['bedrooms_total'] = df['bedrooms_total'].astype(int)
        df['lp_calculated_bath'] = df['lp_calculated_bath'].astype(float)
        # Remove non-numeric characters from lp_listing_id
        df['id'] = df['lp_listing_id'].str.replace(r'\D', '', regex=True).astype(int)
        # Extract photo_url from lp_photos and set it as a list in the lp_photos column
        df['lp_photos'] = df['lp_photos'].apply(lambda photos: [photo['photo_url'] for photo in photos])

        # Convert each specified column to a list if it's an ndarray
        columns_to_convert = [
            'association_amenities', 'interior_features', 'appliances', 'exterior_features',
            'community_features', 'accessibility_features', 'building_features', 'fireplace_features',
            'laundry_features', 'parking_features', 'pool_features', 'security_features', 'waterfront_features',
            'lot_features', 'architectural_style',
        ]
        for col in columns_to_convert:
            if col in df.columns:  # Check if column exists in the DataFrame
                df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        return df.to_dict(orient='records')
    except Exception as ex:
        logging.error(f"Property records loading failed, error: {ex}")
        raise
    finally:
        logging.info(f"Deleting the temp path on s3 bucket: {f's3://{loader.s3_bucket}/{loader.athena_query_path}'}")
        loader.s3_service.wr_client.s3.delete_objects(f's3://{loader.s3_bucket}/{loader.athena_query_path}')