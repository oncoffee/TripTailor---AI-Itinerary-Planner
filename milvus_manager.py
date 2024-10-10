from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections
from settings import settings
import time


class MilvusManager:
    def __init__(self, host=None, port=None):
        self.host = host or settings.milvus_host
        self.port = port or settings.milvus_port
        self.collection_name = "location_embeddings"
        self.dimension = 384

    def connect(self):
        connections.connect("default", host=self.host, port=self.port)
        print(f"Connected to Milvus at {self.host}:{self.port}")

    def create_collection(self):
        if self.collection_name in list_collections():
            print(f"Collection '{self.collection_name}' exists, dropping it...")
            Collection(self.collection_name).drop()

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="location_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="itinerary", dtype=DataType.VARCHAR, max_length=20000),
            FieldSchema(name="location_suggestions", dtype=DataType.VARCHAR, max_length=20000),
            FieldSchema(name="geolocation_data", dtype=DataType.VARCHAR, max_length=20000),
        ]

        schema = CollectionSchema(fields)
        collection = Collection(self.collection_name, schema)
        print(f"Collection created: {self.collection_name}")

    def create_index(self):
        collection = Collection(self.collection_name)

        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }

        collection.create_index(field_name="embedding", index_params=index_params)
        print("Index creation initiated on the 'embedding' field.")

        while not collection.has_index():
            print("Waiting for index to be ready...")
            time.sleep(1)

        print("Index created successfully.")

    def generate_embedding(self, text):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embedding = model.encode(text)
        return embedding

    def store_embedding(self, location_name, embedding, itinerary, location_suggestions, geolocation_data):
        collection = Collection(self.collection_name)

        collection.load()

        collection.insert([
            [location_name],
            [embedding],
            [itinerary],
            [location_suggestions],
            [geolocation_data]
        ])

        collection.flush()

        print(f"Stored embedding and metadata in Milvus for '{location_name}'")

    def retrieve_cached_data(self, location_name):
        collection = Collection(self.collection_name)

        query_embedding = self.generate_embedding(location_name)

        search_result = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=1
        )

        if len(search_result) > 0 and len(search_result[0].ids) > 0:
            hit = search_result[0]
            result_id = hit.ids[0]

            result_data = collection.query(expr=f"id == {result_id}",
                                           output_fields=["location_name", "itinerary", "location_suggestions",
                                                          "geolocation_data"])

            if result_data:
                result = result_data[0]
                print("\nCached Data Found:")
                print(f"Location: {result.get('location_name')}")
                print("Itinerary:", result.get('itinerary'))
                print("Location Suggestions:", result.get('location_suggestions'))
                print("Geolocation Data:", result.get('geolocation_data'))
            else:
                print(f"No cached data found for location: {location_name}")
        else:
            print(f"No cached data found for location: {location_name}")

    def check_existing_embedding(self, location_name):
        collection = Collection(self.collection_name)

        collection.load()

        if collection.num_entities == 0:
            print(f"No data exists in the collection.")
            return False

        query_embedding = self.generate_embedding(location_name)

        search_result = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 20}},
            limit=1
        )

        if len(search_result) > 0 and len(search_result[0].ids) > 0:
            print(f"Location '{location_name}' found in Milvus.")
            return True
        else:
            print(f"Location '{location_name}' not found in Milvus.")
            return False