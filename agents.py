import time
import dspy
import requests
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections

from settings import settings


class LMStudioLLM(dspy.LM):
    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name

    def chat_completion(self, messages):
        headers = {
            'Content-Type': 'application/json',
        }
        data = {
            "model": self.model_name,
            "messages": messages,
        }

        response = requests.post(f"{self.api_url}/v1/chat/completions", json=data, headers=headers)
        response_data = response.json()

        return response_data['choices'][0]['message']['content'].strip()


class LocationAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_message = (
            "You are a travel assistant. Given the user's preferences, suggest popular locations, landmarks, or activities "
            "to explore in a specific city or country. Respond with a brief list of suggestions."
        )

    def create_prompt(self, days, location):
        prompt_text = (
            f"The user wants to travel for {days} days in {location}. "
            f"Please suggest some must-visit locations, landmarks, or activities for them."
        )
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt_text},
        ]

    def identify_locations(self, days, location):
        messages = self.create_prompt(days, location)
        return self.llm.chat_completion(messages)


class ItineraryAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_message = (
            "You are a travel itinerary planner. Based on the given locations, create a detailed day-by-day itinerary "
            "for the user to explore a city or country over a specific number of days. Include activities, recommended times for each day, "
            "and suggest the best time of day to visit each location for the fullest experience. "
            "For each day, suggest breakfast, lunch, and dinner options that are within the route of the planned destinations."
        )

    def create_prompt(self, days, location, suggestions):
        prompt_text = (
            f"The user is traveling to {location} for {days} days. Based on the following suggestions: {suggestions}, "
            "please create a detailed itinerary with daily activities, including the best times of day to visit each place for the fullest experience. "
            "Also suggest places for breakfast, lunch, and dinner each day that are within the route of the destinations."
        )
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt_text},
        ]

    def create_itinerary(self, days, location, suggestions):
        messages = self.create_prompt(days, location, suggestions)
        return self.llm.chat_completion(messages)


class GeolocationAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_message = (
            "You are a geolocation assistant. Given a list of destinations for each day in a travel itinerary, provide the latitude and longitude "
            "for each destination. Respond with the location name, latitude, and longitude in the format: "
            "'location_name: [latitude, longitude]'."
        )

    def create_prompt(self, itinerary):
        prompt_text = (
            f"Based on the following travel itinerary, provide the latitude and longitude for each destination. "
            f"Output the data in the format: 'location_name: [latitude, longitude]'. Itinerary: {itinerary}"
        )
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt_text},
        ]

    def get_geolocation(self, itinerary):
        messages = self.create_prompt(itinerary)
        return self.llm.chat_completion(messages)


def connect_to_milvus():
    connections.connect("default", host=settings.milvus_host, port=settings.milvus_port)
    print("Connected to Milvus")


def create_collection():
    dimension = 384

    if "location_embeddings" in list_collections():
        print("Collection 'location_embeddings' exists, dropping it...")
        Collection("location_embeddings").drop()

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="location_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        FieldSchema(name="itinerary", dtype=DataType.VARCHAR, max_length=20000),
        FieldSchema(name="location_suggestions", dtype=DataType.VARCHAR, max_length=20000),
        FieldSchema(name="geolocation_data", dtype=DataType.VARCHAR, max_length=20000),
    ]

    schema = CollectionSchema(fields)
    collection = Collection("location_embeddings", schema)
    print("Collection created: location_embeddings")


def generate_embedding(text):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding = model.encode(text)
    return embedding


def retrieve_cached_data(location_name):
    collection = Collection("location_embeddings")

    query_embedding = generate_embedding(location_name)

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
            location_name = result.get('location_name')
            itinerary = result.get('itinerary')
            location_suggestions = result.get('location_suggestions')
            geolocation_data = result.get('geolocation_data')

            print("\nCached Data Found:")
            print(f"Location: {location_name}")
            print("Itinerary:", itinerary)
            print("Location Suggestions:", location_suggestions)
            print("Geolocation Data:", geolocation_data)
        else:
            print(f"No cached data found for location: {location_name}")
    else:
        print(f"No cached data found for location: {location_name}")


def store_embedding_in_milvus(embedding, location_name, itinerary, location_suggestions, geolocation_data):
    collection = Collection("location_embeddings")

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


def check_existing_embedding(location_name):
    collection = Collection("location_embeddings")

    collection.load()

    num_entities = collection.num_entities
    if num_entities == 0:
        print(f"No data exists in the collection.")
        return False

    query_embedding = generate_embedding(location_name)

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


def create_index():
    collection = Collection("location_embeddings")

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
