import threading
import time
from settings import settings
from agents import LMStudioLLM, LocationAgent, ItineraryAgent, GeolocationAgent, \
    check_existing_embedding, connect_to_milvus, create_collection, retrieve_cached_data, create_index, \
    generate_embedding, store_embedding_in_milvus


def loader():
    """Displays a simple loading animation in a separate thread."""
    while not stop_loading:
        for char in "|/-\\":
            print(f"\rLoading... {char}", end="", flush=True)
            time.sleep(0.2)


if __name__ == "__main__":
    local_llm = LMStudioLLM(
        api_url=settings.lm_api_url,
        model_name=settings.lm_model_name,
    )

    location_agent = LocationAgent(llm=local_llm)
    itinerary_agent = ItineraryAgent(llm=local_llm)
    geolocation_agent = GeolocationAgent(llm=local_llm)

    connect_to_milvus()
    create_collection()
    create_index()

    while True:
        print("Enter your travel details (e.g., '5 days in Paris') or type 'exit' to quit.")
        user_input = input().strip()
        if user_input.lower() in ['exit', 'quit']:
            break

        try:
            days, location = user_input.split(" days in ")
            days = days.strip()
            location = location.strip()

            if check_existing_embedding(location):
                print(f"\nLocation '{location}' found in Milvus. Retrieving data...")
                retrieve_cached_data(location)
            else:
                print(f"\nLocation '{location}' not found in Milvus. Querying LLM for new data...")

                stop_loading = False
                loader_thread = threading.Thread(target=loader)
                loader_thread.start()

                location_suggestions = location_agent.identify_locations(days, location)

                stop_loading = True
                loader_thread.join()

                print("\nLocation Suggestions:", location_suggestions)

                stop_loading = False
                loader_thread = threading.Thread(target=loader)
                loader_thread.start()

                itinerary = itinerary_agent.create_itinerary(days, location, location_suggestions)

                stop_loading = True
                loader_thread.join()

                print("\nGenerated Itinerary:\n", itinerary)

                stop_loading = False
                loader_thread = threading.Thread(target=loader)
                loader_thread.start()

                geolocation_data = geolocation_agent.get_geolocation(itinerary)

                stop_loading = True
                loader_thread.join()

                print("\nGeolocation Data:\n", geolocation_data)

                print("\nStoring data in Milvus for future use...")
                embedding = generate_embedding(location)  # Create embedding for location
                store_embedding_in_milvus(embedding, location, itinerary, location_suggestions, geolocation_data)
                print(f"Data for '{location}' stored successfully in Milvus!")

        except ValueError:
            print("Invalid input format. Please enter your travel details in the format 'X days in Y city/country'.")
        print("-" * 50)
