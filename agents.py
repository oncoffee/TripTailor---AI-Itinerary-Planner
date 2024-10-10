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
