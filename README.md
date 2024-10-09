
# TripTailor - Intelligent Itinerary Planner 🧳✈️

TripTailor is an intelligent itinerary planning tool designed to optimize your travel experiences. Powered by a local language model (LLM) and Milvus vector database, TripTailor takes user input about travel destinations and durations and generates detailed itineraries, including must-see locations, optimal times to visit, meal suggestions, and geographical data.

## Features

- **Location Suggestions**: Provides recommended landmarks and activities based on the travel duration and destination.
- **Detailed Itinerary Generation**: Creates a day-by-day plan with suggested activities, ideal times to visit, and recommendations for breakfast, lunch, and dinner aligned with the travel route.
- **Geolocation Data**: Enhances itineraries with latitude and longitude information for easy navigation.
- **Data Caching**: Uses Milvus vector database to store location embeddings, so future queries for the same locations are faster and more efficient.

## Getting Started

### Prerequisites

- **Python 3.10+**: Ensure you have Python installed.
- **dspy library**: To interface with your local language model (LLM).
- **Milvus**: A vector database for storing and searching location embeddings.
- **Docker**: Required for running Milvus using the provided `docker-compose.yaml` file.
- **Access to a Local LLM Server**: The LLM server should support an OpenAI-style API for chat completions.

### Installation

1. **Clone the Repository**

   ``
   git clone https://github.com/yourusername/TripTailor.git
   cd TripTailor
   ``

2. **Set Up Python Virtual Environment and Install Dependencies**

   ``
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ``

   The `requirements.txt` includes:

   ``
   dspy-ai
   openai
   pydantic
   pydantic-settings
   sentence-transformers
   pymilvus
   requests
   ``

3. **Configure Environment Variables**

   Copy the `.env_template` to `.env` and provide your local LLM server and Milvus settings:

   ``
   cp .env_template .env
   ``

   Then, edit the `.env` file to include your LLM and Milvus details:

   ``env
   LM_API_URL=your_llm_api_url
   LM_MODEL_NAME=your_llm_model_name
   MILVUS_HOST=your_milvus_host
   MILVUS_PORT=your_milvus_port
   ``

4. **Run Milvus Using Docker**

   If you don't have Milvus running locally, use Docker to start the necessary services (Milvus, MinIO, etc.):

   ``
   docker-compose up -d
   ``

   This will spin up the required services for Milvus to run. Ensure that the Milvus instance is accessible via the host and port specified in your `.env` file.

### Usage

1. **Run the Application**

   To start TripTailor, run the `main.py` script:

   ``
   python main.py
   ``

2. **Provide Your Travel Details**

   When prompted, enter your travel details in the format:

   ``
   [number of days] days in [city/country]
   ``

   _Example:_

   ``
   5 days in Paris
   ``

3. **Interactive Planning**

   The app will generate:
   - **Location Suggestions**: Recommended places to visit based on the LLM.
   - **Itinerary**: A detailed itinerary with optimal times for visiting each destination and meal suggestions.
   - **Geolocation Data**: Latitude and longitude coordinates for each location.

4. **Caching with Milvus**

   - **First-time query**: If the location doesn't exist in Milvus, the system queries the LLM, stores the data in Milvus, and presents it to the user.
   - **Subsequent queries**: If the location is already stored in Milvus, the app retrieves cached data to speed up the process.

### Code Structure

- **main.py**: The main entry point that handles user input, interacts with agents, and manages the Milvus database.
- **agents.py**: Contains the core agents responsible for location suggestions, itinerary generation, and geolocation data:
  - **LMStudioLLM**: Interfaces with the local LLM server to generate responses.
  - **LocationAgent**: Generates a list of suggested locations based on travel duration and destination.
  - **ItineraryAgent**: Creates a detailed itinerary with suggested times and meal spots.
  - **GeolocationAgent**: Provides latitude and longitude for destinations in the itinerary.
- **settings.py**: Manages environment configuration, including LLM and Milvus settings using `pydantic-settings`.
- **.env_template**: Template for environment variables (LLM API and Milvus configuration).
- **docker-compose.yaml**: Docker Compose setup for running Milvus and its dependencies (MinIO and etcd).

### Example Output

``
Enter your travel details (e.g., '5 days in Paris') or type 'exit' to quit.
> 5 days in Paris

Location Suggestions: 
1. Eiffel Tower
2. Louvre Museum
3. Notre-Dame Cathedral
...

Generated Itinerary:
Day 1: Morning: Eiffel Tower, Afternoon: Louvre Museum, Evening: Dinner near Notre-Dame...
...

Geolocation Data:
Eiffel Tower: [48.8583° N, 2.2945° E]
...
``

### Customization

- **LLM Configuration**: Modify the `.env` file to change the LLM server URL and model name.
- **Milvus Settings**: Customize the Milvus host and port in the `.env` file and the `docker-compose.yaml` if necessary.
- **Prompts**: Adjust the prompts in `agents.py` to modify how locations and itineraries are generated.

## Contributing

Feel free to fork this repo and open a pull request if you have suggestions, enhancements, or bug fixes!

## License

[MIT License](LICENSE)

---

Happy Travels! 🌍🚀
