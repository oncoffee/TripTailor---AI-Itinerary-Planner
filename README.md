
# TripTailor - Intelligent Itinerary Planner 🧳✈️

TripTailer is an intelligent itinerary planning tool designed to help you make the most of your travel experiences. Powered by a local language model (LLM), TripTailor takes user input about travel destinations and durations and generates detailed itineraries including must-see locations, optimal times to visit, meal suggestions, and geographical data.

## Features

- **Location Suggestions**: Provides recommended landmarks and activities for a given travel duration and destination.
- **Detailed Itinerary Generation**: Creates a day-by-day plan with recommended activities, times to visit, and suggestions for breakfast, lunch, and dinner that align with the travel route.
- **Geolocation Data**: Enhances the itinerary with latitude and longitude information for easy navigation.

## Getting Started

### Prerequisites

- **Python 3.10+**: Ensure you have Python installed.
- **dspy library**: To interface with your language model.
- **Access to a Local LLM Server**: The LLM server should support an OpenAI-style API for chat completions.

### Installation

1. **Clone the Repository**
   ``
   git clone https://github.com/yourusername/TripTailor.git
   cd TripTailor
   ``

2. **Install Python Dependencies**
   Make sure to have a virtual environment set up:
   ``
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ``
   The \`requirements.txt\` includes:
   ``
   dspy-ai
   openai
   pydantic
   pydantic-settings
   ``

3. **Configure Environment Variables**
   Copy \`.env_template\` to \`.env\` and add your local LLM server details:
   ``
   cp .env_template .env
   ``
   Then, edit \`.env\`:
   ``env
   LM_API_URL=your_llm_api_url
   LM_MODEL_NAME=your_llm_model_name
   ``

### Usage

1. **Run the Application**
   ``
   python main.py
   ``

2. **Provide Your Travel Details**
   When prompted, enter your travel details in the format:
   ```
   [number of days] days in [city/country]
   ```
   _Example_:
   ```
   5 days in Paris
   ```

3. **Interactive Planning**
   The app will generate:
   - **Location Suggestions**: Recommendations for places to visit.
   - **Itinerary**: A day-by-day breakdown of your travel plans with optimal times and meal suggestions.
   - **Geolocation Data**: Latitude and longitude coordinates for your destinations.

### Code Structure

- **main.py**: The main entry point for running the itinerary planner. Handles user input and initiates the agents to provide travel suggestions, itineraries, and geolocation data.
- **agents.py**: Contains the core agent classes:
  - **LMStudioLLM**: Interfaces with the local LLM server to generate responses.
  - **LocationAgent**: Suggests locations and activities for a given travel period and destination.
  - **ItineraryAgent**: Creates a detailed itinerary, including recommended times and meal spots within the travel route.
  - **GeolocationAgent**: Provides geolocation data (latitude and longitude) for each destination in the itinerary.
- **settings.py**: Manages configuration using environment variables. Uses \`pydantic-settings\` to easily load the \`.env\` file and access the LLM server settings.
- **.env_template**: Template for setting up environment variables for connecting to your LLM.

### Example Output

```
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
```

### Customization

- **LLM Configuration**: Modify the \`.env\` file to change the LLM server URL and model name.
- **Travel Preferences**: Adjust prompts in \`agents.py\` to tweak the travel suggestions, itinerary breakdown, and geolocation formats as per your preference.

## Contributing

Feel free to fork this repo and open a pull request if you have any suggestions, enhancements, or bug fixes!

## License

[MIT License](LICENSE)

---

Happy Travels! 🌍🚀
