import os
import pyttsx3
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

# 1. Define the tools
# (The @tool decorator is optional for create_agent, but it works fine with it)
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # In a real app, you would call a real weather API here (e.g., OpenWeatherMap)
    return f"The weather in {location} is currently 72°F and sunny."

@tool
def get_top_news(category: str) -> str:
    """Get the top news headlines for a given category."""
    # In a real app, you would call a real news API here (e.g., NewsAPI)
    return f"Top {category} news for today: 1. AI agents are revolutionizing software development. 2. A new era of productivity has begun."

# 2. Setup the LLM
# Load environment variables from .env file
load_dotenv()

# We use the gemini-2.5-flash model, which is fast and great for general agentic tasks
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 3. Create the agent
tools = [get_weather, get_top_news]

# The installed version of LangChain uses LangGraph under the hood via create_agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful, extremely witty, and charming daily briefing assistant. "
        "Your job is to check the weather and the top news headlines using the tools provided to you, "
        "then write a short, witty summary so the user knows exactly what to expect today before leaving the house."
    )
)

if __name__ == "__main__":
    print("Starting the Daily Briefing Agent...\n")
    user_input = "What do I need to know today before I leave the house? I'm in New York and I care about Tech news."
    
    # Run the agent
    # We pass the input as a list of messages
    response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    
    # The final output is the content of the last message
    agent_output = response["messages"][-1].content
    
    print("\n\n=== Your Daily Briefing ===\n")
    print(agent_output)
    print("\n===========================\n")
    
    # Initialize TTS engine
    print("Reading briefing aloud...")
    engine = pyttsx3.init()
    
    # Optional: adjust speech rate
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 20)
    
    # Speak the output
    engine.say(agent_output)
    engine.runAndWait()
