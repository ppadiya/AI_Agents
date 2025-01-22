# Warning control
import warnings
warnings.filterwarnings('ignore')

from openai import OpenAI
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os
from IPython.display import Markdown
from crewai_tools import (
    SerperDevTool,
    WebsiteSearchTool,
    ScrapeWebsiteTool
)
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


# Possible Custom Tools
# Load customer data
# Tap into previous conversations
# Load data from a CRM
# Checking existing bug reports
# Checking existing feature requests
# Checking ongoing tickets
# ... and more

# Some ways of using CrewAI tools.
# search_tool = SerperDevTool()
# scrape_tool = ScrapeWebsiteTool()

# Load environment variables from .env file
load_dotenv()

# Access the Google AI environment variable

llm = LLM(
    model="gemini/gemini-2.0-flash-exp",
    temperature=0.7,
    api_key= os.getenv("GOOGLE_API_KEY")
)

# Agent number 1
support_agent = Agent(
    role="Senior Support Representative",
	goal="Be the most friendly and helpful "
        "support representative in your team",
	backstory=(
		"You work at crewAI (https://crewai.com) and "
        " are now working on providing "
		"support to {customer}, a super important customer "
        " for your company."
		"You need to make sure that you provide the best support!"
		"Make sure to provide full complete answers, "
        " and make no assumptions."
	),
    llm=llm,
    allow_delegation=False,
	verbose=True
)

#Agent number 2
support_quality_assurance_agent = Agent(
	role="Support Quality Assurance Specialist",
	goal="Get recognition for providing the "
    "best support quality assurance in your team",
	backstory=(
		"You work at crewAI (https://crewai.com) and "
        "are now working with your team "
		"on a request from {customer} ensuring that "
        "the support representative is "
		"providing the best support possible.\n"
		"You need to make sure that the support representative "
        "is providing full"
		"complete answers, and make no assumptions."
	),
    llm=llm,
    # allow_delegation=False, is not used to allow it back to another agent for more suitable agent
    verbose=True
)

docs_scrape_tool = ScrapeWebsiteTool(
    website_url='https://docs.crewai.com/concepts/memory'
)

#Tool on the Task Level.

#Task for Support Agent for inquiry resolution
inquiry_resolution = Task(
    description=(
        "{customer} just reached out with a super important ask:\n"
	    "{inquiry}\n\n"
        "{person} from {customer} is the one that reached out. "
		"Make sure to use everything you know "
        "to provide the best support possible."
		"You must strive to provide a complete "
        "and accurate response to the customer's inquiry."
    ),
    expected_output=(
	    "A detailed, informative response to the "
        "customer's inquiry that addresses "
        "all aspects of their question.\n"
        "The response should include references "
        "to everything you used to find the answer, "
        "including external data or solutions. "
        "Ensure the answer is complete, "
		"leaving no questions unanswered, and maintain a helpful and friendly "
		"tone throughout."
    ),
	tools=[docs_scrape_tool],
    agent=support_agent,
)

#Task for Support Quality Assurance Specialist for resolution review
quality_assurance_review = Task(
    description=(
        "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
        "Ensure that the answer is comprehensive, accurate, and adheres to the "
		"high-quality standards expected for customer support.\n"
        "Verify that all parts of the customer's inquiry "
        "have been addressed "
		"thoroughly, with a helpful and friendly tone.\n"
        "Check for references and sources used to "
        " find the information, "
		"ensuring the response is well-supported and "
        "leaves no questions unanswered."
    ),
    expected_output=(
        "A final, detailed, and informative response "
        "ready to be sent to the customer.\n"
        "This response should fully address the "
        "customer's inquiry, incorporating all "
		"relevant feedback and improvements.\n"
		"Don't be too formal, we are a chill and cool company "
	    "but maintain a professional and friendly tone throughout."
    ),
    agent=support_quality_assurance_agent,
)


# Create a crew with the agents
crew = Crew(
  agents=[support_agent, support_quality_assurance_agent],
  tasks=[inquiry_resolution, quality_assurance_review],
  verbose=True,
  memory=False,
#   embedder={
#         "provider": "google",
#         "config": {
#             "api_key": os.getenv("GOOGLE_API_KEY"),
#             "model_name": "gemini-2.0-flash-exp"
#         }
#     }
)

#run the crew
print("Starting the crew")
inputs = {
    "customer": "DeepLearningAI",
    "person": "PP",
    "inquiry": "What is memory in crewAi?"
    
            # "I need help with setting up a Crew "
            #    "and kicking it off, specifically "
            #    "how can I add memory to my crew when using google AI or Deepseek Ai LLM API keys? "
            #    "Can you provide guidance?"
}
result = crew.kickoff(inputs=inputs)
print("Crew completed")

# Extract the text content from the CrewOutput object
text_content = result.raw

# Print the result
Markdown(text_content)
