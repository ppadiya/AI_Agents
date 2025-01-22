# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew, LLM, Process
from openai import OpenAI
from dotenv import load_dotenv
import os
from IPython.display import Markdown, display
from crewai_tools import (
    ScrapeWebsiteTool,
    WebsiteSearchTool,
    FileReadTool,
    MDXSearchTool
)
from crewai.tools import BaseTool
from IPython.display import Markdown, display
from pydantic import BaseModel
from langchain_openai import ChatOpenAI


# Load environment variables from .env file
load_dotenv()

# Access the DEEPSEEK_API_KEY environment variable

llm = LLM(
    model="deepseek/deepseek-chat",
    temperature=0.7,
    base_url="https://api.deepseek.com/v1",
    api_key= os.getenv("DEEPSEEK_API_KEY")
)

# Initialize the tools
read_resume = FileReadTool(file_path='./fake_resume.md')
scrape_tool = ScrapeWebsiteTool()
search_tool = WebsiteSearchTool(
    config=dict(
        llm=dict(
            provider="google", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="gemini-2.0-flash-exp",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)

semantic_search_resume = MDXSearchTool(
    config=dict(
        llm=dict(
            provider="google",
            config=dict(
                model="gemini-2.0-flash-exp",
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
            ),
        ),
    ),
    mdx='./fake_resume.md' 
)


# from IPython.display import Markdown, display
display(Markdown("./fake_resume.md"))

# Agent 1: Job Researcher
researcher = Agent(
    role="Product Job Researcher",
    goal="Make sure to do amazing analysis on "
         "job posting to help job applicants",
    tools = [scrape_tool, search_tool],
    llm=llm,
    verbose=True,
    backstory=(
        "As a Job Researcher, your prowess in "
        "navigating and extracting critical "
        "information from job postings is unmatched."
        "Your skills help pinpoint the necessary "
        "qualifications and skills sought "
        "by employers, forming the foundation for "
        "effective application tailoring."
    )
)

# Agent 2: Job Profiler
profiler = Agent(
    role="Personal Profiler for Product Managers",
    goal="Do increditble research on job applicants "
         "to help them stand out in the job market",
    tools = [scrape_tool, search_tool,
             read_resume, semantic_search_resume],
    llm=llm,
    verbose=True,
    backstory=(
        "Equipped with analytical prowess, you dissect "
        "and synthesize information "
        "from diverse sources to craft comprehensive "
        "personal and professional profiles, laying the "
        "groundwork for personalized resume enhancements."
    )
)

# Agent 3: Resume Strategist
resume_strategist = Agent(
    role="Resume Strategist for Product Managers",
    goal="Find all the best ways to make a "
         "resume stand out in the job market.",
    tools = [scrape_tool, search_tool,
             read_resume, semantic_search_resume],
    llm=llm,
    verbose=True,
    backstory=(
        "With a strategic mind and an eye for detail, you "
        "excel at refining resumes to highlight the most "
        "relevant skills and experiences, ensuring they "
        "resonate perfectly with the job's requirements."
    )
)

# Agent 4: Interview Preparer
interview_preparer = Agent(
    role="Product Management Interview Preparer",
    goal="Create interview questions and talking points "
         "based on the resume and job requirements",
    tools = [scrape_tool, search_tool,
             read_resume, semantic_search_resume],
    llm=llm,
    verbose=True,
    backstory=(
        "Your role is crucial in anticipating the dynamics of "
        "interviews. With your ability to formulate key questions "
        "and talking points, you prepare candidates for success, "
        "ensuring they can confidently address all aspects of the "
        "job they are applying for."
    )
)

# Task for Researcher Agent: Extract Job Requirements
research_task = Task(
    description=(
        "Analyze the job posting URL provided ({job_posting_url}) "
        "to extract key skills, experiences, and qualifications "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements."
    ),
    expected_output=(
        "A structured list of job requirements, including necessary "
        "skills, qualifications, and experiences."
    ),
    agent=researcher,
    async_execution=True
)

# Task for Profiler Agent: Compile Comprehensive Profile
profile_task = Task(
    description=(
        "Compile a detailed personal and professional profile "
        "using the personal write-up ({personal_writeup}) and the information from the provided resumes. "
        "Utilize tools to extract and synthesize information from these sources."
    ),
    expected_output=(
        "A comprehensive profile document that includes skills, "
        "project experiences, contributions, interests, and "
        "communication style."
    ),
    agent=profiler,
    async_execution=True
)

# Task for Resume Strategist Agent: Align Resume with Job Requirements
resume_strategy_task = Task(
    description=(
        "Using the profile and job requirements obtained from "
        "previous tasks, tailor the resume to highlight the most "
        "relevant areas. Employ tools to adjust and enhance the "
        "resume content. Make sure this is the best resume even but "
        "don't make up any information. Update every section, "
        "inlcuding the initial summary, work experience, skills, "
        "and education. All to better reflect the candidates "
        "abilities and how it matches the job posting."
    ),
    expected_output=(
        "An updated resume that effectively highlights the candidate's "
        "qualifications and experiences relevant to the job."
    ),
    output_file="tailored_resume.md",
    context=[research_task, profile_task],
    agent=resume_strategist
)

# Task for Interview Preparer Agent: Develop Interview Materials
interview_preparation_task = Task(
    description=(
        "Create a set of potential interview questions and talking "
        "points based on the tailored resume and job requirements. "
        "Utilize tools to generate relevant questions and discussion "
        "points. Make sure to use these question and talking points to "
        "help the candiadte highlight the main points of the resume "
        "and how it matches the job posting."
    ),
    expected_output=(
        "A document containing key questions and talking points "
        "that the candidate should prepare for the initial interview."
    ),
    output_file="interview_materials.md",
    context=[research_task, profile_task, resume_strategy_task],
    agent=interview_preparer
)

# Define the crew with agents and tasks
job_application_crew = Crew(
    agents=[researcher,
            profiler,
            resume_strategist,
            interview_preparer],

    tasks=[research_task,
           profile_task,
           resume_strategy_task,
           interview_preparation_task],

    llm=llm,
    verbose=True
)

# Run the crew
job_application_inputs = {
    'job_posting_url': 'https://wise.jobs/job/product-manager-payments-apac-in-singapore-jid-1105',
    'personal_writeup': """As a seasoned and versatile Product Manager and Solutions Leader, I specialize in driving revenue growth, enhancing 
customer experience, and developing innovative solutions across diverse industries, particularly in fintech and 
payments. My expertise includes spearheading cross-functional teams, collaborating with senior stakeholders, and 
implementing strategic initiatives globally. I possess a strong ability to navigate complex market landscapes, 
consistently exceeding organizational targets and positioning myself as an asset to any dynamic organization."""
}

### this execution will take a few minutes to run
result = job_application_crew.kickoff(inputs=job_application_inputs)

# Display the result
display(Markdown("./tailored_resume.md"))
display(Markdown("./interview_materials.md"))