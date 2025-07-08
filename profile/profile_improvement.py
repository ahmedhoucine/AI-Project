#ollama run mistral(start ollama )
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# Define chat-style template
template = """
You are a career advisor helping a job seeker improve their profile.

Profile Description:
{profile_description}

Target Job Title:
{target_job_title}

Provide specific, actionable recommendations on how this person can enhance 
their skills, experience, or education to increase their chances of getting hired as a {target_job_title}.
"""

# Create the prompt template
prompt = ChatPromptTemplate.from_template(template)

# Initialize the model
llm = OllamaLLM(model="mistral")

# Combine prompt and model into a chain
chain = prompt | llm

# Get user input
profile_description = input("Enter your profile description: ")
target_job_title = input("Enter your target job title: ")

# Invoke the chain with user input
response = chain.invoke({
    "profile_description": profile_description,
    "target_job_title": target_job_title
})

# Print the response
print(response)
