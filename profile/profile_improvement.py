from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# Define chat-style template
template = """
You are a career advisor helping a job seeker improve their profile.

Profile Description:
{profile_description}

Target Job Title:
{target_job_title}

Provide specific, actionable recommendations on how this person can enhance their skills, experience, or education to increase their chances of getting hired as a {target_job_title}.
"""

prompt = ChatPromptTemplate.from_template(template)

# Use the modern, recommended LLM
llm = OllamaLLM(model="mistral")

# Chain: prompt → LLM
chain = prompt | llm

# Invoke with chat-friendly input
response = chain.invoke({
    "profile_description": "Recent computer science graduate with internship experience in frontend development using React.",
    "target_job_title": "UI/UX Designer"
})



print(response)
