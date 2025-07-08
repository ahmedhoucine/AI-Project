from flask import Flask, render_template, request
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import markdown

app = Flask(__name__)

# LangChain prompt and LLM setup
template = """
You are a career advisor helping a job seeker improve their profile.

Profile Description:
{profile_description}

Target Job Title:
{target_job_title}

Provide specific, actionable recommendations on how this person can enhance 
their skills, experience, or education to increase their chances of getting hired as a {target_job_title}.
"""

prompt = ChatPromptTemplate.from_template(template)
llm = OllamaLLM(model="mistral")
chain = prompt | llm

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        profile = request.form["profile_description"]
        title = request.form["target_job_title"]
        response = chain.invoke({
            "profile_description": profile,
            "target_job_title": title
        })
        result = markdown.markdown(str(response))
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
