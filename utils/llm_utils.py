import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4", temperature=0)
parser = JsonOutputParser()

def call_llm_json(prompt_or_str, variables=None):
    try:
        response = llm.invoke(prompt_or_str.format(**variables))
        import json, re
        json_text = re.search(r'\{.*\}|\[.*\]', response.content, re.DOTALL)
        if json_text:
            return json.loads(json_text.group())
        else:
            return {}
    except Exception as e:
        return {"error": str(e)}

def call_llm_json_verbose(prompt_or_str, variables=None):
    return call_llm_json(prompt_or_str, variables)