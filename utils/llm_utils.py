import os
import json
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0.3)

def call_llm_json(prompt_or_str, variables=None):
    try:
        # Running the LLM call
        response = llm.invoke(prompt_or_str.format(**(variables or {})))
        content = response.content.strip()

        # Extracting clean JSON block from any junk response
        match = re.search(r'\{[\s\S]+\}', content)
        if match:
            json_block = match.group()
            try:
                return json.loads(json_block)
            except json.JSONDecodeError as e:
                print("⚠️ JSON Decode Error:", e)
                print("⚠️ Raw JSON:\n", json_block)
                return {}
        else:
            print("⚠️ No JSON block found in LLM output.")
            print("Raw content:", content)
            return {}

    except Exception as e:
        print("⚠️ Fatal Exception while calling LLM:", e)
        return {}

def call_llm_json_verbose(prompt_or_str, variables=None):
    return call_llm_json(prompt_or_str, variables)
