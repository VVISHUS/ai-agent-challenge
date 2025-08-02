
from pydantic import BaseModel , Field
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()



primary_url =  os.getenv("primary_url")
primary_llm_key =  os.getenv("GEMINI_API_KEY")



def create_openai_client( api_key: str, base_url: str):
    """Create OpenAI client with the specified API key and base URL"""
    return OpenAI(api_key=api_key, base_url=base_url)


client = create_openai_client(primary_llm_key, primary_url)


def chat_with_llm(
    user_message: str,
    chat_history: list=None,
    get_common_system_prompt: bool = True,
    get_common_system_prompt_args: list[bool, bool] = [True, True],
    custom_system_prompt: str = None,
    response_format=None,
    temp: float = 0.7,
    max_chat_history: int = 6
) -> str:
    """
    Send message to LLM with proper context and optional structured output.

    Parameters:
    - user_message: str → user’s current input
    - chat_history: list → full session history in Streamlit style (list of dicts with 'role' and 'content')
    - get_common_system_prompt: bool → whether to use common system prompt
    - get_common_system_prompt_args: list[bool, bool] → whether to include resume & JD respectively
    - custom_system_prompt: str → override the system prompt completely if provided
    - response_format: Optional BaseModel → for structured LLM outputs
    - temp: float → sampling temperature
    - max_chat_history: int → number of most recent history turns to include
    """

    try:
        # System prompt logic
        if custom_system_prompt:
            system_prompt = custom_system_prompt
        elif get_common_system_prompt:
            system_prompt = "You are a helpful agent"
        else:
            system_prompt = (
                "You are a help bot. Respond appropriately to user queries. "
                "These may involve extracting relevant information or simple Q&A."
            )
        if(chat_history is not None):
            max_chat_history=max_chat_history if max_chat_history<len(chat_history) else int(len(chat_history)-1)
        # Trim chat history to the last `max_chat_history` items
        trimmed_history = chat_history[-max_chat_history:] if chat_history else []

        # Convert chat history (already formatted with {"role": ..., "content": ...})
        formatted_history = [{"role": msg["role"], "content": msg["content"]} for msg in trimmed_history]

        # Compose final message list
        messages = [{"role": "system", "content": system_prompt}] + formatted_history + [
            {"role": "user", "content": user_message}
        ]

        # Call LLM with or without structured output
        if response_format:
            response = client.beta.chat.completions.parse(
                model="gemini-2.0-flash",
                messages=messages,
                response_format=response_format,
                temperature=temp
            )
            return response.choices[0].message.parsed
        else:
            response = client.chat.completions.create(
                model="gemini-2.5-flash-preview-05-20",
                messages=messages,
                temperature=temp
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"[LLM ERROR]: {e}")
        return "I apologize, but I encountered a technical issue. Let's continue with the interview. Could you please repeat your last response?"



# TEsting ad LEarning Purpose
class T1(BaseModel):
    about:str=Field(description="Description of the thing asked in the question")
    code:str=Field(description="An example code snippet to give the user basic implementation")
    capabilities:str=Field(description="What user can do out of it further")


chat_with_llm(user_message="Get me about Langraph, how can" \
" I define it and my Agentic workflow in it",response_format=T1)


# agent.py
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from typing import TypedDict, Literal
import subprocess

class AgentState(TypedDict):
    step: int
    code: str
    passed: bool
    logs: str

def plan_node(state: AgentState):
    print(f"Planning for step {state['step']}")
    return state

def generate_node(state: AgentState):
    
    prompt = f"Write a parser for ICICI PDFs that outputs a DataFrame matching this schema:\n\n..."
    code = llm.invoke(prompt).content
    with open("custom_parser/icici_parser.py", "w") as f:
        f.write(code)
    return {**state, "code": code}

def test_node(state: AgentState):
    try:
        subprocess.run(["pytest", "tests/test_parser.py"], check=True)
        return {**state, "passed": True}
    except subprocess.CalledProcessError as e:
        return {**state, "passed": False, "logs": str(e)}

def fix_node(state: AgentState):
    if state["step"] >= 3 or state["passed"]:
        return "end"
    return "generate"

# Build graph 
graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("generate", generate_node)
graph.add_node("test", test_node)
graph.set_entry_point("plan")
graph.add_edge("plan", "generate")
graph.add_edge("generate", "test")
graph.add_conditional_edges("test", fix_node, {
    "generate": "generate",
    "end": END
})
graph = graph.compile()

# Run
if __name__ == "__main__":
    output = graph.invoke({"step": 0, "code": "", "passed": False, "logs": ""})






