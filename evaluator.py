from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.smith import RunEvalConfig
from langchain_google_genai import GoogleGenerativeAI
from langsmith import Client

from chat_logic import get_agent_prompt, get_llm, get_tools
from utils import config


# Define your runnable or chain below.
def get_agent():
    prompt = get_agent_prompt()
    llm = get_llm()
    tools = get_tools()

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=5,
        max_execution_time=90.0,
        early_stopping_method="generate",
        return_intermediate_steps=True,
    ).with_config({"run_name": "Agent"})
    return agent_executor


agent = get_agent()
eval_llm = GoogleGenerativeAI(
    google_api_key=config.AI_STUDIO_API_KEY, model="gemini-pro", temperature=0.3
)

# Define the evaluators to apply
eval_config = RunEvalConfig(
    evaluators=["cot_qa"],
    custom_evaluators=[],
    eval_llm=eval_llm,
    input_key="input",
    prediction_key="output",
    reference_key="reference",
)

client = Client()
chain_results = client.run_on_dataset(
    dataset_name="answer-eval-dataset",
    llm_or_chain_factory=agent,
    evaluation=eval_config,
    project_name="answer-eval-test-3",
    concurrency_level=1,
    verbose=True,
)
