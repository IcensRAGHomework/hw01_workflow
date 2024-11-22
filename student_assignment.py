import json
import traceback

from model_configurations import get_model_configuration
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_core.tools import tool

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

print("AZURE_OPENAI_GPT4O_ENDPOINT:", os.getenv('AZURE_OPENAI_GPT4O_ENDPOINT'))
print("AZURE_OPENAI_GPT4O_KEY:", os.getenv('AZURE_OPENAI_GPT4O_KEY'))
print("AZURE_OPENAI_GPT4O_DEPLOYMENT_CHAT:", os.getenv('AZURE_OPENAI_GPT4O_DEPLOYMENT_CHAT'))
print("AZURE_OPENAI_GPT4O_VERSION:", os.getenv('AZURE_OPENAI_GPT4O_VERSION'))

def generate_node(question, use_tools_call=False):
    llm = AzureChatOpenAI(
        model = gpt_config['model_name'],
        deployment_name = gpt_config['deployment_name'],
        openai_api_key = gpt_config['api_key'],
        openai_api_version = gpt_config['api_version'],
        azure_endpoint = gpt_config['api_base'],
        temperature= gpt_config['temperature']
    )
        
    system = """
            You are a helpful assistant.
            Please respond in JSON format.
            The top-level key must be 'Result', and its value must be a list of objects.
            Each object should contain two keys: 'Date' (the date of the holiday) and 'Date_Name' (the name of the holiday).
            """
    try:
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Answer the question based on the below description and only use English or traditional Chinese. Please respond in JSON format."),
                ("human", f"Question: {question} \n "),
            ]
        )

        if use_tools_call:
            llm_with_tools = llm
        else:
            llm_with_tools = llm

        json_llm = llm_with_tools.bind(response_format={"type": "json_object"})
        rag_chain = LLMChain(
            llm=json_llm,
            prompt=answer_prompt,
            output_parser=StrOutputParser()
        )

        # RAG generation
        answer = rag_chain.invoke({"question": question})
        return answer
    except Exception as e:
        traceback_info = traceback.format_exc()
        print(traceback_info)
        
def add(a, b):
    """Return the sum of a and b.  """
    return a + b

def sub(a, b):
    """Return the sum of a and b.   """
    return a - b
