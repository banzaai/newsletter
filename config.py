from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import getpass
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

if not os.getenv("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter your Azure OpenAI API key: ")

if not os.getenv("AZURE_OPENAI_ENDPOINT"):
    os.environ["AZURE_OPENAI_ENDPOINT"] = input("Enter your Azure OpenAI endpoint (e.g., https://your-resource-name.openai.azure.com): ")

if not os.getenv("DEPLOYMENT_NAME"):
    os.environ["DEPLOYMENT_NAME"] = input("Enter your Azure OpenAI deployment name: ")

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
    temperature=float(os.getenv("TEMPERATURE", 0.6)),
)

embeddings = AzureOpenAIEmbeddings(
    model = os.getenv("EMBEDDINGS_MODEL")
    )