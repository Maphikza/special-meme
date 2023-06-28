from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
chain = load_summarize_chain(llm=llm, chain_type="stuff")

with open("schedule.txt", "r") as file:
    text = file.read()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=0)
texts = text_splitter.split_text(text)
docs = [Document(page_content=t) for t in texts]
result = chain.run(docs)
print(result)
