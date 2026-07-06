from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# Document loading - transcript data load from youtube
video_id = "LPZh9BOjkQs"
api = YouTubeTranscriptApi()

try: 
    transcript_list = api.list(video_id)
    transcript = transcript_list.find_transcript(['en'])
    data = transcript.fetch()

    text = " ".join(chunk.text for chunk in data)
    
except TranscriptsDisabled:
    print("No captions are available for this particular video")


# split these transcript into small chunks for better processing.
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)
chunks = splitter.create_documents([text])

#converting this chunks into embedding and store in vector store(FAISS)
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
vector_Store = FAISS.from_documents(chunks, embeddings)


# Creating retriever
retriever = vector_Store.as_retriever(search_kwargs={'k':3})

result = retriever.invoke('what is llm and why it existed')


# llm model creation

llm = ChatGoogleGenerativeAI(
    model = 'gemini-3.5-flash',
    temperature = 0.2
)

prompt = PromptTemplate(
    template = """ You are helpfull assistant.
                    Answer only from the provided transcript/text context.
                    If the content is insufficient, just say you don't know.
                    {context}
                    Question:{question}""",
    input_variables=['context', 'question']
)

question = 'is the topic of llms are discussed in this video?is yes then what was discussed'
retrieved_docs = retriever.invoke(question)

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

main_chain.invoke('can you summarise the video?')



