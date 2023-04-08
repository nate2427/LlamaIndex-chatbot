from langchain import OpenAI
from llama_index import (
    download_loader,
    GPTSimpleVectorIndex,
    ServiceContext,
    LLMPredictor,
    PromptHelper,
    QuestionAnswerPrompt,
    ComposableGraph,
    GPTListIndex,
)
from llama_index.langchain_helpers.agents import (
    LlamaToolkit,
    create_llama_chat_agent,
    IndexToolConfig,
    LlamaIndexTool,
    GraphToolConfig,
)
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from typing import List
from langchain.chains.conversation.memory import ConversationBufferMemory

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from contextlib import contextmanager, redirect_stdout
from io import StringIO
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# from streamlit import cache_resource
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

PubMedReader = download_loader("PubmedReader")
loader = PubMedReader()


search_queries = ["fitness supplement"]

chat_chain = None
initialized = False


def define_toolkit(indexes: List[GPTSimpleVectorIndex]):
    summaries = [f"pubmed index {i}" for i in range(len(indexes))]
    llm_predictor = LLMPredictor(
        llm=OpenAI(temperature=0, max_tokens=1000, openai_api_key=OPENAI_API_KEY)
    )

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    # allows us to synthesize information across each index
    graph = ComposableGraph.from_indices(
        GPTListIndex,
        indexes,
        index_summaries=summaries,
        service_context=service_context,
    )

    # [optional] save to disk
    graph.save_to_disk("pubmed_graph_v3.json")

    decompose_transform = DecomposeQueryTransform(llm_predictor, verbose=True)
    graph = ComposableGraph.load_from_disk(
        "pubmed_graph_v3.json", service_context=service_context
    )
    # define query configs for graph
    query_configs = [
        {
            "index_struct_type": "simple_dict",
            "query_mode": "default",
            "query_kwargs": {
                "similarity_top_k": 1,
                # "include_summary": True
            },
            "query_transform": decompose_transform,
        },
        {
            "index_struct_type": "list",
            "query_mode": "default",
            "query_kwargs": {"response_mode": "tree_summarize", "verbose": True},
        },
    ]
    # graph config
    graph_config = GraphToolConfig(
        graph=graph,
        name=f"Graph",
        description="useful for when you want to answer queries about supplement research from pubmed",
        query_configs=query_configs,
        tool_kwargs={"return_direct": True},
    )
    index_configs = []
    for index in indexes:
        tool_config = IndexToolConfig(
            index=index,
            description=f"useful for when you want to answer queries about supplements and fitness.",
            tool_kwargs={"return_direct": True},
            index_query_kwargs={"similarity_top_k": 3},
            name=f"pubmed_index_v3",
        )
        index_configs.append(tool_config)

    tool_kit = LlamaToolkit(index_configs=index_configs, graph_configs=[graph_config])
    return tool_kit


def set_up_llama_chatbot_agent(indexes, memory):
    global initialized
    if initialized == True:
        return
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        model_name="gpt-3.5-turbo",
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
    )
    toolkit = define_toolkit(indexes)
    llama_chat_agent_chain = create_llama_chat_agent(
        toolkit=toolkit, memory=memory, llm=llm, verbose=True
    )
    # llama_chat_agent_chain.run
    chat_chain = llama_chat_agent_chain
    return chat_chain, memory


def load_papers_from_pubmed():
    documents = loader.load_data(search_query="fitness supplements", max_results=10)
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, temperature=0, model_name="text-ada-002"
        )
    )
    # define prompt helper
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context
    )
    index.save_to_disk("pubmed_index_v3.json")
    print("saved!")
    return "docs are loaded! ask awaaaaay!"


def chat(query):
    return chat_chain.run(input=query)


@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield


@st.cache_resource()
def initialize_chatbot():
    memory = ConversationBufferMemory(memory_key="chat_history")
    chat_chain, memory = set_up_llama_chatbot_agent(
        [GPTSimpleVectorIndex.load_from_disk("pubmed_index_v3.json")], memory
    )
    return chat_chain, memory


chat_chain, memory = initialize_chatbot()

print("memory", memory)

st.header("jim AI")
st.subheader(
    "a chatbot for fitness enthusiasts. Ask jim about fitness supplements, nutrition, and more!"
)
user_query = st.text_input("Ask jim")

output = st.empty()

if st.button("ask"):
    # with st_capture(output.write):
    #     chat(user_query)
    st.write(chat(user_query))


if st.button("load"):
    load_papers_from_pubmed()
    st.markdown("docs are loaded! ask awaaaaay!")
