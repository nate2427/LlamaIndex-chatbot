import os
import datetime
from langchain import OpenAI
from llama_index import (
    download_loader,
    GPTSimpleVectorIndex,
    ServiceContext,
    LLMPredictor,
    PromptHelper,
    ComposableGraph,
    GPTListIndex,
)
from llama_index.langchain_helpers.agents import (
    LlamaToolkit,
    create_llama_chat_agent,
    IndexToolConfig,
    GraphToolConfig,
)
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from typing import List
from langchain.chains.conversation.memory import ConversationBufferMemory

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from dotenv import load_dotenv

load_dotenv()
# from streamlit import cache_resource

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")

loader = YoutubeTranscriptReader()


search_queries = ["AI Youtube Video", "Nate", "Nate's YouTube Video"]

chat_chain = None
initialized = False

OpenAI(openai_api_key=OPENAI_API_KEY)


def define_toolkit(indexes: List[GPTSimpleVectorIndex]):
    summaries = [f"Nate Knowledge index {i}" for i in range(len(indexes))]
    llm_predictor = LLMPredictor(
        llm=OpenAI(temperature=0, max_tokens=1000,
                   openai_api_key=OPENAI_API_KEY)
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
    graph.save_to_disk("street_coder_nate_persona_from_youtube_graph.json")

    decompose_transform = DecomposeQueryTransform(llm_predictor, verbose=True)
    graph = ComposableGraph.load_from_disk(
        "street_coder_nate_persona_from_youtube_graph.json", service_context=service_context
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
        description="useful for answering questions about Nate or youtube.",
        query_configs=query_configs,
        tool_kwargs={"return_direct": True},
    )
    index_configs = []
    for index in indexes:
        tool_config = IndexToolConfig(
            index=index,
            description=f"useful for when you want to answer questions about Nate or youtube",
            tool_kwargs={"return_direct": True},
            index_query_kwargs={"similarity_top_k": 3},
            name=f"street_coder_nate_persona_index.json",
        )
        index_configs.append(tool_config)

    tool_kit = LlamaToolkit(index_configs=index_configs,
                            graph_configs=[graph_config])
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
    print("toolkit loaded!")
    llama_chat_agent_chain = create_llama_chat_agent(
        toolkit=toolkit, memory=memory, llm=llm, verbose=True
    )
    # llama_chat_agent_chain.run
    chat_chain = llama_chat_agent_chain
    return chat_chain, memory


def load_videos_from_youtube(youtube_video_urls, dir="./personas"):
    documents = loader.load_data(ytlinks=youtube_video_urls)
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

    # create file name with current timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"persona_index_{current_time}.json"
    file_path = os.path.join(dir, file_name)
    # this essentially allows for many bots to be created/updated because the bot is built from a dir of json file configs
    if not os.path.exists(dir):
        os.makedirs(dir)

    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context
    )
    # save index to disk with the new file name
    index.save_to_disk(file_path)
    return "docs are loaded! ask awaaaaay!"


def chat(chain, query):
    return chain.run(input=query)


def initialize_chatbot(dir="./personas"):
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=False)
    persona_indexes = []
    for file_name in os.listdir(dir):
        if file_name.endswith(".json"):
            persona_index = GPTSimpleVectorIndex.load_from_disk(
                os.path.join(dir, file_name)
            )
            persona_indexes.append(persona_index)
    chat_chain, memory = set_up_llama_chatbot_agent(persona_indexes, memory)

    return chat_chain, memory
