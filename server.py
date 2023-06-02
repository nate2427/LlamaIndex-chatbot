from flask import Flask, request, jsonify
from flask_cors import CORS
from main import initialize_chatbot, chat, load_videos_from_youtube

app = Flask(__name__)
CORS(app)
app.config["chat_chain"] = None


def initialize_chat_chain():
    # Initialize the chatbot chain here
    chat_chain, memory = initialize_chatbot()
    return chat_chain, memory


@app.before_first_request
def init_app():
    chat_chain, memory = initialize_chat_chain()
    app.config["chat_chain"] = chat_chain
    app.config["memory"] = memory


@app.route("/chat", methods=["POST"])
def handle_chat():
    chat_chain = app.config["chat_chain"]
    # Handle chat request here
    return jsonify({"msg": chat(chat_chain, request.json["query"])})


@app.route("/history", methods=["GET"])
def get_chat_history():
    chat_obj = app.config["memory"].load_memory_variables(
        {}
    )
    chat_info = chat_obj["chat_history"]
    return chat_info


@app.route('/train-agent', methods=['POST'])
def train_agent():
    youtube_video_urls = request.json.get('youtube_video_urls')
    load_videos_from_youtube(youtube_video_urls)
    # add code for training the agent on the new material here
    return "Agent trained on new material!"


# start flask server
if __name__ == "__main__":
    app.run()
