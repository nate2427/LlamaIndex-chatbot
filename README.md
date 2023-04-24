# Building a Consciousness from YouTube Videos

This is a chatbot that allows you to build a consciousness from YouTube videos. The chatbot is built using Python and Flask, and uses the LangChain library for natural language processing (NLP) and conversational AI. The chatbot is trained on transcripts of YouTube videos, and can answer questions related to the content of those videos.

Setup
To run the chatbot, you'll need to install Python and the required dependencies. You can do this using the following commands:

📋 Copy code
pip install -r requirements.txt
Usage
To use the chatbot, you can run the Flask server using the following command:

📋 Copy code
python server.py
Once the server is running, you can interact with the chatbot using the /chat endpoint. To send a message to the chatbot, send a POST request to the /chat endpoint with JSON data containing the message in the query field:

📋 Copy code
curl -X POST http://localhost:5000/chat -H 'Content-Type: application/json' -d '{"query": "What is the capital of France?"}'
The chatbot will respond with a message containing the answer to your question.

Training
To train the chatbot on new material, you can use the /train-agent endpoint. Send a POST request to the /train-agent endpoint with JSON data containing a list of YouTube video URLs in the youtube_video_urls field:

📋 Copy code
curl -X POST http://localhost:5000/train-agent -H 'Content-Type: application/json' -d '{"youtube_video_urls": ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]}'
The chatbot will download the transcripts of the YouTube videos and use them to train a new model. You can then interact with the chatbot using the new model to answer questions related to the content of the new videos.

Chat History
To view the chat history, you can use the /history endpoint. Send a GET request to the /history endpoint to retrieve the chat history:

📋 Copy code
curl -X GET http://localhost:5000/history
The chat history will be returned as a JSON object containing the messages sent and received by the chatbot.
