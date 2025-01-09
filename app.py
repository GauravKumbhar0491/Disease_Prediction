from flask import Flask, request, jsonify, make_response, render_template
from flask_cors import CORS
import json
from chat import ChatBot

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
bot = ChatBot()

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Render an HTML template

@app.route('/api/chatbot', methods=['POST'])
def handle_chat():
    data = json.loads(request.data.decode('utf-8'))
    user_message = data['message']
    bot_response = bot.take_response(user_message);
    resp = make_response(jsonify({'response': bot_response}), 200)

    # return the Bot's responce
    return resp

if __name__ == '__main__':
    app.run(debug=True)
