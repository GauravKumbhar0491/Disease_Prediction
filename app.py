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
    return render_template('index.html')  # Ensure index.html is in the templates/ folder

# API route for chatbot interaction
@app.route('/api/chatbot', methods=['POST'])
def handle_chat():
    try:
        # Parse incoming JSON data
        data = json.loads(request.data.decode('utf-8'))
        user_message = data.get('message')

        # Check if message is provided
        if not user_message:
            return make_response(jsonify({'error': 'Message is required'}), 400)

        # Generate chatbot response
        bot_response = bot.take_response(user_message)
        return make_response(jsonify({'response': bot_response}), 200)
    
    except json.JSONDecodeError:
        return make_response(jsonify({'error': 'Invalid JSON'}), 400)
    except Exception as e:
        return make_response(jsonify({'error': str(e)}), 500)

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

