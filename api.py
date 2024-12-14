from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load your trained chatbot model and tokenizer
model = AutoModelForCausalLM.from_pretrained("/home/saber/Desktop/pfe/chat/chat.py")
tokenizer = AutoTokenizer.from_pretrained("/home/saber/Desktop/pfe/chat/chat.py")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')

    # Tokenize the user's input and generate a response
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs)
    bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the chatbot response as a JSON
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
