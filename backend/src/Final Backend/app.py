from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import generate_answer   # import from step 1

app = Flask(__name__)
CORS(app)  # allow frontend to call backend

@app.route("/", methods=["GET"])
def home():
    return "✅ PU Assistant backend is running!"

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message')
    print(f"Received query: {query}")

    try:
        answer = generate_answer(query)
    except Exception as e:
        print(f"❌ Error: {e}")
        answer = "Sorry, something went wrong. Please visit the admin office."

    print(f"Generated answer: {answer}")
    return jsonify({"reply": answer})

if __name__ == '__main__':
    app.run(port=5000)
