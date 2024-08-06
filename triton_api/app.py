import os
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

TRITON_URL = os.getenv(
    "TRITON_URL", "http://localhost:8000/v2/models/ensemble/generate"
)
JSON_PREFIX = '{\n"answer": '


def answer(user_query: str) -> dict:
    payload = {
        "text_input": f"respond in json format and use only the key answer. q: {user_query} \n```json\n {JSON_PREFIX}",
        "max_tokens": 400,
        "bad_words": "",
        "stop_words": "```",
        "pad_id": 2,
        "end_id": 2,
    }
    response = requests.post(TRITON_URL, json=payload)
    return response.json()


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400

    result = JSON_PREFIX + answer(user_query)["text_output"]
    result = result[: result.rfind("```")]
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
