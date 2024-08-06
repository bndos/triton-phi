import os
import json
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

TRITON_URL = os.getenv(
    "TRITON_URL", "http://localhost:8000/v2/models/ensemble/generate"
)
JSON_PREFIX = '{\n"answer": '


def gen_multi_answers(user_query: str) -> dict:
    prompts = [
        (
            f"Respond in json format and only use the key answer. q: {user_query} \n```json\n{JSON_PREFIX}",
            JSON_PREFIX,
        ),
        (
            f"Respond in json format and only use the key answer. q: {user_query} \n```json\n",
            "",
        ),
        (user_query, None),
    ]

    results = {}
    for i, (prompt, json_prefix) in enumerate(prompts):
        results["prompt" + str(i)] = answer(prompt, json_prefix)

    return results


def answer(user_query: str, json_prefix: str | None) -> dict:
    payload = {
        "text_input": user_query,
        "max_tokens": 400,
        "bad_words": "",
        "stop_words": "```",
        "pad_id": 2,
        "end_id": 2,
    }
    response = requests.post(TRITON_URL, json=payload)
    result = response.json()
    result = result["text_output"]
    if json_prefix is not None:
        result = json_prefix + result
        result = result[: result.rfind("```")]
    else:
        result = json.dumps(result)
        result = '{"answer": ' + f"{result}" + "}".strip()

    result = json.loads(result)
    return result


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400

    result = gen_multi_answers(user_query)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
