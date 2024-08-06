import requests
import json


def query_api(user_query: str):
    url = "http://localhost:5000/query"
    payload = {"query": user_query}

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        response_json = response.json()
        return json.dumps(response_json, indent=4)
    else:
        return f"Error: {response.status_code}\n{response.text}"


if __name__ == "__main__":
    user_queries = [
        "Explain Bernoulli's principle.",
        "What is Newton's second law?",
        "Define the theory of relativity in simple terms.",
        "Briefly explain quantum entanglement.",
        "What is Schrodinger's cat thought experiment?",
        "Who is the current president of the USA?",
        "When are the next presidential elections in the USA?",
    ]

    with open("query_results.txt", "w") as file:
        for i, query in enumerate(user_queries, start=1):
            result = query_api(query)
            print(f"Query {i}: '{query}'")
            file.write(f"Query {i}: '{query}' Response:\n")
            file.write(result + "\n")
            file.write("=" * 70 + "\n")  # Clear separator
