FROM python:3.10-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock app.py ./

RUN poetry install --no-dev

EXPOSE 5000

CMD ["poetry", "run", "python", "app.py"]
