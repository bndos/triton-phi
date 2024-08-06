# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy the pyproject.toml and poetry.lock files to the container
COPY pyproject.toml poetry.lock app.py ./

# Install the dependencies
RUN poetry install --no-dev

# Expose the port the app runs on
EXPOSE 5000

# Run the Flask server
CMD ["poetry", "run", "python", "app.py"]
