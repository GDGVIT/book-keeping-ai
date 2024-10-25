# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and necessary directories into the container
COPY . .

# Expose the port your app runs on
EXPOSE 5000

# Define the command to run your app
CMD ["python", "main.py"]
