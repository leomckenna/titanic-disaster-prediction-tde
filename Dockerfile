# Use official lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project files into the container
COPY src ./src

# Set the command to run your model script
CMD ["python", "src/run/model.py", "--data_dir", "src/data"]
