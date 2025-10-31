# Use a lightweight official Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list into the container
COPY requirements.txt .

# Install Python packages listed in requirements.txt
RUN pip install -r requirements.txt

# Copy all source code (including data folder) into the container
COPY src ./src

# Run the model script, telling it where to find the data files
CMD ["python", "src/run/model.py", "--data_dir", "src/data"]
