# Use a Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install necessary Python dependencies
RUN pip install --upgrade pip && \
    pip install pandas matplotlib seaborn

# Copy the Python script into the container
COPY analysis_script.py /app/

# Set the environment variables for dataset paths (these will be passed when running the container)
ENV DATASET_PATH=/app/dataset
ENV TRAIN_PATH=${DATASET_PATH}/train
ENV VAL_PATH=${DATASET_PATH}/val

# Run the analysis script
CMD ["python", "analysis_script.py"]
