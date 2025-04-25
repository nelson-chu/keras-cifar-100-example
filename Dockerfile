# Use tensorflow/tensorflow:2.14.0-gpu as the base image
FROM tensorflow/tensorflow:2.14.0-gpu

# Set the working directory
WORKDIR /app

# Add files into the container
ADD . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Execute cifar-100-example.py
CMD ["python", "cifar-100-example.py"]
