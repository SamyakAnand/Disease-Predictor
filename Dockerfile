# Use an official Python runtime as a parent image (slimmer Debian-based image)
FROM python:3.12-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# Upgrading pip and installing awscli
RUN pip install --upgrade pip && \
    pip install awscli && \
    pip install -r requirements.txt

# Make port 5000 available to the world outside this container (if needed, change the port)


# Run app.py when the container launches
CMD ["python3", "app.py"]
