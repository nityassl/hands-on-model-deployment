#Dockerfile

# Use existing base image
FROM python:3.8

# Set work directory
WORKDIR /app

# Copy and install requirements/dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# Prepare data and train model
COPY ./train.py .
COPY ./data ./data
RUN mkdir ./pipeline
RUN python ./train.py

# Set environments
ENV PORT=80
ENV HOST=*:80

# Copy remaining artifacts
COPY ./app.py .
COPY ./wsgi.py .
COPY ./dto ./dto
COPY ./templates ./templates

# Define run command
CMD ["python", "wsgi.py"]