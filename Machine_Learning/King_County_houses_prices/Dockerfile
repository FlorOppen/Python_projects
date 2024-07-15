# Use a slim Python image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy only the necessary files
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY rfr_model.joblib rfr_model.joblib

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
