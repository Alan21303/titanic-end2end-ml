# Use slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (caching optimization)
COPY requirements-docker.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
