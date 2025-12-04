# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config (disable browser opening inside container)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRS=false
ENV STREAMLIT_SERVER_PORT=8501

# Run the app
CMD ["streamlit", "run", "app.py"]
