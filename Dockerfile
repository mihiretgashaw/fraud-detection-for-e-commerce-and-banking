FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter for notebook execution
RUN pip install jupyter nbconvert

# Copy the app files
COPY . .

# Ensure run_all.sh is executable
RUN chmod +x run_all.py

# Set entrypoint
CMD ["python","./run_all.py"]
