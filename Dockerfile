# -------------------------------
# 1️⃣ Use lightweight Python base
# -------------------------------
FROM python:3.10-slim

# -------------------------------
# 2️⃣ Set working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# 3️⃣ Copy all project files
# -------------------------------
COPY . /app

# -------------------------------
# 4️⃣ Install dependencies
# -------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# 5️⃣ Set environment variable so Flask knows it's in Docker
# -------------------------------
ENV RUNNING_IN_DOCKER=true

# -------------------------------
# 6️⃣ Expose Flask port
# -------------------------------
EXPOSE 5000

# -------------------------------
# 7️⃣ Start the Flask app
# -------------------------------
CMD ["python", "app.py"]
