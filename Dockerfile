FROM apache/airflow:2.10.4-python3.11

# Java is required by PySpark. Debian-based image, so apt installs it as root.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# IST everywhere (plan section 2). Airflow's own timezone setting goes in docker-compose.yml.
ENV TZ=Asia/Kolkata

# Make `from src import config` work inside the container
ENV PYTHONPATH=/opt/airflow

# Python dependencies, installed as the airflow user
USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt