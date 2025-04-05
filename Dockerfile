FROM python:3.10

RUN apt-get update && apt-get install -y curl libpq-dev python3-dev \
    && curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
	pip install -r $HOME/app/requirements.txt


# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app


COPY --chown=user . .



# COPY ./start.sh /app/start.sh

RUN chmod +x start.sh

# ENV PATH=/root/.local/bin:$PATH

CMD ["./start.sh"]

