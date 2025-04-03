FROM python:3.10

WORKDIR /app


COPY . .

RUN apt-get update && apt-get install -y curl libpq-dev python3-dev \
    && curl -fsSL https://ollama.com/install.sh | sh

RUN pip install --upgrade pip
RUN pip install --user -r requirements.txt


COPY ./start.sh /app/start.sh

RUN chmod +x /app/start.sh

ENV PATH=/root/.local/bin:$PATH

CMD ["/app/start.sh"]

