FROM python:3.10

RUN apt-get update && apt-get install -y \
	curl libpq-dev python3-dev \
	&& curl -fsSL https://ollama.com/install.sh | sh \
	&& rm -rf /var/lib/apt/lists/*

# azure cli
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt


# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user && \
    mkdir -p /home/user/app && \
    chown -R user:user /home/user

# Switch to the "user" user
USER user
WORKDIR $HOME/app

COPY --chown=user:user . .


RUN chmod +x start.sh

ENV KEY_VAULT_URL=https://comp7940keyvault.vault.azure.net/
CMD ["./start.sh"]

