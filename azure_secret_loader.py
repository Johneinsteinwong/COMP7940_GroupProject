from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os

def load_secrets():
    # Initialize Key Vault client
    credential = DefaultAzureCredential()
    client = SecretClient(
        vault_url=os.environ['KEY_VAULT_URL'],
        credential=credential
    )
    
    return {
        "CHATGPT-ACCESS-TOKEN": client.get_secret("CHATGPT-ACCESS-TOKEN").value,
        "CHATGPT-APIVERSION": client.get_secret("CHATGPT-APIVERSION").value,
        "CHATGPT-BASICURL": client.get_secret("CHATGPT-BASICURL").value,
        "CHATGPT-MODELNAME": client.get_secret("CHATGPT-MODELNAME").value,
        "CONNECTION-STRING": client.get_secret("CONNECTION-STRING").value,
        "INDEX-NAME": client.get_secret("INDEX-NAME").value,
        "TELEGRAM-ACCESS-TOKEN": client.get_secret("TELEGRAM-ACCESS-TOKEN").value,
        "TWITTER-ACCESS-TOKEN": client.get_secret("TWITTER-ACCESS-TOKEN").value,
        "TWITTER-ACCESS-TOKEN-SECRET": client.get_secret("TWITTER-ACCESS-TOKEN-SECRET").value,
        "TWITTER-API-KEY": client.get_secret("TWITTER-API-KEY").value,
        "TWITTER-API-SECRET-KEY":client.get_secret("TWITTER-API-SECRET-KEY").value,
        "TWITTER-BEARER-TOKEN": client.get_secret("TWITTER-BEARER-TOKEN").value,
    }

