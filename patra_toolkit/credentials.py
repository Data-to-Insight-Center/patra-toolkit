import requests


def get_huggingface_credentials(patra_server_url: str, timeout: int = 10) -> dict:
    """
    Retrieves Hugging Face credentials.

    Parameters:
    - patra_server_url (str): URL of the Patra server.
    - timeout (int): Timeout for the request in seconds.

    Returns:
    - dict: Hugging Face credentials with 'username' and 'token'

    Raises:
    - Exception: If the server response is invalid.
    """
    hf_creds_url = f"{patra_server_url}/get_hf_credentials"
    headers = {"Content-Type": "application/json"}
    response = requests.get(hf_creds_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    creds = response.json()

    if "username" not in creds or "token" not in creds:
        raise Exception("Invalid Hugging Face credentials response from server.")
    return creds
