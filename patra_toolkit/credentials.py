import logging

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

def get_github_credentials(patra_server_url: str, timeout: int = 10) -> dict:
    """
    Retrieves GitHub credentials.

    Parameters:
    - patra_server_url (str): URL of the Patra server.
    - timeout (int): Timeout for the request in seconds.

    Returns:
    - dict: GitHub credentials with 'username' and 'token'

    Raises:
    - Exception: If the server response is invalid.
    """
    gh_creds_url = f"{patra_server_url}/get_github_credentials"
    headers = {"Content-Type": "application/json"}
    response = requests.get(gh_creds_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    creds = response.json()

    if "username" not in creds or "token" not in creds:
        raise Exception("Invalid GitHub credentials response from server.")
    return creds

def get_ndp_credentials(patra_server_url: str, timeout: int = 10) -> dict:
    """
    Retrieves NDP credentials.

    Parameters:
    - patra_server_url (str): URL of the Patra server.
    - timeout (int): Timeout for the request in seconds.

    Returns:
    - dict: NDP credentials with 'api_key'

    Raises:
    - Exception: If the server response is invalid.
    """
    ndp_creds_url = f"{patra_server_url}/get_ndp_credentials"
    headers = {"Content-Type": "application/json"}
    response = requests.get(ndp_creds_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    creds = response.json()

    if "api_key" not in creds:
        raise Exception("Invalid NDP credentials response from server.")
    return creds
