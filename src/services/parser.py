# //download file temprarily from url

import requests
import os
import tempfile

def download_file_from_url(url: str) -> str:
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    else:
        raise Exception(f"Failed to download file from URL: {url}")
    

