import requests
from requests.auth import HTTPBasicAuth
from .endpoints import SPACE_PAGES_ENDPOINT
import os
import json
from typing import Any

class SpaceMetadata:
    def __init__(self, space_id: int) -> None:
        self._space_id = space_id
        self._metadata_request_url = SPACE_PAGES_ENDPOINT.format(DOMAIN=os.getenv("ATLASSIAN_DOMAIN"), SPACE_ID=space_id)
        self._pages_metadata = self._query_space_pages_metadata()

    def _query_space_pages_metadata(self) -> list[dict[str, Any]]:
        auth = HTTPBasicAuth(os.getenv("ATLASSIAN_USER"), os.getenv("ATLASSIAN_API_KEY"))
        headers = {'Accept': 'application/json'}
        response = requests.request("GET", self._metadata_request_url, headers=headers, auth=auth)
        if response.status_code != 200:
            raise Exception(f"Failed to get space metadata. Status code: {response.status_code} - {response.content}")
        return json.loads(response.text)["results"]

    @property
    def space_id(self) -> int:
        return self._space_id

    @property
    def metadata_url(self) -> str:
        return self._metadata_request_url

    @property
    def unique_page_ids(self) -> list[int]:
        return list(set(int(x["id"]) for x in self._pages_metadata))
