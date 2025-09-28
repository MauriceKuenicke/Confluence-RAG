import requests
from requests.auth import HTTPBasicAuth
from .endpoints import PAGE_METADATA_ENDPOINT
import os
import json
from typing import Any

class PageMetadata:
    def __init__(self, page_id: int) -> None:
        self._page_id = page_id
        self._metadata_request_url = PAGE_METADATA_ENDPOINT.format(DOMAIN=os.getenv("ATLASSIAN_DOMAIN"), PAGE_ID=page_id)
        self._pages_metadata = self._query_page_metadata()

    def _query_page_metadata(self) -> dict[str, Any]:
        auth = HTTPBasicAuth(os.getenv("ATLASSIAN_USER"), os.getenv("ATLASSIAN_API_KEY"))
        response = requests.request("GET", self._metadata_request_url, headers={'Accept': 'application/json'}, auth=auth)
        if response.status_code != 200:
            raise Exception(f"Failed to get page metadata. Status code: {response.status_code} - {response.content}")
        return json.loads(response.text)

    @property
    def page_id(self) -> int:
        return int(self._page_id)

    @property
    def space_id(self) -> int:
        return int(self._pages_metadata["spaceId"])


    @property
    def created_at(self) -> str:
        return self._pages_metadata["createdAt"]

    @property
    def updated_at(self) -> str:
        return self._pages_metadata["version"]["createdAt"]

    @property
    def version_number(self) -> int:
        return int(self._pages_metadata["version"]["number"])

    @property
    def raw_content(self) -> str:
        return self._pages_metadata["body"]["view"]["value"]

    @property
    def title(self) -> str:
        return self._pages_metadata["title"]

    @property
    def metadata_url(self) -> str:
        return self._metadata_request_url
