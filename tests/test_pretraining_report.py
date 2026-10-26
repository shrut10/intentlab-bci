"""Check public results against the archive and preserve research-page boundaries."""

import json
from html.parser import HTMLParser
from pathlib import Path

from scripts.render_pretraining_report import render

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a":
            self.links.append(attrs.get("href", ""))


def test_public_pretraining_report_and_api_agree_with_archived_results(client):
    archive = json.loads((ROOT / "artifacts/pretraining-v1/evaluation.json").read_text())
    response = client.get("/research/pretraining")
    assert response.status_code == 200
    assert response.text == render(archive)
    assert client.get("/api/pretraining").json() == archive
    assert "989 trials" in response.text and "validation" in response.text
    links = Links()
    links.feed(response.text)
    assert len(links.ids) == len(set(links.ids))
    assert all(link[1:] in links.ids for link in links.links if link.startswith("#"))
    assert client.get("/").text.count('href="/research/pretraining"') >= 1
    assert client.get("/research").text.count('href="/research/pretraining"') >= 1
