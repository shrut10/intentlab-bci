"""Check published evidence against archived results and keep citations navigable."""

import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
        self.rows = {}
        self.model = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a":
            self.links.append(attrs.get("href", ""))
        if tag == "tr" and "data-model" in attrs:
            self.model = attrs["data-model"]
            self.rows[self.model] = []
        if tag == "td" and self.model:
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag == "td" and self.model:
            self.rows[self.model].append(self.cell)
            self.cell = None
        if tag == "tr":
            self.model = None


def test_published_research_matches_archived_model_comparison(client):
    response = client.get("/research")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    report = ReportParser()
    report.feed(response.text)
    evaluation = json.loads((ROOT / "artifacts/evaluation.json").read_text())
    assert report.rows.keys() == evaluation["models"].keys()
    for key, model in evaluation["models"].items():
        test = model["test"]
        low, high = test["bootstrap_95_ci"]
        assert report.rows[key] == [
            f"{100 * model['validation_participant_balanced_accuracy']:.2f}%",
            f"{100 * test['participant_balanced_accuracy']:.2f}%",
            f"{100 * low:.2f}–{100 * high:.2f}%",
        ]


def test_research_citations_and_assets_are_accessible(client):
    report = ReportParser()
    report.feed(client.get("/research").text)
    assert len(report.ids) == len(set(report.ids))
    for link in report.links:
        if link.startswith("#"):
            assert link[1:] in report.ids, f"Broken report anchor: {link}"
    assert {f"ref-{i}" for i in range(1, 9)} <= set(report.ids)
    assert {f"#ref-{i}" for i in range(1, 9)} <= set(report.links)
    homepage = ReportParser()
    homepage.feed(client.get("/").text)
    for link in homepage.links:
        if link.startswith("/research#"):
            assert link.partition("#")[2] in report.ids
    assert "/research" in homepage.links
    assert client.get("/static/research.css").status_code == 200
    bibliography = client.get("/static/references.bib")
    assert bibliography.status_code == 200
    assert bibliography.text.count("\n@") + bibliography.text.startswith("@") == 8
