"""A new deployment must refresh assets even when size and mtime are unchanged."""

import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from intentlab.api import ContentStaticFiles


def test_changed_asset_refreshes_with_normalised_build_metadata(tmp_path):
    asset = tmp_path / "theme.css"
    old_css = "button { color: #123; }"
    new_css = "button { color: #456; }"
    asset.write_text(old_css)
    original = asset.stat()
    app = FastAPI()
    app.mount("/static", ContentStaticFiles(directory=tmp_path))

    with TestClient(app) as client:
        first = client.get("/static/theme.css")
        asset.write_text(new_css)
        os.utime(asset, ns=(original.st_atime_ns, original.st_mtime_ns))
        assert asset.stat().st_size == original.st_size
        assert asset.stat().st_mtime_ns == original.st_mtime_ns

        updated = client.get("/static/theme.css", headers={"If-None-Match": first.headers["ETag"]})
        assert updated.status_code == 200 and updated.text == new_css
        assert updated.headers["ETag"] != first.headers["ETag"]
        assert "Last-Modified" not in updated.headers

        unchanged = client.get("/static/theme.css", headers={"If-None-Match": updated.headers["ETag"]})
        assert unchanged.status_code == 304

        # Older cached responses may send only the deployment's fixed timestamp.
        date_only = client.get(
            "/static/theme.css", headers={"If-Modified-Since": "Sat, 20 Oct 2018 01:46:40 GMT"}
        )
        assert date_only.status_code == 200 and date_only.text == new_css
