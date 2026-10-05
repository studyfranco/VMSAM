"""Minimal urllib client used by the public API to reach the internal merge worker.

Uses only the standard library and never imports the merge engine, so the
public API stays free of its dependencies. Calls stay on the loopback interface.
"""

import json
import urllib.error
import urllib.request

import tools


# An empty ProxyHandler keeps loopback calls away from any http_proxy setting.
internal_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def build_internal_url(path):
    """Return the loopback URL of `path` on the internal API."""
    return f"http://127.0.0.1:{tools.internal_api_port}{path}"


def call_internal_api(method, path, payload=None, timeout=15):
    """Call the internal worker and return (status_code, decoded_body).

    An HTTP error status is returned like any response; transport failures raise
    urllib.error.URLError or OSError.
    """
    data = None
    headers = {}
    if payload != None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(build_internal_url(path), data=data, headers=headers, method=method)
    try:
        with internal_opener.open(request, timeout=timeout) as response:
            return response.status, decode_body(response.read())
    except urllib.error.HTTPError as e:
        return e.code, decode_body(e.read())


def decode_body(raw_body):
    """Decode a response body as JSON, else wrap the text as `{"detail": text}`."""
    body = raw_body.decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except ValueError:
        return {"detail": body}
