"""Thin client used by the public API to reach the internal fusion worker.

The runtime image ships no HTTP client library (no httpx, no requests) and the
Dockerfile is read-only, so this deliberately sticks to urllib from the standard
library. Calls never leave the loopback interface.

Imports nothing from fusion: the public API depends on this module and must stay
free of the merge engine's dependency tree.
"""

import json
import urllib.error
import urllib.request

import tools


# urlopen honours http_proxy/HTTP_PROXY, which would send loopback traffic to a
# proxy when the container defines one. An empty ProxyHandler pins every call to
# the local interface.
internal_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def build_internal_url(path):
    return f"http://127.0.0.1:{tools.internal_api_port}{path}"


def call_internal_api(method, path, payload=None, timeout=15):
    """Call the internal worker and return (status_code, decoded_body).

    An HTTP error status is returned like any other response so the caller can
    forward the worker's own status and detail. Transport failures (worker down,
    still starting) surface as urllib.error.URLError / OSError.
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
    body = raw_body.decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except ValueError:
        return {"detail": body}
