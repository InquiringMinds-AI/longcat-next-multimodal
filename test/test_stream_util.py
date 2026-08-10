#!/usr/bin/env python3
"""Offline checks for upstream stream opening (no server, no httpx transport needed).

The defect these lock down: the streaming paths opened the upstream INSIDE the response
generator, by which point HTTP 200 was already committed. A backend 500 arrived as a body
whose lines never start with "data:", so the delta loop matched nothing and the client
received a successful-looking EMPTY completion instead of an error.

The stub client below returns whatever status the case asks for, so the check is on the
branch taken -- not on a live backend that happens to be healthy today.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import httpx  # noqa: E402
from stream_util import open_upstream_stream  # noqa: E402


class StubResponse:
    def __init__(self, status, body=b""):
        self.status_code = status
        self._body = body
        self.closed = False

    async def aread(self):
        return self._body

    async def aclose(self):
        self.closed = True


class StubClient:
    """Minimal stand-in: build_request is a no-op token, send returns/raises to order."""
    def __init__(self, response=None, raise_exc=None):
        self._response, self._raise = response, raise_exc

    def build_request(self, *a, **k):
        return object()

    async def send(self, req, stream=False):
        if self._raise:
            raise self._raise
        return self._response


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("200 upstream is handed back to the caller unclosed")
def _():
    resp = StubResponse(200)
    r, err = asyncio.run(open_upstream_stream(StubClient(resp), "http://x", {}))
    assert err is None and r is resp, (r, err)
    assert not resp.closed, "a live stream must stay open for the caller"
    return "passthrough, still open"


@case("backend 500 becomes an ERROR, not an empty success")
def _():
    resp = StubResponse(500, b'{"error":"engine died"}')
    r, err = asyncio.run(open_upstream_stream(StubClient(resp), "http://x", {}))
    assert r is None, r
    status, msg = err
    assert status == 500, status
    assert "engine died" in msg, msg
    # The upstream body was consumed for the message, so the response must be released.
    assert resp.closed, "non-200 upstream must be closed, not leaked"
    return "%d %s" % (status, msg)


@case("backend 422 status is preserved, not flattened to 500")
def _():
    r, err = asyncio.run(open_upstream_stream(
        StubClient(StubResponse(422, b"bad params")), "http://x", {}))
    assert err[0] == 422, err
    return "%d %s" % err


@case("connect failure maps to 503 (model still loading)")
def _():
    r, err = asyncio.run(open_upstream_stream(
        StubClient(raise_exc=httpx.ConnectError("refused")), "http://x", {}))
    assert err[0] == 503, err
    return "%d %s" % err


@case("other transport failure maps to 502")
def _():
    r, err = asyncio.run(open_upstream_stream(
        StubClient(raise_exc=httpx.ReadTimeout("slow")), "http://x", {}))
    assert err[0] == 502, err
    return "%d %s" % err


if __name__ == "__main__":
    fails = 0
    for name, fn in CASES:
        try:
            print("PASS  %s\n      %s" % (name, fn()))
        except AssertionError as e:
            fails += 1
            print("FAIL  %s\n      %s" % (name, str(e)[:300]))
    print("\n%d/%d passed" % (len(CASES) - fails, len(CASES)))
    sys.exit(1 if fails else 0)
