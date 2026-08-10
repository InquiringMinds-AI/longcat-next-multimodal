"""Upstream SSE stream opening, shared by the OpenAI and Anthropic routes.

Why this exists: a StreamingResponse commits its HTTP status with the first byte it
sends, so any backend failure discovered INSIDE the response generator can no longer be
reported as a failure. Both routes previously opened the upstream stream inside the
generator and only caught httpx.ConnectError, which produced two silent-failure modes:

  * A backend 500 (or any non-200) arrived as a body whose lines never start with
    "data:", so the delta loop matched nothing, exited cleanly, and the client received
    HTTP 200 with zero content and finish_reason "stop" -- an empty completion that
    looks successful.
  * A mid-stream transport error (ReadError, RemoteProtocolError, timeout) escaped the
    generator, killing the connection with no terminating [DONE] and no error text.

Opening the stream here settles the status BEFORE the caller commits to 200, so a
backend error becomes a real error response.

Lives in its own module rather than gateway.py because anthropic_route needs it too and
gateway imports anthropic_route -- importing back the other way would be circular.
"""
import httpx


async def open_upstream_stream(client, url, body):
    """Open an upstream streaming POST and settle its status.

    Returns (response, None) when the upstream is streaming a 200; the caller owns the
    response and MUST aclose() it (do this in a finally inside the generator).
    Returns (None, (status_code, message)) when it is not, so each route can render the
    error in its own error schema.
    """
    req = client.build_request("POST", url, json=body)
    try:
        r = await client.send(req, stream=True)
    except httpx.ConnectError:
        return None, (503, "backend unavailable (model may still be loading)")
    except httpx.HTTPError as e:
        return None, (502, "backend error: " + str(e)[:200])
    if r.status_code != 200:
        try:
            detail = (await r.aread()).decode("utf-8", "replace")[:500]
        except Exception:
            detail = "(no body)"
        finally:
            await r.aclose()
        return None, (r.status_code, "backend error: " + detail)
    return r, None
