import asyncio
from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, quote_plus, urljoin, urlsplit

import fastapi
import orjson
from fastapi import routing as fastapi_routing
from fastapi.datastructures import Default, DefaultPlaceholder
from multipart.exceptions import MultipartParseError
from sentry_sdk import capture_exception
from starlette.background import BackgroundTask
from starlette.datastructures import FormData
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket as StarletteWebSocket
from starlette.websockets import WebSocketState

if TYPE_CHECKING:
    from models import User
    from starlette.types import Message as StarletteMessage


__all__ = ("Request", "WebSocket", "ORJSONResponse")


def iri_to_uri(iri: str) -> str:
    """
    Convert an Internationalized Resource Identifier (IRI) to a Uniform Resource Identifier (URI).
    This function assumes that the input is a valid IRI.
    """
    return quote_plus(iri, safe="/#%[]=:;$&()+,!?*@'~")


def escape_uri_path(path: str) -> str:
    """
    A utility function to escape a URI path component.
    """
    return quote(path, safe="/%")


class Request(StarletteRequest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Added for faster user setting for tests; not safe for production
        self.u: "User"

    @property
    def user(self) -> "User":
        return self.scope.get("user") or self.u

    @property
    def auth(self):
        headers = self.scope.get("headers")
        authorization_header = None
        for header in headers:
            if header[0].lower() == b"authorization":
                authorization_header = header[1]

        if authorization_header and authorization_header.startswith(b"Bearer "):
            token = authorization_header.split(b"Bearer ")[1].decode("utf-8")
            return token

    @property
    def _messages(self):
        return self.scope["_messages"]

    async def form(self, *args: Any, **kwargs: Any) -> FormData:
        """
        Ref: https://github.com/encode/starlette/issues/847

        If we access the request data in the middleware, the stream is exhausted and
        the request body is empty.
        """

        if data := self.scope.get("_form_data"):
            setattr(self, "_form", data)
            return data
        try:
            form_data = await super().form(*args, **kwargs)
        except MultipartParseError:
            form_data = FormData()
        self.scope["_form_data"] = form_data
        return form_data

    def build_absolute_uri(self, location=None) -> str:
        """
        Build an absolute URI from the location and the variables available in
        this request. If no ``location`` is specified, build the absolute URI
        using request.get_full_path(). If the location is absolute, convert it
        to an RFC 3987 compliant URI and return it. If location is relative or
        is scheme-relative (i.e., ``//example.com/``), urljoin() it to a base
        URL constructed from the request variables.
        """
        if location is None:
            # Make it an absolute url (but schemeless and domainless) for the
            # edge case that the path starts with '//'.
            location = "//%s" % self.get_full_path()
        else:
            # Coerce lazy locations.
            location = str(location)
        bits = urlsplit(location)
        if not (bits.scheme and bits.netloc):
            # Handle the simple, most common case. If the location is absolute
            # and a scheme or host (netloc) isn't provided, skip an expensive
            # urljoin() as long as no path segments are '.' or '..'.
            if (
                bits.path.startswith("/")
                and not bits.scheme
                and not bits.netloc
                and "/./" not in bits.path
                and "/../" not in bits.path
            ):
                # If location starts with '//' but has no netloc, reuse the
                # schema and netloc from the current request. Strip the double
                # slashes and continue as if it wasn't specified.
                if location.startswith("//"):
                    location = location[2:]
                location = self.current_scheme_host + location
            else:
                # Join the constructed URL with the provided location, which
                # allows the provided location to apply query strings to the
                # base path.
                location = urljoin(self.current_scheme_host + self.url.path, location)
        return iri_to_uri(location)

    def get_full_path(self, force_append_slash=False):
        return self._get_full_path(self.url.path, force_append_slash)

    def _get_full_path(self, path, force_append_slash):
        # RFC 3986 requires query string arguments to be in the ASCII range.
        # Rather than crash if this doesn't happen, we encode defensively.
        return "{}{}{}".format(
            escape_uri_path(path),
            "/" if force_append_slash and not path.endswith("/") else "",
            ("?" + iri_to_uri(self.url.query)) if self.url.query else "",
        )

    @cached_property
    def current_scheme_host(self):
        url = f"{self.url.scheme}://{self.url.hostname}"
        # FIXME due to reverse proxy, we don't know if it's https
        if not self.url.hostname.startswith("localhost"):
            url = url.replace("http", "https", 1).replace("httpss", "https", 1)
        if self.url.port:
            return f"{url}:{self.url.port}"
        return url

    @property
    def ip(self) -> str:
        if x_forwarded_for := self.headers.get("X-Forwarded-For"):
            return x_forwarded_for.split(",")[0]
        elif remote_addr := self.headers.get("Remote-Addr"):
            return remote_addr
        else:
            return self.client.host


def request_response(func: Callable) -> ASGIApp:
    """
    Takes a function or coroutine `func(request) -> response`,
    and returns an ASGI application.
    """
    # from starlette._utils import is_async_callable
    # from starlette.concurrency import run_in_threadpool
    # is_coroutine = is_async_callable(func)

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive=receive, send=send)
        # Force all views to be a coroutine
        response = await func(request)
        await response(scope, receive, send)

    return app


fastapi_routing.request_response = request_response


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(
            content,
            option=orjson.OPT_NON_STR_KEYS
            | orjson.OPT_SERIALIZE_NUMPY
            | orjson.OPT_NAIVE_UTC,
        )


def api_route(self, path, **kwargs):
    def decorator(func):
        if isinstance(kwargs.get("response_class"), DefaultPlaceholder):
            kwargs["response_class"] = Default(ORJSONResponse)
        self.add_api_route(
            path,
            func,
            **kwargs,
        )
        return func

    return decorator


fastapi_routing.APIRouter.api_route = api_route


class WebSocket(StarletteWebSocket):
    @property
    def user(self) -> "User":
        return self.scope["user"]

    @cached_property
    def chat_user_id_document_id(self) -> str:
        return f"{self.chat_user.id}:{self.chat_user.document_id}"

    @property
    def websocket_session_id(self) -> str:
        return self.scope["websocket_session_id"]

    async def send(self, message: "StarletteMessage") -> None:
        """
        Send ASGI websocket messages, ensuring valid state transitions.
        """
        # avoid raising RuntimeError
        if self.application_state == WebSocketState.DISCONNECTED:
            return
        return await super().send(message)

    async def receive_json(self, mode: str = "text") -> Any:
        if mode not in {"text", "binary"}:
            raise RuntimeError('The "mode" argument should be "text" or "binary".')
        if self.application_state != WebSocketState.CONNECTED:
            raise RuntimeError(
                'WebSocket is not connected. Need to call "accept" first.'
            )
        message = await self.receive()
        self._raise_on_disconnect(message)

        if mode == "binary":
            text = message["bytes"]
        else:
            text = message["text"].encode("utf-8")
        return orjson.loads(text)

    async def send_json(self, data: Any, mode: str = "text") -> None:
        if mode not in {"text", "binary"}:
            raise RuntimeError('The "mode" argument should be "text" or "binary".')
        text = orjson.dumps(data)
        if mode == "binary":
            try:
                await self.send({"type": "websocket.send", "bytes": text})
            except Exception as e:
                capture_exception(e)
        else:
            try:
                await self.send(
                    {"type": "websocket.send", "text": text.decode("utf-8")}
                )
            except Exception as e:
                capture_exception(e)


def websocket_session(func: Callable) -> ASGIApp:
    """
    Takes a coroutine `func(session)`, and returns an ASGI application.
    """
    # assert asyncio.iscoroutinefunction(func), "WebSocket endpoints must be async"

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        session = WebSocket(scope, receive=receive, send=send)
        await func(session)

    return app


fastapi_routing.websocket_session = websocket_session

old_background_task_constructor = fastapi.BackgroundTasks.__init__


def background_task_constructor(self, *args, **kwargs):
    old_background_task_constructor(self, *args, **kwargs)
    self.at_once = True


async def background_task_call(self) -> None:
    if self.at_once:
        await asyncio.gather(*(task() for task in self.tasks), return_exceptions=True)
    else:
        for task in self.tasks:
            await task()


def add_task(self, func, *args, **kwargs) -> None:
    task = BackgroundTask(func, *args, **kwargs)
    task.is_async = True
    self.tasks.append(task)


fastapi.BackgroundTasks.add_task = add_task
fastapi.BackgroundTasks.__init__ = background_task_constructor
fastapi.BackgroundTasks.__call__ = background_task_call
