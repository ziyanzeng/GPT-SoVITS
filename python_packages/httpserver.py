import asyncio
from contextlib import asynccontextmanager
import time
import traceback
from fastapi.responses import Response
from fastapi import FastAPI, Request
import logging as log

from .config import get_config

conf = get_config()

swagger_conf: dict = {
    "title": conf.get("title", ""), 
    "description": conf.get("description", ""),
    "version": conf.get("version", "1.0.0"),
    "docsURL": f"/api/v1/{conf.get('name', '')}",
}
for k,v in conf.get("swagger", {}).items():
    swagger_conf[k] = v

app = FastAPI(
    title=swagger_conf.get("title", ""),
    description=swagger_conf.get("description", ""),
    version=swagger_conf.get("version", "1.0.0"),
    docs_url=swagger_conf.get("docsURL", None),
    openapi_url=f'{swagger_conf.get("docsURL")}/openapi.json',
)

def default():
    return app

@app.middleware("http")
async def log_requests(request: Request, call_next):
    startTime = time.time()
    try:
        request_body = await request.body()
        response: Response = await call_next(request)
    except RuntimeError as exc:
        if str(exc) == "No response returned." and await request.is_disconnected():
            return Response(status_code=204)

    if response.status_code == 422:
        log.info(
            f"request method {request.method} path {request.url.path} body {request_body.decode()}"
        )

    processTime = (time.time() - startTime) * 1000
    formattedProcessTime = "{0:.2f}".format(processTime)
    log.info(
        f"http response request_uri: {request.url.path} status_code: {response.status_code} cost_time: {formattedProcessTime}"
    )

    return response


@app.middleware("http")
async def cors_handler(request: Request, call_next):
    response: Response = await call_next(request)

    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


class Hook(object):
    def __init__(self) -> None:
        self._max_request_num = 12345678
        self._timeout_value = 60
        self._request_id = 0
        self._timeout_call = 0
        self._call_num = 0

        if "httpServer" in conf and "maxRequestNum" in conf["httpServer"]:
            self._max_request_num = conf["httpServer"]["maxRequestNum"]

        if "httpServer" in conf and "timeout" in conf["httpServer"]:
            self._timeout_value = conf["httpServer"]["timeout"]

    async def handle(self, request: Request, call_next):
        self._request_id += 1

        t1 = time.time()
        for delay in [0, 0.5, 1, 2, 5, -1]:
            if delay < 0:
                return Response("Service Unavailable", status_code=503)

            allow_call_num = self._max_request_num
            if self._timeout_call > 3:
                allow_call_num = self._max_request_num / 2
            if allow_call_num >= self._call_num:
                break
            if delay > 0:
                await asyncio.sleep(delay)

        try:
            self._call_num += 1

            start_time = time.time()
            result = await call_next(request)
            cost_time = time.time() - start_time
            if cost_time > self._timeout_value:
                self._timeout_call += 1
            else:
                self._timeout_call = 0

            self._call_num -= 1
            return result
        except:
            log.error(f"{traceback.format_exc()}")

        self._call_num -= 1
        return Response("Internal server error", status_code=500)


hook = Hook()


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    return await hook.handle(request, call_next)
