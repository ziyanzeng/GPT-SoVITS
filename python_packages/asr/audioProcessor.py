from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn


class AudioProcessor:
    def recognize(self, raw_audio: bytes):
        raise NotImplementedError("recognize")
    def process(self, raw_audio: bytes):
        raise NotImplementedError("process")
    def start(self):
        raise NotImplementedError("start")

def start_server(processor: AudioProcessor):
    processor.start()
    app = FastAPI(
        title="asr server",
        description="asr server",
        version="v1.0.0",
        docs_url=f"/api/v1/audio",
        openapi_url=f"/api/v1/audio/openapi.json",
    )

    @app.post("/api/v1/audio/process", tags=["语音转写（含过程）"])
    async def process(audio: UploadFile = File(...),):
        audio = await audio.read()
        result = processor.process(audio)
        return JSONResponse(content=result)
    
    @app.post("/api/v1/audio/recognize", tags=["语音转写（含过程）"])
    async def recognize(audio: UploadFile = File(...),):
        audio = await audio.read()
        result = processor.recognize(audio)
        return JSONResponse(content=result)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=12108,
        timeout_keep_alive=60,
    )

