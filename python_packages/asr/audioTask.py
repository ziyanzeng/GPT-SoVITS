import signal
import sys
import httpx
import multiprocessing

import redis
import json
import uuid
from loguru import logger as log
import time
import asyncio
import concurrent

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0]))
sys.path.append(str(Path(__file__).resolve().parents[2]))
from python_packages import objectStorage


from feature import FeatureProcessor
from python_packages.asr.audioProcessor import AudioProcessor, start_server


def init_process_pool():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    objectStorage.get_client().reset()

MAX_TIMEOUT = 60

def process_and_save_features(p: FeatureProcessor, taskid, raw_data):
    return p.process_and_save_features(taskid, raw_data)

class TaskPool:
    def __init__(self, processor: AudioProcessor, redis_addr: str) -> None:
        self._feature_processor = FeatureProcessor()
        # self.mpp = multiprocessing.Pool(initializer=init_pool)
        self._pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=4, initializer=init_process_pool
        )
        redis_addr = redis_addr.split(":")
        self._redis_client = redis.Redis(host=redis_addr[0], port=int(redis_addr[1]))

        ctx = multiprocessing.get_context("spawn")
        self._proc = ctx.Process(target=start_server, args=(processor,), daemon=True)
        self._proc.start()

    async def recognize(self, raw_data: bytes):
        async with httpx.AsyncClient(timeout=MAX_TIMEOUT) as client:
            response = await client.post('http://localhost:12108/api/v1/audio/recognize',files={'audio': raw_data})
        return response.json()

    def querytask(self, task_id: str):
        s = self._redis_client.get(task_id)
        return json.loads(s)

    async def addtask(self, raw_data: bytes):
        taskid = uuid.uuid4().hex
        # 识别语音
        async with httpx.AsyncClient(timeout=MAX_TIMEOUT) as client:
            response = await client.post('http://localhost:12108/api/v1/audio/process',files={'audio': raw_data})
        asr_result = response.json()

        """
        asr_result = {
            "text": "春天里春姑娘唤醒了桃花她急忙穿上了新装她又摸了摸柳树柳树立刻醒了过来发出了嫩芽在微风的吹拂下翩翩起舞她又走过了迎春花的身旁迎春花被惊醒了她揉了揉眼睛探出了一个个小脑袋她们都在欢迎春姑娘的到来"
        }
        pinyin_result = {"top": [], "pinyin": ""}

        feature_result = process_and_save_features(
            self.feature_processor, taskid, raw_data
        )
        """
        loop = asyncio.get_running_loop()
        feature_result = await loop.run_in_executor(
            self._pool,
            process_and_save_features,
            self._feature_processor,
            taskid,
            raw_data,
        )

        result = {
            "taskid": taskid,
            "text": asr_result["text"],
            "pinyin": asr_result["pinyin"],
            "hanzi": [v["text"] for v in asr_result["top"]],
            "score": [v["score"] for v in asr_result["top"]],
            "save_files": feature_result,
            'check_language': 'cn' if 'lang' not in asr_result else asr_result['lang'],
        }
        self._redis_client.set(taskid, json.dumps(result), ex=30 * 60)

        return result

    def shutdown(self):
        # self.mpp.terminate()
        # self.mpp.join()
        self._pool.shutdown(wait=True)
        self._proc.terminate()



async def test():
    import os
    import re

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from python_packages import config

    conf = config.get_config()
    p = TaskPool(conf)
    for i in range(1):
        for item in os.scandir("../../data/"):
            if (
                item.is_file()
                and re.search(".*\.wav$", item.path)
                and item.path == "../../data/en1.wav"
            ):
                print("start process wav file", item.path)
                f = open(item.path, "rb")
                raw_data = f.read()

                t0 = time.time()
                result = await p.addtask(raw_data)
                t1 = time.time()
                print(t1 - t0, item.path, result)
                f.close()
    p.shutdown()


def testrun():
    import asyncio

    asyncio.run(test())


if __name__ == "__main__":
    import time

    testrun()
