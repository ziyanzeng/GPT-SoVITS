import atexit
import signal
from minio import Minio
from .config import get_config
import uuid
import io
import hashlib
import os
import mimetypes
import multiprocessing as mp
import logging as log

mimetypes.init()

conf = get_config()
minio_conf = conf["minio"]


class ObjectStorageClient:
    def __init__(self) -> None:
        self.is_start = False
        self.is_close = False
        self.reset()

    def start_child_process(self, q):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.reset()

        while True:
            task = q.get()
            result = self.minio_client.put_object(
                task["bucket"],
                task["remote_path"],
                io.BytesIO(task["file"]),
                len(task["file"]),
                content_type=task["content_type"],
            )
            log.info(f"child process upload files {task['remote_path']}")

    def start(self):
        self.is_start = True
        self.q = mp.Queue()
        self.p = mp.Process(target=self.start_child_process, args=(self.q,))
        self.p.start()

    def reset(self):
        self.minio_client = Minio(
            minio_conf["endpoint"],
            access_key=minio_conf["accessKey"],
            secret_key=minio_conf["secretKey"],
            secure=False,
        )

    def upload_files(self, path: str, files: list, block=True):
        if not self.is_start and not block:
            self.start()

        file_paths = []

        content_type = ""
        ext = os.path.splitext(path)[-1]
        if ext in mimetypes.types_map:
            content_type = mimetypes.types_map[ext]
        for buf in files:
            id = uuid.uuid4()
            remote_path = path.replace("{uuid}", str(id))
            if path.find("{md5}") >= 0:
                md5sum = hashlib.md5(buf).hexdigest()
                path = path.replace("{md5}", md5sum)
            if block:
                result = self.minio_client.put_object(
                    minio_conf["bucket"],
                    remote_path,
                    io.BytesIO(buf),
                    len(buf),
                    content_type=content_type,
                )

            else:
                self.q.put(
                    {
                        "bucket": minio_conf["bucket"],
                        "remote_path": remote_path,
                        "file": buf,
                        "content_type": content_type,
                    }
                )

            # f = open(remote_path, "w")
            # f.write(str(buf))
            file_paths.append(
                f'{minio_conf["proxy"]}/{minio_conf["bucket"]}/{remote_path}'
            )
        return file_paths

    def shutdown(self):
        if self.is_start and not self.is_close:
            self.p.terminate()
            self.p.join()
        self.is_close = True


objectStorageClient = ObjectStorageClient()


def shutdown():
    objectStorageClient.shutdown()


def upload_files(path: str, files: list, block=True) -> list[str]:
    return objectStorageClient.upload_files(path, files, block)


def upload_single_file(path: str, file: bytes, block=True) -> str:
    upload_files = objectStorageClient.upload_files(path, [file], block)
    return upload_files[0]


def get_client():
    return objectStorageClient


atexit.register(shutdown)
