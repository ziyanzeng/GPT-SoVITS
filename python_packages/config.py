import yaml
import os
import sys
import re
import logging as stdlog
from loguru import logger as structlog


local_config = os.environ.get("CONFIG", "config.yaml")

with open(local_config, "r", encoding="utf-8") as f:
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)

    if "env" in conf:
        for k, v in conf["env"].items():
            env_value = os.getenv(k)
            if not env_value:
                continue
            segments = v.split(".")

            val = conf
            for segment in segments[:-1]:
                val = val[segment]
            val[segments[-1]] = env_value


def get_config() -> dict:
    return conf


def debug():
    return bool(conf["enableDebug"])


# M/MB/GB/G/KB/K
def get_file_size(s):
    result = re.search(r"^\d+(\.\d+)?", s)
    if not result:
        return 0

    num = float(result.group(0))
    unit = s[len(result.group(0)) :].upper()
    if unit == "G" or unit == "GB":
        num = num * 1024 * 1024 * 1024
    elif unit == "M" or unit == "MB":
        num = num * 1024 * 1024
    elif unit == "K" or unit == "KB":
        num = num * 1024
    return int(num)


_std_log_config = {
    "format": "[ %(levelname)s ] %(filename)s:%(lineno)d %(message)s",
    "level": stdlog.INFO,
}
stdlog.basicConfig(
    format=_std_log_config["format"],
    level=_std_log_config["level"],
    stream=sys.stdout,
    force=True,
)


def init_logger(l: stdlog.Logger):
    l.setLevel(_std_log_config["level"])
    for h in l.handlers:
        h.setFormatter(stdlog.Formatter(_std_log_config["format"]))


_struct_log_config = {
    "handlers": [{"sink": sys.stdout, "serialize": True, "enqueue": True}],
    "extra": {},
}

if "log" in conf:
    _struct_log_config = conf["log"]

if "handlers" in _struct_log_config:
    for h in _struct_log_config["handlers"]:
        if "sink" in h:
            if h["sink"] == "stdout":
                h["sink"] = sys.stdout
            if h["sink"] == "stderr":
                h["sink"] = sys.stderr

structlog.configure(**_struct_log_config)
