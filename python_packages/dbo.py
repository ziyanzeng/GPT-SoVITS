import redis
from .config import get_config
import mariadb
import logging as log

_conf = get_config()
_databases = {}

if "mariadb" in _conf:
    for db in _conf["mariadb"]:
        host, port = db["addr"].split(":")
        args = {
            "host": host,
            "user": db["user"],
            "password": db["password"],
            "database": db["database"],
            "port": int(port),
        }
        pool = mariadb.ConnectionPool(
            pool_name=db["name"],
            pool_size=5 if "poolSize" not in db else db["poolSize"],
        )
        pool.set_config(**args)
        _databases[db["name"]] = {
            "pool": pool,
            "args": args,
        }


def get_pool_connection(pool_name: str = "default") -> mariadb.ConnectionPool:
    try:
        pool = _databases[pool_name]
        return pool["pool"].get_connection()
    except mariadb.PoolError as e:
        log.error(f"opening connection from pool fail: {e}")
        return mariadb.connect(**pool["args"])


_redis_client = None
if "redis" in _conf:
    # 连接redis
    addr = _conf["redis"]["addr"].split(":")
    _redis_client = redis.Redis(host=addr[0], port=int(addr[1]))


def get_redis_client():
    return _redis_client
