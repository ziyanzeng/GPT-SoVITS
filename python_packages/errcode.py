from json import JSONEncoder
import typing


def new_error(code_or_key: typing.Union[int, str], msg: str, data: dict = None):
    code = 0
    key = "ok"
    if type(code_or_key) == str:
        key = code_or_key
        code = 0 if key == "ok" else 1
    elif type(code_or_key) == int:
        code = code_or_key
        key = "ok" if code == 0 else "error"

    e = {"code": code, "key": key, "msg": msg, "data": data}
    if e["data"] is None:
        del e["data"]
    return e


def ok(data=None):
    return new_error(0, "ok", data)

# example
err_retry = new_error("retry", "please retry", {"params1": 1, "params2": 2})

# deprecated
face_and_label_not_equal = new_error(1001, "face and label not equal")
limit_only_one_face = new_error(1002, "limit only one face")
landmarks_not_found = new_error(1003, "landmarks not found")
identify_not_found = new_error(1003, "identify not found")
too_big_image = new_error(1004, "too big image")
face_label_existed = new_error(1005, "face label is existed")
not_found_face = new_error(1006, "not found face")
too_much_face_in_group = new_error(1007, "too much face in group")
