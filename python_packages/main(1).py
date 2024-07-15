# coding=utf-8

from __future__ import print_function
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


import os
from fastapi import FastAPI, HTTPException, status, Depends, File, UploadFile, Query

from starlette.responses import FileResponse
import base64
import time
import requests
import json
from PIL import Image
from io import BytesIO
from typing import Optional
from pydantic import BaseModel
from python_packages import objectStorage
from volcengine.visual.VisualService import VisualService
import uvicorn
from loguru import logger as log


from python_packages import config, errcode, objectStorage, httpserver

conf = config.get_config()

app = httpserver.default()


# 创建BaiduV1Text2img模型
class BaiduV1Text2img(BaseModel):
    text: str
    resolution: str
    style: str
    num: Optional[int] = 1


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": conf["baidu"]["client_id"], "client_secret": conf["baidu"]["client_secret"]}
    return str(requests.post(url, params=params).json().get("access_token"))



@app.post("/api/v1/genImage/baidutext2img")
async def baidu_create_v1_text2img(baiduV1Text2img: BaiduV1Text2img):

    url = "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/txt2img?access_token=" + get_access_token()

    payload = json.dumps({
        'text': baiduV1Text2img.text,
        'resolution': baiduV1Text2img.resolution,
        'style': baiduV1Text2img.style,
        'num': baiduV1Text2img.num
    })


    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }


    response = requests.request("POST", url, headers=headers, data=payload)

    res2Json = response.json()

    if "error_code" in res2Json:
        returnJson = {
            'error_code': res2Json.get("error_code"),
            'error_msg': res2Json.get("error_msg")
        }
    else:
        returnJson = {
            'taskId': res2Json.get("data").get("taskId")
        }

    return(returnJson)


#taskId:18697527
#18699536
@app.post("/api/v1/genImage/baidugetimg")
async def baidu_get_v1_img(taskId:int):

    url = "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/getImg?access_token=" + get_access_token()

    payload = json.dumps({
        'taskId': taskId
    })


    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    res2Json = response.json()

    if "error_code" in res2Json:
        returnJson = {
            'error_code': res2Json.get("error_code"),
            'error_msg': res2Json.get("error_msg")
        }
    else:
        returnJson = {
            'taskId': res2Json.get("data").get("taskId"),
            'status': res2Json.get("data").get("status"),
            'waiting': res2Json.get("data").get("waiting"),
            'imgUrls': res2Json.get("data").get("imgUrls")
        }

    return(returnJson)


@app.post("/api/v1/genImage/volcenginetext2img")
async def volcengine_create_high_aes_smart_drawing(prompt:str):
    timestamp = int(time.time())
    timestamp_str = str(timestamp)
    visual_service = VisualService()

    # call below method if you don't set ak and sk in $HOME/.volc/config
    visual_service.set_ak(conf["volcengine"]["ak"])
    visual_service.set_sk(conf["volcengine"]["sk"])

    # 高美感通用V1.3-文生图
    form = {
        "req_key": "high_aes",
        "model_version": "general_v1.3",
        "seed": -1,
        "scale": 3.5,
        "ddim_steps": 25,
        "width": 50,
        "height": 50,
        "use_sr": False,
        "sr_seed": -1,
        "logo_info": {
            "add_logo": False,
            "position": 0,
            "language": 0,
            "opacity": 0.3
        }
    }

    form['prompt']=prompt

    resp = visual_service.high_aes_smart_drawing(form)

    if "code" in resp:
        if resp.get("code") != 10000:
            returnJson = json.dumps({
            'error_code': resp.get("code"),
            'error_msg': resp.get("message"),
            'imgUrls': []
            })
            return returnJson

    # 将 base64 数据解码
    base64_data = ''.join(resp['data']['binary_data_base64']);
    image_data = base64.b64decode(base64_data)

    # 将二进制数据读入内存
    image = Image.open(BytesIO(image_data))

    # 保存图片
    image_name = "output"+timestamp_str+".png";
    image.save(image_name)

    #image_path = "output.png"
    return FileResponse(image_name)



@app.post("/api/v1/genImage/volcenginetext2img2Url")
async def volcengine_create_high_aes_smart_drawing(prompt:str,width:int,height:int):
    visual_service = VisualService()

    # call below method if you don't set ak and sk in $HOME/.volc/config
    visual_service.set_ak(conf["volcengine"]["ak"])
    visual_service.set_sk(conf["volcengine"]["sk"])

    # 高美感通用V1.3-文生图
    form = {
        # "req_key": "high_aes",
        # "model_version": "general_v1.3",
        "req_key": "high_aes_general_v14",
        "model_version": "general_v1.4",
        "seed": -1,
        "scale": 3.5,
        "ddim_steps": 25,
        "width": width,
        "height": height,
        "use_sr": False,
        "sr_seed": -1,
        "logo_info": {
            "add_logo": False,
            "position": 0,
            "language": 0,
            "opacity": 0.3
        }
    }

    form['prompt']=prompt

    try:
        resp = visual_service.high_aes_smart_drawing(form)
        if "code" in resp:
            if resp.get("code") != 10000:
                returnJson = {
                    'error_code': resp.get("code"),
                    'error_msg': resp.get("message"),
                    'request_id': resp.get("request_id"),
                    'imgUrls': []
                }
                return returnJson

        # 将 base64 数据解码
        base64_str_data = ''.join(resp['data']['binary_data_base64'])
        request_id = resp['request_id']
        time_elapsed = resp['time_elapsed']

        image_binary_data = base64.b64decode(base64_str_data)

        file_url = objectStorage.upload_single_file(f"genImage/volcengine/{request_id}-{time_elapsed}.jpg", image_binary_data, block=True)

        ret_json = {
            'request_id': request_id,
            "imgUrls": [
                {
                    "image": file_url
                }
            ]
        }

        return ret_json
    except Exception as e:
        returnJson = {
            'error_code': -1,
            'error_msg': str(e)
        }
        return returnJson


@app.post("/api/v1/genImage/baidupaddlepaddletext2img2Url")
async def baidu_paddlepaddle_create_v1_text2img(prompt:str,appCode:str,size:str=None,n:int=1):
    API_URL = f"https://{appCode}.aistudio-hub.baidu.com/image/generations"
    headers = {
        # 请前往 https://aistudio.baidu.com/index/accessToken 查看 访问令牌
        "Authorization": "token 3ca7629acc73f6ff7f710b56e72faf7fa4aac058",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "size": size,
        "n": n
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    resJson =response.json()

    if "errorCode" in resJson:
        return {
            'error_code': resJson["errorCode"],
            'error_msg': resJson["errorMsg"],
            'request_id': resJson["id"],
            "imgUrls": []
        }

    data_list = resJson["data"]
    # 将JSON数据写入文件
    # with open('data.json', 'w') as json_file:
    #     json.dump(data_list, json_file)

    request_id = resJson['id']

    imgUrls = []
    for item in data_list:
        index = item["index"]
        base64_str_data = item["b64_image"]
        image_binary_data = base64.b64decode(base64_str_data)
        file_url = objectStorage.upload_single_file(f"genImage/baidupaddlepaddle/{request_id}-{index}.jpg", image_binary_data, block=True)
        imgurl = {
            "image":file_url,
            "index":index
        }
        imgUrls.append(imgurl)

    ret_json = {
        'request_id': request_id,
        "imgUrls": imgUrls
    }
    return ret_json


@app.post("/api/v1/genImage/uploadImageToOss")
async def upload_image_to_oss(base64_str: str , request_id: str = None):
    # 测试数据可以从README中取
    # 将 base64 数据解码为二进制
    image_binary_data = base64.b64decode(base64_str)

    timestamp = int(time.time())
    if request_id is None:
        request_id = str(timestamp);

    file_url = objectStorage.upload_single_file(f"genImage/volcengine/{request_id}.jpg", image_binary_data, block=True)

    ret_json = {
        "imgUrls": [
            {
                "image": file_url
            }
        ]
    }

    return ret_json



if __name__ == "__main__":
    host, port = conf["httpServer"]["addr"].split(":")
    uvicorn.run(app=app, host=host, port=int(port), date_header=True)

