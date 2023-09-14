import argparse
import base64
import io
import json
import torch
import uvicorn

from fastapi import FastAPI, Depends, Request
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)

app = FastAPI()

DETECTION_URL = "/upload"


def image_to_byte_array(in_image: Image) -> bytes:
	imgByteArr = io.BytesIO()
	in_image.save(imgByteArr, format=in_image.format)
	byteIm = imgByteArr.getvalue()  
	return byteIm


# получить байты изображения из запроса
async def parse_body(request: Request):
	data: bytes = await request.body()
	return data


@app.post(DETECTION_URL)
async def image_process(image_bytes: bytes = Depends(parse_body)):
	image = Image.open(io.BytesIO(image_bytes))
	results = model(image)

	detect_res = results.pandas().xyxy[0].to_json(orient="records")  # привязка информации о поле к формату json с помощью pandas
	detect_res_json = json.loads(detect_res)

	response_dict = {
		'bound_box_info': detect_res_json
	}

	results.render()

	imgByteArr = io.BytesIO()
	img_base64 = Image.fromarray(results.imgs[0])  
	img_base64.save(imgByteArr, format="JPEG")
	img_str = base64.b64encode(imgByteArr.getvalue()).decode('utf-8')  # изображение в кодировке base64 с результатами
	response_dict['image'] = img_str  

	return response_dict


# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument("--port", default=8000, type=int, help="port number")
# 	opt = parser.parse_args()
#
#
#
# 	uvicorn.run(app, host="127.0.0.1", port=opt.port)  # debug=True causes Restarting with stat
