import argparse
import base64
import io
import requests

from PIL import Image

import pandas as pd


# DETECTION_URL = "http://127.0.0.1:8000/upload/"
# IMAGE = Image.open('images/test.jpg')

# преобразовать изображение в байты
def image_to_byte_array(in_image: Image) -> bytes:
	imgByteArr = io.BytesIO()
	in_image.save(imgByteArr, format=in_image.format)
	byteIm = imgByteArr.getvalue()  # 
	return byteIm


def send_request(img, url):
	img_bytes = image_to_byte_array(Image.open(img))  # convert image 
	response = requests.post(url, data=img_bytes).json()  # send image 

	img_str = response['image']  # получить изображение в строковом формате base64 после обнаружения
	img_base64 = base64.b64decode(img_str)  # преобразовать строку base64 в байты
	img_bio = io.BytesIO(img_base64)  # Преобразовать в BytesIO для обработки с помощью Pillow

	img = Image.open(img_bio)  
	img.save('output.png', format='png')  # сохранить изображение c ограничивающими рамками

	special_info = response['bound_box_info']  # инфа о боксах

	df = pd.DataFrame.from_dict(special_info)  # создать фрейм данных из словаря
	df.to_csv(r'output.csv', index=False, header=True)  


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", type=str, default='test.jpg', help="Path to image")
	parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/upload/", help="URL to upload")
	opt = parser.parse_args()

	send_request(img=opt.source, url=opt.url)
