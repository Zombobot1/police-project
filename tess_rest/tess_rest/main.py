import uvicorn
from typing import Optional
from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI
from PIL import ImageEnhance, ImageFilter, Image
import pytesseract
import io
import binascii
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"preds": pred(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}


def start():
    """Launched with `poetry run start` at root level"""

    print(pytesseract.get_languages())
    uvicorn.run("tess_rest.main:app", host="0.0.0.0", port=9000)


def pred(file):
    image = Image.open(io.BytesIO(file))
    data_dict = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(data_dict['level'])
    res = []
    for i in range(n_boxes):
        (x, y, w, h) = (data_dict['left'][i], data_dict['top']
                        [i], data_dict['width'][i], data_dict['height'][i])

        right = x+w
        low = y+h
        img = image.crop((x, y, right, low))
        res.append(pytesseract.image_to_string(img, lang='deu'))

    return res
