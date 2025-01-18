import re
from enum import Enum
from http import HTTPStatus
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from app.vision import predict_step

app = FastAPI()


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


# @app.get("/items/{item_id}")
# def read_item(item_id: int):
#     return {"item_id": item_id}


class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


# @app.get("/restric_items/{item_id}")
# def read_item(item_id: ItemEnum):
#     return {"item_id": item_id}


@app.get("/query_items")
def read_item(item_id: int):
    return {"item_id": item_id}


@app.get("/text_model/")
def contains_email(data: str):
    regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data) is not None,
    }
    return response


# @app.post("/cv_model/")
# async def cv_model(
#     data: UploadFile = File(...), w: Optional[int] = 28, h: Optional[int] = 28
# ):
#     content = await data.read()

#     # Save the image
#     with open("image.jpg", "wb") as image:
#         image.write(content)

#     # Process the image after saving it
#     img = cv2.imread("image.jpg")

#     print(img.shape)
#     res = cv2.resize(
#         img, (w, h)
#     )  # Note: cv2.resize takes (width, height), not (height, width)
#     print(res.shape)

#     cv2.imwrite("resized_image.jpg", res)
#     FileResponse("image_resize.jpg")

#     response = {
#         "input": data,
#         "message": HTTPStatus.OK.phrase,
#         "status-code": HTTPStatus.OK,
#     }
#     return response


@app.post("/subtitles/")
async def image_transformer(
    data: UploadFile = File(...),
    max_length: Optional[int] = 16,
    num_beams: Optional[int] = 8,
    num_return_sequences: Optional[int] = 1,
):
    content = await data.read()

    # Save the image
    with open("image_to_transform.jpg", "wb") as image:
        image.write(content)

    subtitles = predict_step(["image_to_transform.jpg"], max_length, num_beams, num_return_sequences)
    print(subtitles)

    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "subtitles": subtitles,
    }
    return response
