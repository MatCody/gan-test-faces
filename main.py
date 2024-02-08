import base64
import io
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from motor.motor_asyncio import AsyncIOMotorClient
from gfpgan import GFPGANer
import cv2
import os
import numpy as np

app = FastAPI()

# MongoDB Settings
MONGO_URL = "mongodb://localhost:27017"
DATABASE_NAME = "mine"
COLLECTION_NAME = "teste_gan"

# Connect to MongoDB
client = AsyncIOMotorClient(MONGO_URL)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Set up GFPGAN restorer globally
restorer = GFPGANer(
    model_path="GFPGANv1.3.pth",
    #model_path="GFPGANCleanv1-NoCE-C2.pth",
    #upscale=2,4,8...
    #arch=clean | original
    #channel_multiplier=2,
    #bg_upsampler=None, # Set your background upsampler here
    only_center_face: bool = True 
)


@app.post("/process_image")
async def process_image(
    alarm: UploadFile = File(...),
    bg_upsampler: str = "realesrgan"
):
    try:
        contents = await input_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Restore image using GFPGAN
        _, _, restored_img = restorer.enhance(
            image_array,
            has_aligned=aligned,
            only_center_face=only_center_face,
            paste_back=True,
            weight=weight,
        )

        # Create a temporary directory, just locally to see the output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the upscaled image in the temporary directory
            output_path = os.path.join(temp_dir, f"{input_file.filename}_upscaled_better.png")
            cv2.imwrite(output_path, restored_img)
            _, img_encoded = cv2.imencode(".png", restored_img)
            img_base64 = base64.b64encode(img_encoded).decode("utf-8")
            # Save results to MongoDB
            '''result_data = {
                "input_file_name": input_file.filename,
                "output_folder": output_folder
            }'''
            #await collection.insert_one(result_data)

            # Move the saved image to the specified output folder
            output_path_final = os.path.join(output_folder, os.path.basename(output_path))
            shutil.move(output_path, output_path_final)

            # Return the link to the saved image
            result_json = {

                "input_file_name": input_file.filename,
                "output_folder": output_folder,
                "encoded_image": img_base64
                # Add more fields... or remove...
            }
            await collection.insert_one(dict(result_json))

            return JSONResponse(content=jsonable_encoder(result_json), status_code=200)

    except Exception as e:
    
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}, the image has not been upscayled")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
