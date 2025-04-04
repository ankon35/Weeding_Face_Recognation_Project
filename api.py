from io import BytesIO
import os
from tkinter import Image
from typing import List
import cloudinary
import cloudinary.api
import cloudinary.uploader 
import cv2
import dlib
import face_recognition
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv 
from PIL import Image as PilImage 
import torch
import uvicorn
from bson import ObjectId  # Import ObjectId to handle it

import numpy as np
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO

# Load environment variables
load_dotenv()

# Initialize Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET"),
)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["Ankon"]
image_collection = db["image_encoded"]
folders_collection = db["folders"]

app = FastAPI()

# Pydantic model for request body
class FolderCreateRequest(BaseModel):
    folder_name: str
    cost_per_pic: float

# Helper function to serialize ObjectId to string
def serialize_objectid(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError("ObjectId not serializable")

@app.post("/create_folder/")
async def create_folder(folder: FolderCreateRequest):
    """
    Creates a folder in Cloudinary and stores folder metadata in MongoDB.
    """
    try:
        # Create the parent folder in Cloudinary (the name passed by the user)
        cloudinary.api.create_folder(folder.folder_name)

        # Create the 'main','preview' & 'temp' subfolders inside the parent folder
        cloudinary.api.create_folder(f"{folder.folder_name}/main")
        cloudinary.api.create_folder(f"{folder.folder_name}/preview")
        cloudinary.api.create_folder(f"{folder.folder_name}/temp")

        # Save folder details in MongoDB
        folder_data = {
            "folder_name": folder.folder_name,
            "cost_per_pic": folder.cost_per_pic,
            "created_at": datetime.now().strftime('%Y-%m-%d'), # Use only Year-Month-Day
            "folder_size": 0,  # Initially 0
            "main_folder": f"{folder.folder_name}/main",
            "preview_folder": f"{folder.folder_name}/preview"
        }

        # Insert the folder data into MongoDB
        insert_result = folders_collection.insert_one(folder_data)

        # Prepare the folder data for response, convert ObjectId to string
        folder_data["_id"] = serialize_objectid(insert_result.inserted_id)

        return {"message": "Folder created successfully", "folder_info": folder_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def detect_and_encode(image_data):
    """Detect faces in an image and encode them using face_recognition."""
    # Decode the image from bytes
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces using face_recognition
    face_locations = face_recognition.face_locations(img_rgb)
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

    face_data = []
    for i, encoding in enumerate(face_encodings):
        x1, y1, x2, y2 = face_locations[i]
        face_data.append({
            "box": [x1, y1, x2, y2],
            "embedding": encoding.tolist()
        })

    return face_data

@app.post("/upload_image/")
async def upload_image(
    folder_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    try:
        uploaded_files_info = []
        current_date = datetime.now().strftime('%Y-%m-%d')
        total_size_bytes = 0

        # Create folders
        cloudinary.api.create_folder(f"{folder_name}/main")
        cloudinary.api.create_folder(f"{folder_name}/preview")
        cloudinary.api.create_folder(f"{folder_name}/temp")

        for file in files:
            image_data = await file.read()
            pil_image = PilImage.open(BytesIO(image_data))

            # Upload image to 'main' folder
            main_upload_result = cloudinary.uploader.upload(
                image_data,
                folder=f"{folder_name}/main",
                public_id=file.filename.split('.')[0],  # Keeps the name consistent
                overwrite=True
            )

            total_size_bytes += main_upload_result.get("bytes", 0)

            # Encode face from image
            face_data = detect_and_encode(image_data)

            # Prepare metadata
            image_metadata = {
                "folder_name": folder_name,
                "file_name": file.filename,
                "created_at": current_date,
                "temp_folder_url": "",  # Not used anymore
                "faces_detected": len(face_data),
                "face_encodings": face_data,
                "main_folder_url": main_upload_result["secure_url"],
                "preview_folder_url": ""  # Will be filled after upload
            }

            # Insert metadata
            insert_result = image_collection.insert_one(image_metadata)
            image_metadata["_id"] = str(insert_result.inserted_id)

            # Resize image for preview
            preview_image = pil_image.copy()
            preview_image.thumbnail((150, 150))  # Resize to 150x150
            preview_image_bytes = BytesIO()
            preview_image.save(preview_image_bytes, format="JPEG")
            preview_image_bytes.seek(0)

            # Upload resized image to 'preview' folder with the same public_id
            preview_upload_result = cloudinary.uploader.upload(
                preview_image_bytes,
                folder=f"{folder_name}/preview",
                public_id=file.filename.split('.')[0],
                overwrite=True
            )

            # Update preview URL in metadata
            image_metadata["preview_folder_url"] = preview_upload_result["secure_url"]

            # Update MongoDB with preview URL
            image_collection.update_one(
                {"_id": insert_result.inserted_id},
                {"$set": {
                    "preview_folder_url": preview_upload_result["secure_url"]
                }}
            )

            uploaded_files_info.append(image_metadata)

        # Folder stats
        total_size_gb = round(total_size_bytes / (1024 ** 3), 3)
        image_count = image_collection.count_documents({"folder_name": folder_name})

        # Update or insert folder info
        folder_doc = folders_collection.find_one({"folder_name": folder_name})
        if folder_doc:
            folders_collection.update_one(
                {"folder_name": folder_name},
                {"$set": {"folder_size": total_size_gb, "image_count": image_count}}
            )
        else:
            folders_collection.insert_one({
                "folder_name": folder_name,
                "folder_size": total_size_gb,
                "image_count": image_count,
                "created_at": current_date,
                "main_folder": f"{folder_name}/main",
                "preview_folder": f"{folder_name}/preview",
                "temp_folder": f"{folder_name}/temp"
            })

        return {
            "message": "Images uploaded and processed successfully",
            "total_size_gb": total_size_gb,
            "image_count": image_count,
            "files": uploaded_files_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Encode face using FaceNet
def encode_face(image_data):
    """Encode the face in the uploaded image using FaceNet"""
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect face using face_recognition library
    faces = face_recognition.face_locations(img_rgb)
    if len(faces) == 0:
        raise HTTPException(status_code=404, detail="No face detected in the image.")

    encodings = face_recognition.face_encodings(img_rgb, faces)

    return encodings[0]  # Return the encoding of the first detected face


@app.post("/image_finder/")
async def image_finder(folder_name: str = Form(...), image: UploadFile = File(...)):
    """
    Accepts a folder name and a single image, encodes the uploaded image,
    and returns preview images with matching faces found in the folder.
    """
    try:
        # Read the uploaded image
        image_data = await image.read()
        uploaded_image_encoding = encode_face(image_data)  # Get the encoding of the uploaded image

        # Retrieve images from MongoDB in the specified folder
        images_in_folder = image_collection.find({"folder_name": folder_name})
        matching_images = []

        for image_metadata in images_in_folder:
            for face_data in image_metadata.get("face_encodings", []):
                # Compare embeddings (using cosine similarity or a simple comparison)
                dist = np.linalg.norm(np.array(uploaded_image_encoding) - np.array(face_data['embedding']))
                if dist < 0.6:  # Threshold for considering face as a match (this can be adjusted)
                    matching_images.append({
                        "file_name": image_metadata["file_name"],
                        "preview_folder_url": image_metadata["preview_folder_url"],
                        "match_distance": dist
                    })

        if not matching_images:
            raise HTTPException(status_code=404, detail="No matching faces found in the folder.")

        return {"message": "Matching images found", "matches": matching_images}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/delete_folder/")
async def delete_folder(folder_name: str = Form(...), db_name: str = Form(...)):
    try:
        # Connect to the specified database
        selected_db = client[db_name]

        # Define collections dynamically
        image_collection = selected_db["image_encoded"]
        folders_collection = selected_db["folders"]

        # --- Delete from Cloudinary ---
        folders = ["main", "preview", "temp"]
        for subfolder in folders:
            full_path = f"{folder_name}/{subfolder}"

            # Delete all resources inside subfolder
            cloudinary.api.delete_resources_by_prefix(full_path)

            # Delete the subfolder itself
            try:
                cloudinary.api.delete_folder(full_path)
            except Exception:
                pass  # Folder may already be empty or deleted

        # Delete main folder itself
        try:
            cloudinary.api.delete_folder(folder_name)
        except Exception:
            pass

        # --- Delete from MongoDB ---
        # Delete all image metadata from the selected DB
        image_result = image_collection.delete_many({"folder_name": folder_name})
        folder_result = folders_collection.delete_one({"folder_name": folder_name})

        return {
            "message": f"Folder '{folder_name}' deleted successfully from database '{db_name}'.",
            "images_deleted": image_result.deleted_count,
            "folder_record_deleted": folder_result.deleted_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
