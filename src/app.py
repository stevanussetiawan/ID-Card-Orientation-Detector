from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from face_recognition_service import FaceRecognitionService

# Pydantic model for request body validation
class Item(BaseModel):
    base64_id_card: str = Field(..., description="Base64 encoded ID card image")

# FastAPI app instance
app = FastAPI(
    title="ID Card Orientation Detector",
    description="An API to detect the orientation of ID cards using face recognition.",
    version="1.0.0"
)

# Adding CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def get_api():
    return {
        "validation": "success",
        "message": "ID Card Orientation Detector"
    }

def get_face_recognition_service():
    return FaceRecognitionService()

@app.post("/predictIDCardOrientationDetector/")
async def predict(
    data: Item, 
    face_recog_service: FaceRecognitionService = Depends(get_face_recognition_service)
):
    res_ai = {}
    try:
        b64_image_id = data.base64_id_card
        result_orientation_angle = face_recog_service.process_request(b64_image_id)
        res_ai["Prediction_AI"] = result_orientation_angle
        return {
            "Status": True,
            "Msg": "Proses AI berhasil",
            "Result_AI": res_ai
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))