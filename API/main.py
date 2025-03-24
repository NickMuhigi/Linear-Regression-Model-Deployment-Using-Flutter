from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import joblib  
import os
import nest_asyncio

app = FastAPI()

# CORS setup to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class InputData(BaseModel):
    feature: float  # Input feature to the model

@app.post("/predict")
def predict(data: InputData):
    model_path = "C:/Users/kabat/Downloads/model.pkl"  # Path to your model

    # Check if the model file exists
    if not os.path.exists(model_path):
        return {"error": f"Model file '{model_path}' not found."}

    try:
        # Load the model
        model = joblib.load(model_path)
    except Exception as e:
        return {"error": f"Error loading the model: {str(e)}"}

    input_data = [[data.feature]]  # Prepare the data for prediction

    try:
        # Predict the result
        prediction = model.predict(input_data)[0]
        return {"prediction": prediction}
    except Exception as e:
        return {"error": f"Error during prediction: {str(e)}"}


if __name__ == "__main__":
    nest_asyncio.apply()  # To handle the async event loop for FastAPI
    uvicorn.run(app, host="192.168.1.82", port=8000)  # Start the API server
