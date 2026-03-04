import json
from fastapi import FastAPI, HTTPException
from src.schemas.UserRequest import UserRequest
from src.schemas.UserResponse import UserResponse
from src.predict import Predict

app = FastAPI(title="Rutina neural network service")
predict = Predict()

@app.get("/")
def ping_rnn():
    return {
        "message": "this is rutina neural network service"
    }

@app.post("/advice")
def get_advice(userRequest: UserRequest):
    habit = userRequest.habit
    if not habit:
        raise HTTPException(status_code=400, detail="Bad request")
    advice = predict.give_advice(habit)
    return UserResponse(advice=advice)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)