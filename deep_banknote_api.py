import uvicorn
import numpy as np
from fastapi import FastAPI
from api_utils import get_prediction, BankNote

# FastAPI app
app = FastAPI()

print("murad hooon")


@app.get("/")
def index():
    return {
        "Message": "Go to /docs endpoint to get predictions. Let's see if it is working."
    }


@app.post("/predict")
def predict(data: BankNote):
    """
    Function to predict bank note authenticity
    """
    data = np.array(
        [data.variance, data.skewness, data.curtosis, data.entropy], dtype=np.float32
    )
    prediction = get_prediction(data)

    if prediction == 1:
        return {"Prediction": "Warning! The bank note is Fake"}
    else:
        return {"Prediction": ":) The bank note is Real"}


# uvicorn deep_banknote_api:app --reload


# def run_app():
#     config = uvicorn.Config(
#         "deep_banknote_api:app", port=8000, log_level="info", reload=False
#     )
#     server = uvicorn.Server(config)
#     server.run()


# if __name__ == "__main__":
#     run_app()
