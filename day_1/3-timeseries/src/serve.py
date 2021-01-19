import joblib
import os
import json
import pandas as pd


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def input_fn(request_body, request_content_type):
    
    """handles input received from the API endpoint
    In this case input is expected in this format:
    {
        "start": "1/1/2016",
        "end": "2/28/2017"
    }
    Where start and end represent the peirod for which we wish the model to provide a forecast
    """
    request = json.loads(request_body)

    future = pd.DataFrame(pd.date_range(start=request["start"], 
                                        end=request["end"], 
                                        freq="D"), 
                          columns=["ds"])
    return future


def predict_fn(input_object, model):
    
    """Takes the output from input_fn and passes it to the model"""
    
    prediction = model.predict(input_object)
    
    return prediction

def output_fn(prediction, content_type):
    
    """Takes the ouput of predict_fn and converts it into the final format for serving
    The output format is as follows:
    [
        {"ds":1/1/2016, "yhat_lower": 5,"yhat_upper": 20, "yhat": 12},
        ...
    ]
    Essentially a json list of predictions for each day
    """
    
    return_cols = ["ds","yhat_lower","yhat_upper","yhat"]
    prediction["ds"] = prediction["ds"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    
    predictions = prediction[return_cols].apply(lambda x: dict(zip(return_cols, x.values)), axis=1).values.tolist()
    
    return json.dumps(predictions).encode("utf8")
    
