from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Load model and scaler
model = joblib.load("LinearRegression_CLV_model.pkl")
scaler = joblib.load("scaler.pkl")

# Encoding maps (adjust to your data)
gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
income_map = {'Low': 0, 'Medium': 1, 'High': 2}
country_map = {'USA': 0, 'UK': 1, 'Germany': 2, 'India': 3, 'Canada': 4}

# SQLite DB setup
engine = create_engine("sqlite:///clv_logs.db", echo=False)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    age = Column(Integer)
    gender = Column(String)
    income_bracket = Column(String)
    country = Column(String)
    recency = Column(Integer)
    avg_order_value = Column(Float)
    total_orders = Column(Integer)
    preferred_month = Column(Integer)
    preferred_weekday = Column(Integer)
    predicted_clv = Column(Float)
    segment = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)

# Define request schema
class CustomerInput(BaseModel):
    Age: int
    Gender: str
    IncomeBracket: str
    Country: str
    DaysSinceLastPurchase: int
    AvgOrderValue: float
    TotalOrders: int
    PreferredMonth: int
    PreferredWeekday: int

@app.post("/predict")
def predict_clv(customer: CustomerInput):
    try:
        # Encode categorical variables
        gender_encoded = gender_map.get(customer.Gender, 2)
        income_encoded = income_map.get(customer.IncomeBracket, 1)
        country_encoded = country_map.get(customer.Country, 0)

        # Prepare DataFrame
        input_df = pd.DataFrame([{
            "DaysSinceLastPurchase": customer.DaysSinceLastPurchase,
            "AvgOrderValue": customer.AvgOrderValue,
            "TotalOrders": customer.TotalOrders,
            "PreferredMonth": customer.PreferredMonth,
            "PreferredWeekday": customer.PreferredWeekday,
            "Age": customer.Age,
            "IncomeBracket": income_encoded,
            "Gender": gender_encoded,
            "Country": country_encoded
        }])

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Segment the customer
        if prediction >= 3000:
            segment = "High Value"
        elif prediction >= 1000:
            segment = "Medium Value"
        else:
            segment = "Low Value"

        # Log to DB
        session = SessionLocal()
        log_entry = PredictionLog(
            age=customer.Age,
            gender=customer.Gender,
            income_bracket=customer.IncomeBracket,
            country=customer.Country,
            recency=customer.DaysSinceLastPurchase,
            avg_order_value=customer.AvgOrderValue,
            total_orders=customer.TotalOrders,
            preferred_month=customer.PreferredMonth,
            preferred_weekday=customer.PreferredWeekday,
            predicted_clv=round(prediction, 2),
            segment=segment
        )
        session.add(log_entry)
        session.commit()
        session.close()

        return {
            "predicted_clv": round(prediction, 2),
            "segment": segment
        }

    except Exception as e:
        return {"error": str(e)}
