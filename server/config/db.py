from dotenv import load_dotenv
from pymongo import MongoClient
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME" , "rbac-diagnosis")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]


user_collection = db.get_collection("users")
report_collection = db.get_collection("reports")
diagnosis_collection = db.get_collection("diagnoses")