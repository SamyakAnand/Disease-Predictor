import pymongo
import os
from dotenv import load_dotenv
load_dotenv()

client = pymongo.MongoClient(os.getenv("MONGO_DB_URL"))
db = client["NetworkSecurity"]  # or whatever your DB name is
collection = db["NetworkData"]  # or whatever your collection name is

count = collection.count_documents({})
print(f"📦 Total documents in collection: {count}")

if count > 0:
    for doc in collection.find().limit(5):
        print(doc)
else:
    print("⚠️ Your MongoDB collection is empty.")