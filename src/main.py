import os
from dotenv import load_dotenv

load_dotenv()

api = os.environ["LSE_KEY"]

if __name__ == "__main__":
    print(api)