import requests
from pathlib import Path

# API Server URL
API_URL = "http://127.0.0.1:8000"

# Find a test image in raw dataset
TEST_IMAGE_PATH = Path("data/raw/test/img/X00016469670.jpg")


def test_server():
    # 1. Check if the server is running
    print("Checking if FastAPI server is online...")
    try:
        response = requests.get(f"{API_URL}/")
        print("Server Response:", response.json())
    except requests.exceptions.ConnectionError:
        print(
            "\n[ERROR] FastAPI server is not running! "
            "Please start it first using:\n"
            "uvicorn app.main:app --reload\n"
        )
        return

    # 2. Test /predict endpoint (Upload receipt and run models)
    if not TEST_IMAGE_PATH.is_file():
        print(f"[ERROR] Test image not found at: {TEST_IMAGE_PATH}")
        return

    print(f"\nUploading {TEST_IMAGE_PATH.name} to /predict endpoint...")
    with TEST_IMAGE_PATH.open("rb") as f:
        files = {"file": (TEST_IMAGE_PATH.name, f, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files)

    if response.status_code == 200:
        print("Success! Response from Server:")
        print(response.json())
    else:
        print(f"Failed with status code {response.status_code}:")
        print(response.text)

    # 3. Test /receipts endpoint (Get all records from SQLite)
    print("\nFetching all receipts in the database...")
    response = requests.get(f"{API_URL}/receipts")
    if response.status_code == 200:
        print("Database Records:")
        print(response.json())
    else:
        print("Failed to fetch database records.")


if __name__ == "__main__":
    test_server()
