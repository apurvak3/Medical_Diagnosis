from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health_check():
    return {"message": "ok"}
def main():
    print("Hello from medical-diagnosis!")


if __name__ == "__main__":
    main()
