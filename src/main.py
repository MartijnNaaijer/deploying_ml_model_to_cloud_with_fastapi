from fastapi import FastAPI

app = FastAPI()

@app.get('/')
async def say_hello():
    return({'greeting': 'Hi, what about making an inference?'})
