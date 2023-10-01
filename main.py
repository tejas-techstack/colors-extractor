from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router as api_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,            # Add Cross-Origin Resource Sharing (CORS) middleware to handle browser security restrictions
    allow_origins = ['*'],     # Allow requests from all origins (insecure, for development only)
    allow_credentials = True,  # Allow credentials like cookies in the requests
    allow_methods = ['*'],     # Allow all HTTP methods (GET, POST, etc.)
    allow_headers = ['*'],     # Allow all headers in requests
)

@app.get('/')


async def root():
    
    return {'message':'API for color extraction from images.'}

app.include_router(api_router, prefix='/api')



