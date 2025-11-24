#!/usr/bin/env python3
"""
Simple test server to verify the setup works
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ECE Paris RAG - Test Server</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f0f0; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            .success { color: #28a745; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎉 ECE Paris RAG - Test Server</h1>
            <p class="success">✅ Server is working correctly!</p>
            <p>If you can see this page, your setup is working.</p>
            <p>Now you can start the full file manager:</p>
            <code>python file_manager.py</code>
        </div>
    </body>
    </html>
    """)

@app.get("/api/test")
async def test_api():
    return {"message": "API is working!", "status": "success"}

if __name__ == "__main__":
    print("🚀 Starting test server on http://127.0.0.1:8002")
    print("📱 Open your browser and go to: http://127.0.0.1:8002")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")

