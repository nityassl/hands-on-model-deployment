from app import app
import os
from waitress import serve

if __name__ == '__main__':
    # app.run(port=os.getenv("PORT", default=5000))
    serve(app, listen=os.getenv("HOST", default="127.0.0.1:8000"))