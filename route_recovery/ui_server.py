"""Separate Laptop 2 UI: python -m uvicorn route_recovery.ui_server:app --port 8080."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse

app = FastAPI(title='PathSnap UI')


@app.get('/', response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / 'index.html').read_text(encoding='utf-8')


@app.get('/app.js')
def ui_script():
    return FileResponse(Path(__file__).parent / 'app.js', media_type='text/javascript')
