"""
File Management Web Interface for ECE Paris RAG System
Provides web interface for managing files, classifications, and database operations
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

from backend.database_manager import DatabaseManager


# Pydantic models for API
class FileMetadata(BaseModel):
    matiere: str
    sous_matiere: str
    enseignant: str
    semestre: str
    promo: str
    doc_id: Optional[str] = None
    doc_label: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class FileUpdate(BaseModel):
    matiere: Optional[str] = None
    sous_matiere: Optional[str] = None
    enseignant: Optional[str] = None
    semestre: Optional[str] = None
    promo: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class SearchRequest(BaseModel):
    query: Optional[str] = None
    matiere: Optional[str] = None
    sous_matiere: Optional[str] = None
    enseignant: Optional[str] = None
    semestre: Optional[str] = None
    promo: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(title="ECE Paris RAG File Manager", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global database manager
db_manager = None


def get_db_manager():
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    global db_manager
    db_manager = DatabaseManager()


@app.on_event("shutdown")
async def shutdown_event():
    """Close database on shutdown"""
    global db_manager
    if db_manager:
        db_manager.close()


# API Endpoints

@app.get("/")
async def root(request: Request):
    """Main file management interface"""
    return templates.TemplateResponse("file_manager_rag.html", {"request": request})


@app.get("/api/files")
async def get_files(
    matiere: Optional[str] = Query(None),
    sous_matiere: Optional[str] = Query(None),
    enseignant: Optional[str] = Query(None),
    semestre: Optional[str] = Query(None),
    promo: Optional[str] = Query(None),
    search: Optional[str] = Query(None)
):
    """Get files with optional filtering"""
    db = get_db_manager()
    
    if search:
        files = db.search_files(
            query=search,
            matiere=matiere,
            sous_matiere=sous_matiere,
            enseignant=enseignant,
            semestre=semestre,
            promo=promo
        )
    else:
        files = db.get_files_by_classification(
            matiere=matiere,
            sous_matiere=sous_matiere,
            enseignant=enseignant,
            semestre=semestre,
            promo=promo
        )
    
    return {"files": files}


@app.get("/api/files/{file_id}")
async def get_file(file_id: int):
    """Get specific file details"""
    db = get_db_manager()
    file_info = db.get_file_by_id(file_id)
    
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
    
    return file_info


@app.post("/api/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    matiere: str = Form(...),
    sous_matiere: str = Form(...),
    enseignant: str = Form(...),
    semestre: str = Form(...),
    promo: str = Form(...),
    doc_id: Optional[str] = Form(None),
    doc_label: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """
    Upload and classify a new file, then automatically process it for RAG.
    Steps:
    1. Save uploaded file to disk
    2. Add metadata to database (status: 'en attente')
    3. Launch background processing (chunking + embeddings)
    4. Update database status: 'en traitement' → 'traité'
    """
    import subprocess
    import threading

    db = get_db_manager()
    
    # Parse tags if provided
    tag_list = None
    if tags:
        try:
            tag_list = json.loads(tags)
        except json.JSONDecodeError:
            tag_list = [tag.strip() for t in tags.split(',') if t.strip()]
    
    # Save uploaded file
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
         # Add file entry in database (initially 'en attente')
        file_id = db.add_file(
            file_path=str(file_path),
            matiere=matiere,
            sous_matiere=sous_matiere, 
            enseignant=enseignant,
            semestre=semestre,
            promo=promo,
            doc_id=doc_id,
            doc_label=doc_label,
            description=description,
            tags=tag_list
        )
       # 🟡 Mark as "en traitement"
        db.update_file_status(file_id, "en traitement")

        # 🧠 Define background function to run enhanced_ingest.py
        def process_in_background():
            try:
                cmd = [
                    "python", "enhanced_ingest.py",
                    "--pdf", str(file_path),
                    "--matiere", matiere,
                    "--sous_matiere", sous_matiere,
                    "--enseignant", enseignant,
                    "--promo", str(promo),
                    "--semestre", semestre,
                    "--file_id", str(file_id)
                ]

                print(f"[🚀] Processing {file.filename} (ID={file_id}) ...") #?? Pas sur pour l'ID A LA MANO
                subprocess.run(cmd, check=True)

                # ✅ Update status once processed
                db.update_file_status(file_id, "traite")
                print(f"[✅] File {file_id} processed successfully.")
            except subprocess.CalledProcessError as e:
                db.update_file_status(file_id, f"erreur: {e.returncode}")
                print(f"[❌] Error processing file {file_id}: {e}")

        # 🚀 Launch processing in background thread
        thread = threading.Thread(target=process_in_background, daemon=True) #?? Recheck pour remplacer par threading.Thread(target=process_in_background, args=(file_id, file_path, metadata)).start()

        thread.start()
        return {
            "message": "File uploaded successfully, processing started in background",
            "file_id": file_id,
            "filename": file.filename,
            "status": "en traitement"
        }
    
    except Exception as e:
        # Clean up uploaded file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/files/{file_id}")
async def update_file(file_id: int, update_data: FileUpdate):
    """Update file metadata"""
    db = get_db_manager()
    
    success = db.update_file_metadata(
        file_id=file_id,
        matiere=update_data.matiere,
        sous_matiere=update_data.sous_matiere,
        enseignant=update_data.enseignant,
        semestre=update_data.semestre,
        promo=update_data.promo,
        description=update_data.description,
        tags=update_data.tags
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="File not found or no changes made")
    
    return {"message": "File updated successfully"}


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: int):
    """Delete a file and all its chunks"""
    db = get_db_manager()
    
    success = db.delete_file(file_id)
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    
    return {"message": "File deleted successfully"}


@app.get("/api/classifications")
async def get_classifications():
    """Get all unique classification values"""
    db = get_db_manager()
    classifications = db.get_unique_classifications()
    return classifications


@app.get("/api/summary")
async def get_summary():
    """Get database summary"""
    db = get_db_manager()
    summary = db.get_file_summary()
    classifications = db.get_unique_classifications()
    
    return {
        "file_count": len(summary),
        "total_chunks": sum(f.get('actual_chunks', 0) for f in summary),
        "processed_files": len([f for f in summary if f.get('is_processed', False)]),
        "classifications": classifications
    }


@app.post("/api/process/{file_id}")
async def process_file(file_id: int):
    """Process a file for RAG (placeholder for future implementation)"""
    # This would integrate with your existing RAG processing pipeline
    # For now, just mark as processed
    db = get_db_manager()
    
    file_info = db.get_file_by_id(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
    
    # TODO: Implement actual RAG processing
    # This would call your enhanced_ingest.py or similar processing
    
    return {"message": f"Processing initiated for file {file_id}"}


@app.get("/api/export")
async def export_database():
    """Export database to vector_db.json format"""
    db = get_db_manager()
    
    try:
        from backend.database_manager import export_to_vector_db
        export_to_vector_db(db, "vector_db_export.json")
        return {"message": "Database exported successfully", "file": "vector_db_export.json"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# Serve static files
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting ECE Paris RAG File Manager...")
    print("📱 Open your browser and go to: http://127.0.0.1:8001")
    print("📱 Or try: http://localhost:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
