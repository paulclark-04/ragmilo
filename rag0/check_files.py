#!/usr/bin/env python3
"""
Check database files status
"""

from database_manager import DatabaseManager

def check_files():
    with DatabaseManager() as db:
        print("📊 Database Status Check")
        print("=" * 40)
        
        # Check all files
        all_files = db.get_files_by_classification()
        print(f"Total files in database: {len(all_files)}")
        
        # Check Informatique files specifically
        info_files = db.get_files_by_classification(matiere="Informatique")
        print(f"\n📚 Informatique files: {len(info_files)}")
        
        for file_info in info_files:
            status = "✅ Processed" if file_info.get('is_processed', False) else "⏳ Not processed"
            chunks = file_info.get('chunk_count', 0)
            print(f"  - {file_info['filename']}: {status} ({chunks} chunks)")
        
        # Check if chunks exist for Informatique
        chunks = db.get_rag_chunks_by_classification(matiere="Informatique")
        print(f"\n📄 Informatique chunks: {len(chunks)}")
        
        if chunks:
            print("Sample chunks:")
            for chunk in chunks[:3]:
                print(f"  - {chunk['chunk_id']}: {chunk['chunk_text'][:50]}...")
        else:
            print("  No chunks found - files need to be processed!")

if __name__ == "__main__":
    check_files()
