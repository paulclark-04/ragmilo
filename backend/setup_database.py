#!/usr/bin/env python3
"""
Setup script for ECE Paris RAG Database Integration
Initializes the database and provides migration tools
"""

import argparse
import json
from pathlib import Path
from backend.database_manager import DatabaseManager, import_from_vector_db, export_to_vector_db


def setup_database(db_path: str = "rag_database.db"):
    """Initialize the database with default configuration"""
    print("🔧 Initializing database...")
    
    with DatabaseManager(db_path) as db:
        print("✅ Database initialized successfully")
        
        # Show initial stats
        summary = db.get_file_summary()
        print(f"📊 Database contains {len(summary)} files")
        
        return db


def migrate_existing_data(vector_db_path: str, db_path: str = "rag_database.db"):
    """Migrate existing vector_db.json to the database"""
    if not Path(vector_db_path).exists():
        print(f"❌ File not found: {vector_db_path}")
        return False
    
    print(f"🔄 Migrating data from {vector_db_path}...")
    
    with DatabaseManager(db_path) as db:
        try:
            import_from_vector_db(db, vector_db_path)
            print("Migration completed successfully")
            

            summary = db.get_file_summary()
            print(f"Database now contains {len(summary)} files")
            print(f"Total chunks: {sum(f.get('actual_chunks', 0) for f in summary)}")
            
            return True
        except Exception as e:
            print(f"Migration failed: {e}")
            return False


def export_database(db_path: str = "rag_database.db", output_path: str = "vector_db_export.json"):
    """Export database to vector_db.json format"""
    print(f"📤 Exporting database to {output_path}...")
    
    with DatabaseManager(db_path) as db:
        try:
            export_to_vector_db(db, output_path)
            print("Export completed successfully")
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False


def show_database_info(db_path: str = "rag_database.db"):
    """Show database information and statistics"""
    print("Database Information")
    print("=" * 50)
    
    with DatabaseManager(db_path) as db:
        # File summary
        summary = db.get_file_summary()
        print(f"📁 Total files: {len(summary)}")
        print(f"📄 Total chunks: {sum(f.get('actual_chunks', 0) for f in summary)}")
        print(f"✅ Processed files: {len([f for f in summary if f.get('is_processed', False)])}")
        
        # Classifications
        classifications = db.get_unique_classifications()
        print(f"\n🏷️  Classifications:")
        for field, values in classifications.items():
            print(f"   {field}: {len(values)} values")
            if len(values) <= 10:  
                print(f"      {', '.join(values)}")
            else:
                print(f"      {', '.join(values[:5])}... (+{len(values)-5} more)")
        
        
        print(f"\n📅 Recent files:")
        for file_info in summary[:5]:
            status = "✅" if file_info.get('is_processed', False) else "⏳"
            print(f"   {status} {file_info['filename']} - {file_info['matiere']} - {file_info['enseignant']}")


def interactive_setup():
    """Interactive setup wizard"""
    print("🎯 ECE Paris RAG Database Setup")
    print("=" * 40)
    
    
    vector_db_path = Path("vector_db.json")
    if vector_db_path.exists():
        print(f"📁 Found existing vector_db.json ({vector_db_path.stat().st_size} bytes)")
        migrate = input("Do you want to migrate this data to the database? (y/n): ").lower().strip()
        
        if migrate == 'y':
            if migrate_existing_data(str(vector_db_path)):
                print("✅ Migration completed!")
            else:
                print("❌ Migration failed!")
                return
    
    # Initialize database
    setup_database()
    
    # Show information
    show_database_info()
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Launch file manager: python file_manager.py")
    print(f"   2. Use enhanced ingestion: python enhanced_ingest.py --help")
    print(f"   3. View database info: python setup_database.py --info")


def main():
    parser = argparse.ArgumentParser(description="ECE Paris RAG Database Setup")
    parser.add_argument("--setup", action="store_true", help="Initialize database")
    parser.add_argument("--migrate", help="Migrate from vector_db.json file")
    parser.add_argument("--export", help="Export database to JSON file")
    parser.add_argument("--info", action="store_true", help="Show database information")
    parser.add_argument("--interactive", action="store_true", help="Run interactive setup")
    parser.add_argument("--db-path", default="rag_database.db", help="Database file path")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_setup()
    elif args.setup:
        setup_database(args.db_path)
    elif args.migrate:
        migrate_existing_data(args.migrate, args.db_path)
    elif args.export:
        export_database(args.db_path, args.export)
    elif args.info:
        show_database_info(args.db_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

