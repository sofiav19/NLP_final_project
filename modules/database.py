# modules/database.py
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

PROFILE_DB = 'static/profiles/profiles.json'
profiles = {}

# Ensure directory exists
os.makedirs(os.path.dirname(PROFILE_DB), exist_ok=True)

def load_profiles():
    global profiles
    if os.path.exists(PROFILE_DB):
        with open(PROFILE_DB, 'r', encoding='utf-8') as f:
            profiles.update(json.load(f))

def save_profiles():
    with open(PROFILE_DB, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2)

def save_profile(profile, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # <-- ✅ ensure folder exists
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2)

def add_profile(name, profile):
    profiles[name] = profile

def is_duplicate(name):
    return name in profiles

load_profiles()

class Database:
    def __init__(self, db_path="data/book_profiles.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
        self._lock = threading.Lock()
        self._initialize_db()
        
    def _get_connection(self):
        """Get a database connection with retry logic"""
        if self._connection is None:
            try:
                self._connection = sqlite3.connect(
                    str(self.db_path),
                    timeout=30,  # Increase timeout for busy database
                    check_same_thread=False  # Allow multi-threading
                )
                self._connection.row_factory = sqlite3.Row
                # Enable foreign keys and set busy timeout
                self._connection.execute("PRAGMA foreign_keys = ON")
                self._connection.execute("PRAGMA busy_timeout = 5000")
            except sqlite3.Error as e:
                print(f"Database connection error: {str(e)}")
                raise RuntimeError(f"Could not connect to database: {str(e)}")
        return self._connection
        
    def _initialize_db(self):
        """Initialize database with proper error handling"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS book_profiles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL UNIQUE,
                        author TEXT,
                        genre TEXT,
                        emotional_profile TEXT,
                        topic_profile TEXT,
                        character_profiles TEXT,
                        analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_path TEXT UNIQUE,
                        file_hash TEXT UNIQUE,
                        processing_status TEXT DEFAULT 'pending',
                        error_message TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Add indices for better query performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON book_profiles(title)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_author ON book_profiles(author)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_genre ON book_profiles(genre)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON book_profiles(processing_status)")
                
                # Add trigger to update last_updated timestamp
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS update_timestamp 
                    AFTER UPDATE ON book_profiles
                    BEGIN
                        UPDATE book_profiles SET last_updated = CURRENT_TIMESTAMP 
                        WHERE id = NEW.id;
                    END
                """)
                
        except sqlite3.Error as e:
            print(f"Database initialization error: {str(e)}")
            raise RuntimeError(f"Could not initialize database: {str(e)}")
            
    def _execute_with_retry(self, operation, *args, max_retries=3, **kwargs):
        """Execute database operation with retry logic"""
        last_error = None
        for attempt in range(max_retries):
            try:
                with self._lock:  # Ensure thread safety
                    with self._get_connection() as conn:
                        cursor = conn.cursor()
                        result = operation(cursor, *args, **kwargs)
                        conn.commit()
                        return result
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower():
                    last_error = e
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                raise
            except sqlite3.Error as e:
                print(f"Database error: {str(e)}")
                raise RuntimeError(f"Database operation failed: {str(e)}")
                
        raise RuntimeError(f"Database operation failed after {max_retries} attempts: {str(last_error)}")
        
    def save_profile(self, profile_data):
        """Save book profile with improved error handling"""
        try:
            def _save(cursor, data):
                cursor.execute("""
                    INSERT OR REPLACE INTO book_profiles (
                        title, author, genre, emotional_profile, 
                        topic_profile, character_profiles, file_path,
                        file_hash, processing_status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['title'],
                    data.get('author'),
                    data.get('genre'),
                    json.dumps(data.get('emotional_profile', {})),
                    json.dumps(data.get('topic_profile', {})),
                    json.dumps(data.get('character_profiles', {})),
                    data.get('file_path'),
                    data.get('file_hash'),
                    data.get('processing_status', 'completed'),
                    data.get('error_message')
                ))
                return cursor.lastrowid
                
            return self._execute_with_retry(_save, profile_data)
            
        except Exception as e:
            print(f"Error saving profile: {str(e)}")
            raise RuntimeError(f"Could not save book profile: {str(e)}")
            
    def get_profile(self, identifier):
        """Get book profile by title or file path with improved error handling"""
        try:
            def _get(cursor, id_value):
                cursor.execute("""
                    SELECT * FROM book_profiles 
                    WHERE title = ? OR file_path = ? OR file_hash = ?
                """, (id_value, id_value, id_value))
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
                
            return self._execute_with_retry(_get, identifier)
            
        except Exception as e:
            print(f"Error retrieving profile: {str(e)}")
            raise RuntimeError(f"Could not retrieve book profile: {str(e)}")
            
    def update_processing_status(self, identifier, status, error_message=None):
        """Update processing status with improved error handling"""
        try:
            def _update(cursor, id_value, new_status, error):
                cursor.execute("""
                    UPDATE book_profiles 
                    SET processing_status = ?, error_message = ?
                    WHERE title = ? OR file_path = ? OR file_hash = ?
                """, (new_status, error, id_value, id_value, id_value))
                return cursor.rowcount > 0
                
            return self._execute_with_retry(_update, identifier, status, error_message)
            
        except Exception as e:
            print(f"Error updating status: {str(e)}")
            raise RuntimeError(f"Could not update processing status: {str(e)}")
            
    def get_all_profiles(self, status=None):
        """Get all profiles with optional status filter"""
        try:
            def _get_all(cursor, filter_status):
                if filter_status:
                    cursor.execute("""
                        SELECT * FROM book_profiles 
                        WHERE processing_status = ?
                        ORDER BY last_updated DESC
                    """, (filter_status,))
                else:
                    cursor.execute("""
                        SELECT * FROM book_profiles 
                        ORDER BY last_updated DESC
                    """)
                return [dict(row) for row in cursor.fetchall()]
                
            return self._execute_with_retry(_get_all, status)
            
        except Exception as e:
            print(f"Error retrieving profiles: {str(e)}")
            raise RuntimeError(f"Could not retrieve book profiles: {str(e)}")
            
    def delete_profile(self, identifier):
        """Delete profile with improved error handling"""
        try:
            def _delete(cursor, id_value):
                cursor.execute("""
                    DELETE FROM book_profiles 
                    WHERE title = ? OR file_path = ? OR file_hash = ?
                """, (id_value, id_value, id_value))
                return cursor.rowcount > 0
                
            return self._execute_with_retry(_delete, identifier)
            
        except Exception as e:
            print(f"Error deleting profile: {str(e)}")
            raise RuntimeError(f"Could not delete book profile: {str(e)}")
            
    def close(self):
        """Close database connection"""
        if self._connection is not None:
            try:
                self._connection.close()
            except sqlite3.Error as e:
                print(f"Error closing database: {str(e)}")
            finally:
                self._connection = None

# Instantiate globally
db = Database()
