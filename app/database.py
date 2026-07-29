import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Define where our SQLite database file will live (stored locally as receipts.db)
DATABASE_URL = "sqlite:///./receipts.db"

# Create the SQLAlchemy engine. 
# check_same_thread=False is required for SQLite because FastAPI handles requests 
# in multiple threads, and we want to share the connection safely.
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a session creator. Every time we need to talk to the database, 
# we will instantiate a new session from this factory.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This is the Base class that all database models (tables) will inherit from.
Base = declarative_base()


# Dependency injection helper to get a database session for each request.
# It automatically closes the connection when the web request is done.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
