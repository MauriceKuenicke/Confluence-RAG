from sqlalchemy import create_engine
import os


def get_backend_db_engine():
    engine = create_engine(os.getenv("APP__DB__SQL_ALCHEMY_CONN"),
                           echo=False,
                           pool_pre_ping=True)
    return engine