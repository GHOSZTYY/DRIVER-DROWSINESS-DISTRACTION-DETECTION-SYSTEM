import sqlite3
import os
from datetime import datetime

# Database file lives in the data/sessions folder
DB_PATH = os.path.join("data", "sessions", "driver_safety.db")


def get_connection():
    """Open and return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn


def create_tables():
    """
    Create the sessions and events tables if they don't exist yet.
    Call this once when the app starts.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # One row per driving session
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_name   TEXT    NOT NULL,
            started_at    TEXT    NOT NULL,
            ended_at      TEXT,
            total_alerts  INTEGER DEFAULT 0
        )
    """)

    # One row per alert event
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    INTEGER NOT NULL,
            timestamp     TEXT    NOT NULL,
            alert_level   TEXT    NOT NULL,
            ear_score     REAL,
            mar_score     REAL,
            yaw_angle     REAL,
            phone_detected INTEGER DEFAULT 0,
            emotion_label TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    conn.commit()
    conn.close()
    print("Tables created successfully.")


def start_session(driver_name):
    """
    Create a new driving session row.

    Input:  driver_name — string e.g. 'Mridul'
    Output: session_id  — integer (use this in all log_event() calls)
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO sessions (driver_name, started_at)
        VALUES (?, ?)
    """, (driver_name, datetime.now().isoformat()))

    session_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"Session started for {driver_name} — session_id: {session_id}")
    return session_id


def log_event(session_id, level, ear=None, mar=None,
              yaw=None, phone=False, emotion=None):
    """
    Log one alert event to the database.

    Input:  session_id — from start_session()
            level      — 'low', 'medium', or 'critical'
            ear        — EAR float from eye_ear.py
            mar        — MAR float from mouth_mar.py
            yaw        — yaw angle from head_pose.py
            phone      — True/False from phone_detect.py
            emotion    — string from emotion.py e.g. 'angry'
    Output: None (writes to DB)
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO events
            (session_id, timestamp, alert_level,
             ear_score, mar_score, yaw_angle,
             phone_detected, emotion_label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        datetime.now().isoformat(),
        level,
        ear,
        mar,
        yaw,
        1 if phone else 0,
        emotion
    ))

    conn.commit()
    conn.close()


def end_session(session_id, total_alerts):
    """
    Mark a session as finished.

    Input:  session_id    — from start_session()
            total_alerts  — how many alerts fired this session
    Output: None (updates DB)
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE sessions
        SET ended_at     = ?,
            total_alerts = ?
        WHERE id = ?
    """, (datetime.now().isoformat(), total_alerts, session_id))

    conn.commit()
    conn.close()
    print(f"Session {session_id} ended. Total alerts: {total_alerts}")


def get_all_sessions():
    """
    Fetch all past sessions — used by Sarthak's dashboard.
    Output: list of session rows
    """
    conn    = get_connection()
    cursor  = conn.cursor()
    cursor.execute("SELECT * FROM sessions ORDER BY started_at DESC")
    rows    = cursor.fetchall()
    conn.close()
    return rows


def get_events_for_session(session_id):
    """
    Fetch all events for one session — used by Sarthak's dashboard.
    Output: list of event rows
    """
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM events
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))
    rows   = cursor.fetchall()
    conn.close()
    return rows


# Test: run this file directly to verify everything works
if __name__ == "__main__":
    print("Testing database...")

    # Step 1: create tables
    create_tables()

    # Step 2: start a fake session
    sid = start_session("Mridul")

    # Step 3: log some fake events
    log_event(sid, "medium",   ear=0.18, mar=0.3, yaw=5.1,  phone=False, emotion="neutral")
    log_event(sid, "critical", ear=0.12, mar=0.4, yaw=28.5, phone=True,  emotion="angry")
    log_event(sid, "medium",   ear=0.19, mar=0.5, yaw=3.2,  phone=False, emotion="neutral")

    # Step 4: end the session
    end_session(sid, total_alerts=3)

    # Step 5: read it back
    print("\n--- All Sessions ---")
    for row in get_all_sessions():
        print(dict(row))

    print("\n--- Events for session 1 ---")
    for row in get_events_for_session(sid):
        print(dict(row))

    print("\nAll done! Check data/sessions/driver_safety.db")