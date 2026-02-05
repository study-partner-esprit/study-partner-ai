import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.course_ingestion.agent import ingest_course
from pprint import pprint

# Replace with your test file(s) - can be PDF or TXT
pdf_files = ["./01.pdf"]
course_title = "course"

# Check if PDF exists
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
full_pdf_files = []
for pdf in pdf_files:
    full_path = os.path.join(script_dir, pdf)
    print(f"Looking for PDF at: {full_path}")
    if not os.path.exists(full_path):
        print(f"Test file not found at {full_path}. Please provide a valid file path (PDF or TXT).")
        exit(1)
    full_pdf_files.append(full_path)

# Call the ingestion agent
course_id = ingest_course(course_title, full_pdf_files)

print("Course saved with ID:", course_id)

# Optionally, fetch from DB to inspect
from agents.course_ingestion.services.database_service import DatabaseService
db = DatabaseService()
course_json = db.get_course(course_id)
pprint(course_json)
