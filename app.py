
import streamlit as st

st.title("Dependency Check")

try:
    import dlib
    st.success(f"dlib installed: {dlib.__version__}")
except Exception as e:
    st.error(f"dlib error: {e}")

try:
    import face_recognition
    st.success("face_recognition installed successfully")
except Exception as e:
    st.error(f"face_recognition error: {e}")
    import io
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import face_recognition

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
STUDENTS_DIR = DATA_DIR / "students"
ATTENDANCE_DIR = DATA_DIR / "attendance"
STUDENTS_DIR.mkdir(parents=True, exist_ok=True)
ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["Data Visualization", "Machine Learning", "App Development"]
ATTENDANCE_COLUMNS = ["Enrollment", "Name", "Class", "Subject", "Time Stamp"]

st.set_page_config(page_title="Smart Attendance System", page_icon="📸", layout="wide")

def load_students():
    records = []
    for file in STUDENTS_DIR.glob("*.pkl"):
        try:
            with file.open("rb") as handle:
                record = pickle.load(handle)
            if isinstance(record, dict) and record.get("encodings"):
                records.append(record)
        except Exception:
            continue
    return records

def image_to_rgb(uploaded_image):
    image = Image.open(io.BytesIO(uploaded_image.getvalue())).convert("RGB")
    return np.asarray(image)

def extract_one_face_encoding(rgb_image):
    locations = face_recognition.face_locations(rgb_image, model="hog")
    if len(locations) != 1:
        return None, len(locations)
    encodings = face_recognition.face_encodings(rgb_image, locations)
    return (encodings[0] if encodings else None), len(locations)

def save_student(name, enrollment, student_class, encoding):
    safe_name = "_".join(name.strip().split())
    path = STUDENTS_DIR / f"{enrollment.strip()}_{safe_name}.pkl"
    record = {
        "name": name.strip(),
        "enrollment_id": enrollment.strip(),
        "class": student_class.strip(),
        "encodings": [encoding],
    }
    with path.open("wb") as handle:
        pickle.dump(record, handle)
    return path

def attendance_file(subject, day=None):
    day = day or datetime.now().strftime("%Y-%m-%d")
    safe_subject = subject.replace("/", "-").replace("\\\\", "-")
    return ATTENDANCE_DIR / f"attendance_{safe_subject}_{day}.xlsx"

def mark_attendance(student, subject):
    path = attendance_file(subject)
    if path.exists():
        try:
            df = pd.read_excel(path)
        except Exception:
            df = pd.DataFrame(columns=ATTENDANCE_COLUMNS)
    else:
        df = pd.DataFrame(columns=ATTENDANCE_COLUMNS)

    enrollment = str(student["enrollment_id"]).strip()
    if not df.empty and df["Enrollment"].astype(str).str.strip().eq(enrollment).any():
        return False, path

    row = {
        "Enrollment": enrollment,
        "Name": student["name"],
        "Class": student.get("class", "N/A"),
        "Subject": subject,
        "Time Stamp": datetime.now().strftime("%I:%M:%S %p"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(path, index=False)
    return True, path

def match_student(encoding, records, tolerance=0.48):
    known_encodings, owners = [], []
    for record in records:
        for known in record.get("encodings", []):
            known_encodings.append(np.asarray(known))
            owners.append(record)
    if not known_encodings:
        return None, None
    distances = face_recognition.face_distance(known_encodings, encoding)
    best = int(np.argmin(distances))
    if float(distances[best]) <= tolerance:
        return owners[best], float(distances[best])
    return None, float(distances[best])

st.title("📸 AI-Based Smart Attendance System")
st.caption("Face recognition • Subject-wise attendance • Monthly reports")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a page", ["Dashboard", "Enroll Student", "Mark Attendance", "View Attendance", "Monthly Report", "Students"])
    st.divider()
    st.caption("Camera access is requested by your browser when you take a photo.")
    st.warning("Demo hosting note: Streamlit Community Cloud's local files are not durable. Use a persistent database/storage service before real institutional use.")

students = load_students()
today = datetime.now().strftime("%Y-%m-%d")

if page == "Dashboard":
    st.subheader("Dashboard")
    attendance_files = list(ATTENDANCE_DIR.glob("attendance_*.xlsx"))
    total_rows = 0
    today_rows = 0
    for file in attendance_files:
        try:
            df = pd.read_excel(file)
            total_rows += len(df)
            if today in file.name:
                today_rows += len(df)
        except Exception:
            pass
    c1, c2, c3 = st.columns(3)
    c1.metric("Registered students", len(students))
    c2.metric("Today's attendance entries", today_rows)
    c3.metric("All attendance entries", total_rows)
    st.info("Use the sidebar to enroll students, mark attendance from a browser camera snapshot, or view reports.")
    st.markdown("**Subjects configured:** " + " · ".join(SUBJECTS))

elif page == "Enroll Student":
    st.subheader("Enroll a student")
    st.write("Enter student details, then capture a clear image showing exactly one face.")
    with st.form("enroll_form"):
        name = st.text_input("Student name")
        enrollment = st.text_input("Enrollment ID")
        student_class = st.text_input("Class / section")
        submitted = st.form_submit_button("Save student details")
    if submitted:
        if not all([name.strip(), enrollment.strip(), student_class.strip()]):
            st.error("Please fill in all fields.")
        elif any(str(s.get("enrollment_id")) == enrollment.strip() for s in students):
            st.error("This enrollment ID is already registered.")
        else:
            st.session_state["pending_student"] = {
                "name": name.strip(), "enrollment": enrollment.strip(), "student_class": student_class.strip()
            }
            st.success("Details saved for this session. Capture the student's face below.")
    pending = st.session_state.get("pending_student")
    if pending:
        st.info(f"Ready to enroll: {pending['name']} ({pending['enrollment']})")
        photo = st.camera_input("Capture student face", key="enroll_camera")
        if photo:
            if st.button("Process face and enroll", type="primary"):
                with st.spinner("Detecting face and generating encoding..."):
                    rgb = image_to_rgb(photo)
                    encoding, count = extract_one_face_encoding(rgb)
                if count != 1 or encoding is None:
                    st.error(f"Expected exactly one face; detected {count}. Please retake the photo.")
                else:
                    save_student(pending["name"], pending["enrollment"], pending["student_class"], encoding)
                    st.session_state.pop("pending_student", None)
                    st.success("Student enrolled successfully.")
                    st.rerun()

elif page == "Mark Attendance":
    st.subheader("Mark attendance using face recognition")
    subject = st.selectbox("Select subject", SUBJECTS)
    if not students:
        st.warning("No students are enrolled yet. Enroll a student first.")
    photo = st.camera_input("Capture face to mark attendance", key="attendance_camera")
    if photo and st.button("Recognize and mark attendance", type="primary"):
        with st.spinner("Recognizing face..."):
            rgb = image_to_rgb(photo)
            encoding, count = extract_one_face_encoding(rgb)
        if count != 1 or encoding is None:
            st.error(f"Expected exactly one face; detected {count}. Please take another photo.")
        else:
            student, distance = match_student(encoding, students)
            if student is None:
                st.error("Face not recognized. Ask an administrator to enroll this student.")
            else:
                added, file_path = mark_attendance(student, subject)
                if added:
                    st.success(f"Attendance marked for {student['name']} ({student['enrollment_id']}) in {subject}.")
                else:
                    st.warning(f"Attendance was already marked today for {student['name']} in {subject}.")
                st.caption(f"Match distance: {distance:.3f}")

elif page == "View Attendance":
    st.subheader("Attendance records")
    files = sorted(ATTENDANCE_DIR.glob("attendance_*.xlsx"), reverse=True)
    if not files:
        st.info("No attendance records available yet.")
    else:
        selected = st.selectbox("Choose attendance file", files, format_func=lambda p: p.name)
        try:
            df = pd.read_excel(selected)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download Excel", selected.read_bytes(), file_name=selected.name,
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as exc:
            st.error(f"Could not read attendance file: {exc}")

elif page == "Monthly Report":
    st.subheader("Monthly attendance report")
    subject = st.selectbox("Subject", SUBJECTS, key="report_subject")
    month = st.text_input("Month (YYYY-MM)", value=datetime.now().strftime("%Y-%m"))
    if st.button("Generate report"):
        try:
            datetime.strptime(month, "%Y-%m")
            frames, session_dates = [], set()
            for file in ATTENDANCE_DIR.glob(f"attendance_{subject}_*.xlsx"):
                date_str = file.stem.rsplit("_", 1)[-1]
                if date_str.startswith(month):
                    try:
                        frame = pd.read_excel(file)
                        frame["Date"] = date_str
                        frames.append(frame)
                        session_dates.add(date_str)
                    except Exception:
                        pass
            if not frames:
                st.warning("No attendance data found for this subject and month.")
            else:
                all_data = pd.concat(frames, ignore_index=True)
                unique = all_data.drop_duplicates(subset=["Enrollment", "Date"])
                report = unique.groupby(["Enrollment", "Name", "Class"], as_index=False)["Date"].nunique()
                report.rename(columns={"Date": "Total Attendance"}, inplace=True)
                report["Attendance Percentage"] = (report["Total Attendance"] / len(session_dates) * 100).round(2)
                st.dataframe(report, use_container_width=True, hide_index=True)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    report.to_excel(writer, index=False)
                st.download_button("Download monthly report", output.getvalue(),
                                   file_name=f"monthly_report_{subject}_{month}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except ValueError:
            st.error("Enter the month in YYYY-MM format.")

elif page == "Students":
    st.subheader("Registered students")
    if not students:
        st.info("No students registered.")
    else:
        student_table = pd.DataFrame([{
            "Enrollment": s.get("enrollment_id", ""),
            "Name": s.get("name", ""),
            "Class": s.get("class", ""),
            "Face samples": len(s.get("encodings", [])),
        } for s in students])
        st.dataframe(student_table, use_container_width=True, hide_index=True)
        st.caption("Student face encodings are sensitive biometric data. Do not expose this folder in a public repository.")
