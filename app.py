import io
import re
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import face_recognition

st.set_page_config(page_title="SnapClass | AI Attendance", page_icon="♟️", layout="wide")

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
STUDENTS_DIR = DATA_DIR / "students"
ATTENDANCE_DIR = DATA_DIR / "attendance"
STUDENTS_DIR.mkdir(parents=True, exist_ok=True)
ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["Data Visualization", "Machine Learning", "App Development"]
ATTENDANCE_COLUMNS = ["Enrollment", "Name", "Class", "Subject", "Time Stamp"]

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
    safe_enrollment = re.sub(r"[^A-Za-z0-9_-]", "_", enrollment.strip())
    path = STUDENTS_DIR / f"{safe_enrollment}_{safe_name}.pkl"
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
    safe_subject = subject.replace("/", "-").replace("\\", "-")
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

def valid_person_name(value):
    value = value.strip()
    return len(value) >= 2 and len(value) <= 60 and re.fullmatch(
        r"[^\W\d_]+(?:[ '-][^\W\d_]+)*", value, flags=re.UNICODE
    ) is not None

def valid_enrollment(value):
    return 3 <= len(value.strip()) <= 30 and re.fullmatch(
        r"[A-Za-z0-9/-]+", value.strip()
    ) is not None

def valid_class(value):
    return 2 <= len(value.strip()) <= 40 and re.fullmatch(
        r"[A-Za-z0-9 .&()/-]+", value.strip()
    ) is not None
 # SNAPCLASS DESIGN

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Archivo+Black&display=swap');

.stApp {
    background: #f4f6ff;
    color: #202a3d;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1240px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

.snap-logo {
    font-family: 'Archivo Black', sans-serif;
    font-size: 30px;
    color: #ffffff;
    background: #050505;
    padding: 18px 24px;
    border-radius: 14px;
    margin-bottom: 30px;
}

.snap-logo span {
    background: #ffe600;
    color: #111111;
    padding: 5px 10px;
    border-radius: 8px;
}

.hero-wrap {
    text-align: center;
    padding: 55px 15px 35px;
}

.pill {
    display: inline-block;
    background: white;
    border: 1px solid #d4d9e7;
    border-radius: 30px;
    color: #6268f4;
    font-weight: 800;
    padding: 10px 22px;
    margin-bottom: 25px;
}

.hero-title {
    font-family: 'Archivo Black', sans-serif;
    font-size: clamp(42px, 6vw, 78px);
    line-height: 1;
    letter-spacing: -3px;
    color: #202a3d;
    margin-bottom: 25px;
}

.hero-title em {
    color: #6268f4;
    font-style: normal;
}

.hero-sub {
    font-size: 18px;
    line-height: 1.7;
    color: #667895;
    max-width: 780px;
    margin: auto;
}

.section-head {
    font-family: 'Archivo Black', sans-serif;
    font-size: 34px;
    color: #202a3d;
    margin-bottom: 16px;
}

.step-label {
    color: #6268f4;
    font-weight: 800;
    letter-spacing: 2px;
}

.feature-card {
    background: white;
    border: 1px solid #e1e5f0;
    border-radius: 22px;
    padding: 25px;
    min-height: 150px;
}

.feature-card h3 {
    font-family: 'Archivo Black', sans-serif;
    color: #202a3d;
}

.feature-card p {
    color: #687994;
    line-height: 1.7;
}

div[data-testid="stTextInput"] input {
    background: white;
    color: #202a3d;
    border: 1px solid #d9deeb;
    border-radius: 10px;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: white;
    border-radius: 10px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: #6268f4;
    color: white;
    border: none;
    border-radius: 30px;
    min-height: 44px;
    font-weight: 800;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #4d54e7;
    color: white;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e1e5f0;
    border-radius: 18px;
    padding: 18px;
}

div[data-testid="stMetricValue"] {
    color: #202a3d;
}

div[data-testid="stMetricLabel"] {
    color: #667895;
}

div[data-testid="stCameraInput"] {
    background: white;
    border: 1px dashed #aeb8d0;
    border-radius: 16px;
    padding: 12px;
}

[data-testid="stDataFrame"] {
    background: white;
    border-radius: 14px;
}
</style>


# NAVIGATION

if "snap_page" not in st.session_state:
    st.session_state.snap_page = "Home"

nav1, nav2, nav3, nav4, nav5 = st.columns([2, 1, 1, 1, 1.7])

with nav1:
    st.markdown(
        '<div class="snap-logo"><span>♟</span> SnapClass</div>',
        unsafe_allow_html=True
    )

for col, label in [
    (nav2, "Home"),
    (nav3, "Features"),
    (nav4, "Journey"),
    (nav5, "Start AI Attendance")
]:
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.snap_page = (
                "Dashboard" if label == "Start AI Attendance" else label
            )
            st.rerun()

page = st.session_state.snap_page

students = load_students()
today = datetime.now().strftime("%Y-%m-%d")


def page_title(kicker, title, subtitle=""):
    st.markdown(
        f'<div class="step-label">{kicker}</div>'
        f'<div class="section-head">{title}</div>',
        unsafe_allow_html=True
    )
    if subtitle:
        st.markdown(
            f'<p style="color:#687994;font-size:17px;line-height:1.7">'
            f'{subtitle}</p>',
            unsafe_allow_html=True
        )
     # HOME PAGE

if page == "Home":

    st.markdown("""
    <div class="hero-wrap">
        <div class="pill">Welcome to SnapClass</div>

        <div class="hero-title">
            AI Powered<br>
            Attendance <em>System</em>
        </div>

        <div class="hero-sub">
            Revolutionizing the classroom with next-gen computer vision.
            Fast, simple and secure subject-wise attendance powered by
            face recognition.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
           st.subheader("📋 Attendance")
            <p>Manage attendance and download reports easily.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            st.subheader("📸 Face Recognition")
            <p>AI-powered student identification and verification.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
           st.subheader("👨‍🏫 Teacher Portal")
            <p>Manage students and classroom attendance.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    if st.button("Start AI Attendance →", type="primary",
                 use_container_width=True):
        st.session_state.snap_page = "Dashboard"
        st.rerun()

    st.markdown(
        '<div class="section-head" style="text-align:center">'
        'Everything you need in one place</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="feature-card">
            <h3>01. Secure Enrollment</h3>
            <p>Register students with their face profiles.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="feature-card">
            <h3>02. Smart Attendance</h3>
            <p>Recognize faces and mark attendance automatically.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="feature-card">
            <h3>03. Monthly Reports</h3>
            <p>Generate attendance reports and export Excel files.</p>
        </div>
        """, unsafe_allow_html=True)


# FEATURES PAGE

elif page == "Features":

    page_title(
        "SNAPCLASS FEATURES",
        "A smarter way to manage attendance",
        "Explore the tools available in your teacher workspace."
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="feature-card">
           st.subheader("🔐 Student Enrollment")
            <p>Register students using face recognition.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="feature-card">
            <h3>Face Recognition</h3>
            <p>Identify students through their registered face.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="feature-card">
           st.subheader("📊 Reports")
            <p>View attendance and download reports.</p>
        </div>
        """, unsafe_allow_html=True)


# JOURNEY PAGE

elif page == "Journey":

    page_title(
        "TEACHER'S JOURNEY",
        "From enrollment to attendance",
        "A simple workflow for managing your classroom."
    )

    steps = [
        ("STEP 01", "Teacher Dashboard",
         "Access the attendance management workspace."),
        ("STEP 02", "Student Enrollment",
         "Register student details and capture their face."),
        ("STEP 03", "Face Recognition",
         "Capture the face and verify the enrolled student."),
        ("STEP 04", "Attendance Reports",
         "Review records and generate monthly reports.")
    ]

    for label, title, description in steps:

        st.markdown(f"""
        <div class="feature-card" style="margin-bottom:15px">
            <div class="step-label">{label}</div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)


# TEACHER DASHBOARD

elif page == "Dashboard":

    page_title(
        "TEACHER WORKSPACE",
        "Interactive Dashboard",
        "Manage your students and classroom attendance."
    )

    files = list(ATTENDANCE_DIR.glob("attendance_*.xlsx"))

    total_rows = 0
    today_rows = 0

    for file in files:
        try:
            df = pd.read_excel(file)
            total_rows += len(df)

            if today in file.name:
                today_rows += len(df)

        except Exception:
            pass

    m1, m2, m3 = st.columns(3)

    m1.metric("Registered Students", len(students))
    m2.metric("Today's Attendance", today_rows)
    m3.metric("Total Attendance", total_rows)

    st.markdown("### Teacher Tools")

    a, b, c = st.columns(3)

    with a:
        if st.button("Enroll Student", use_container_width=True):
            st.session_state.snap_page = "Enroll Student"
            st.rerun()

    with b:
        if st.button("Mark Attendance", use_container_width=True):
            st.session_state.snap_page = "Mark Attendance"
            st.rerun()

    with c:
        if st.button("View Records", use_container_width=True):
            st.session_state.snap_page = "Attendance Records"
            st.rerun()

    st.markdown("### Available Subjects")

    st.write(" · ".join(SUBJECTS))
 
# -------------------- ENROLL STUDENT --------------------
elif page == "Enroll Student":
    page_title("TEACHER WORKSPACE", "Student enrollment", "Enter student details, then capture exactly one clear face.")

    with st.form("enroll_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Student name *", placeholder="e.g. Anash Khan", max_chars=60)
            enrollment = st.text_input("Enrollment ID *", placeholder="e.g. 2301234567", max_chars=30)
        with c2:
            student_class = st.text_input("Class / Section *", placeholder="e.g. B.Tech CSE - 3A")
            st.caption("Name accepts letters, spaces, apostrophes and hyphens only.")

        submitted = st.form_submit_button(
            "Continue to face capture",
            type="primary",
            use_container_width=True
        )

    if submitted:
        errors = []

        if not valid_person_name(name):
            errors.append("Name must contain 2–60 letters. Numbers are not allowed.")
        if not valid_enrollment(enrollment):
            errors.append("Enrollment ID: 3–30 characters; letters, numbers, / and - only.")
        if not valid_class(student_class):
            errors.append("Class / Section: 2–40 characters; letters, numbers and . & ( ) / - allowed.")
        if any(str(s.get("enrollment_id", "")).strip().lower() == enrollment.strip().lower() for s in students):
            errors.append("This Enrollment ID is already registered.")

        if errors:
            for error in errors:
                st.error(error)
        else:
            st.session_state["pending_student"] = {
                "name": name.strip(),
                "enrollment": enrollment.strip(),
                "student_class": student_class.strip()
            }
            st.success("Details accepted. Capture the student's face below.")

    pending = st.session_state.get("pending_student")

    if pending:
        st.info(f"Ready to enroll: **{pending['name']}** · {pending['enrollment']}")
        photo = st.camera_input("Capture student face", key="enroll_camera")

        if photo and st.button("Process face & enroll", type="primary", use_container_width=True):
            try:
                rgb = image_to_rgb(photo)
                encoding, count = extract_one_face_encoding(rgb)

                if count != 1 or encoding is None:
                    st.error(f"Expected exactly one face, but detected {count}. Retake the photo.")
                else:
                    save_student(
                        pending["name"],
                        pending["enrollment"],
                        pending["student_class"],
                        encoding
                    )
                    st.session_state.pop("pending_student", None)
                    st.success("Student enrolled successfully.")
                    st.rerun()

            except Exception as exc:
                st.error(f"Could not process image: {exc}")


# -------------------- MARK ATTENDANCE --------------------
elif page == "Mark Attendance":
    page_title("TEACHER WORKSPACE", "Face ID attendance", "Select a subject and capture the enrolled student's face.")

    subject = st.selectbox("Select subject", SUBJECTS)

    if not students:
        st.warning("No students enrolled yet. Enroll a student first.")

    photo = st.camera_input("Capture face to mark attendance", key="attendance_camera")

    if photo and st.button("Recognize & save attendance", type="primary", use_container_width=True):
        try:
            rgb = image_to_rgb(photo)
            encoding, count = extract_one_face_encoding(rgb)

            if count != 1 or encoding is None:
                st.error(f"Expected exactly one face, but detected {count}.")
            else:
                student, distance = match_student(encoding, students)

                if student is None:
                    st.error("Face not recognized. Enroll this student first.")
                else:
                    added, _ = mark_attendance(student, subject)

                    if added:
                        st.success(f"Attendance marked for {student['name']} in {subject}.")
                    else:
                        st.warning(f"Attendance already marked today for {student['name']} in {subject}.")

                    st.caption(f"Face match distance: {distance:.3f}")

        except Exception as exc:
            st.error(f"Could not process attendance: {exc}")


# -------------------- ATTENDANCE RECORDS --------------------
elif page == "Attendance Records":
    page_title("TEACHER WORKSPACE", "Attendance records", "Review and export saved attendance.")

    files = sorted(ATTENDANCE_DIR.glob("attendance_*.xlsx"), reverse=True)

    if not files:
        st.info("No attendance records available yet.")
    else:
        selected = st.selectbox("Choose attendance file", files, format_func=lambda p: p.name)

        try:
            df = pd.read_excel(selected)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Excel",
                selected.read_bytes(),
                file_name=selected.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as exc:
            st.error(f"Could not read attendance file: {exc}")


# -------------------- MONTHLY REPORT --------------------
elif page == "Monthly Report":
    page_title("TEACHER WORKSPACE", "Monthly report", "Generate a subject-wise attendance summary.")

    subject = st.selectbox("Subject", SUBJECTS, key="report_subject")
    month = st.text_input(
        "Month (YYYY-MM)",
        value=datetime.now().strftime("%Y-%m"),
        max_chars=7
    )

    if st.button("Generate report", type="primary"):
        try:
            datetime.strptime(month, "%Y-%m")

            frames = []
            session_dates = set()

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

                report = unique.groupby(
                    ["Enrollment", "Name", "Class"],
                    as_index=False
                )["Date"].nunique()

                report.rename(columns={"Date": "Total Attendance"}, inplace=True)

                report["Attendance Percentage"] = (
                    report["Total Attendance"] / len(session_dates) * 100
                ).round(2)

                st.dataframe(report, use_container_width=True, hide_index=True)

                output = io.BytesIO()

                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    report.to_excel(writer, index=False)

                st.download_button(
                    "Download monthly report",
                    output.getvalue(),
                    file_name=f"monthly_report_{subject}_{month}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except ValueError:
            st.error("Enter month in YYYY-MM format, e.g. 2026-11.")


# -------------------- STUDENTS --------------------
elif page == "Students":
    page_title("TEACHER WORKSPACE", "Student roster", "View all enrolled students.")

    if not students:
        st.info("No students registered.")
    else:
        table = pd.DataFrame([{
            "Enrollment": s.get("enrollment_id", ""),
            "Name": s.get("name", ""),
            "Class": s.get("class", ""),
            "Face samples": len(s.get("encodings", [])),
        } for s in students])

        st.dataframe(table, use_container_width=True, hide_index=True)
