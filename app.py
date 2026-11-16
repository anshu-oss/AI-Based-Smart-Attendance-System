 photo = st.camera_input("Capture student face", key="enroll_camera")
 if photo and st.button("Process face & enroll", type="primary", use_container_width=True):
            try:
                rgb = image_to_rgb(photo)
                encoding, count = extract_one_face_encoding(rgb)
                if count != 1 or encoding is None:
                    st.error(f"Expected exactly one face, but detected {count}. Retake the photo.")
                else:
                    save_student(pending["name"], pending["enrollment"], pending["student_class"], encoding)
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

# -------------------- RECORDS --------------------
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
            st.download_button("Download Excel", selected.read_bytes(), file_name=selected.name,
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as exc:
            st.error(f"Could not read attendance file: {exc}")

# -------------------- MONTHLY REPORT --------------------
elif page == "Monthly Report":
    page_title("TEACHER WORKSPACE", "Monthly report", "Generate a subject-wise attendance summary.")
    subject = st.selectbox("Subject", SUBJECTS, key="report_subject")
    month = st.text_input("Month (YYYY-MM)", value=datetime.now().strftime("%Y-%m"), max_chars=7)
    if st.button("Generate report", type="primary"):
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
