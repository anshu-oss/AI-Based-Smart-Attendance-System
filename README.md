# AI-Based Smart Attendance System — Streamlit

A browser-based adaptation of the original Tkinter face-recognition attendance project.

## Features
- Student enrollment from a browser camera snapshot
- Face matching using `face_recognition`
- Subject-wise attendance (Data Visualization, Machine Learning, App Development)
- Attendance table and Excel downloads
- Monthly attendance summary

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder's files to a GitHub repository.
2. Create a new app at https://share.streamlit.io/
3. Select the repository, branch, and `app.py`.
4. Deploy.

## Important limitations
- The app uses browser camera snapshots, not a continuous camera stream.
- Local `data/` files on Streamlit Community Cloud are ephemeral and may disappear on restart/redeploy.
- For real deployment, move biometric encodings and attendance records to secured persistent storage (for example, a private database/object store), add proper admin authentication, consent and retention controls, and restrict access.
- Do not commit student face encodings or attendance workbooks to a public GitHub repository.
- Face recognition accuracy depends on lighting, camera quality, and image quality. This is a prototype, not a verified identity system.
