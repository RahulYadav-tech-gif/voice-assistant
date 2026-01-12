# import os
# import time
# import base64
# import tempfile
# import requests
# import streamlit as st

# from faster_whisper import WhisperModel
# from openai import OpenAI
# from gtts import gTTS
# from dotenv import load_dotenv

# # --------------------------------------------------
# # LOAD ENV
# # --------------------------------------------------
# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# DID_API_KEY = os.getenv("DID_API_KEY")

# client = OpenAI(api_key=OPENAI_API_KEY)

# MODEL_SIZE = "small.en"
# AVATAR_IMAGE_URL = "https://storage.googleapis.com/magicpoint/outputs/biometric-photo-output.png"

# # --------------------------------------------------
# # SESSION STATE
# # --------------------------------------------------
# if "avatar_video" not in st.session_state:
#     st.session_state.avatar_video = None

# if "conversation" not in st.session_state:
#     st.session_state.conversation = []

# # --------------------------------------------------
# # LOAD WHISPER
# # --------------------------------------------------
# @st.cache_resource
# def load_whisper():
#     return WhisperModel(
#         MODEL_SIZE,
#         device="cpu",
#         compute_type="int8"
#     )

# whisper_model = load_whisper()

# # --------------------------------------------------
# # LLM
# # --------------------------------------------------
# def query_llm(text: str) -> str:
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "Answer briefly in 1–2 sentences."},
#             {"role": "user", "content": text}
#         ]
#     )
#     return response.choices[0].message.content.strip()

# # --------------------------------------------------
# # TTS (WAV for D-ID)
# # --------------------------------------------------
# def generate_tts_audio(text: str) -> str:
#     tts = gTTS(text=text, lang="en")
#     mp3_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#     tts.save(mp3_file.name)

#     return mp3_file.name

# # --------------------------------------------------
# # D-ID AVATAR (FIXED)
# # --------------------------------------------------
# def generate_avatar_video_url(audio_path: str) -> str:
#     auth = base64.b64encode(f"{DID_API_KEY}:".encode()).decode()
#     headers = {"Authorization": f"Basic {auth}"}

#     # Upload audio
#     audio_upload = requests.post(
#         "https://api.d-id.com/audios",
#         headers=headers,
#         files={"audio": open(audio_path, "rb")},
#         timeout=30
#     )

#     if audio_upload.status_code != 201:
#         raise Exception(audio_upload.text)

#     audio_url = audio_upload.json()["url"]

#     payload = {
#         "source_url": AVATAR_IMAGE_URL,
#         "script": {
#             "type": "audio",
#             "audio_url": audio_url
#         }
#     }

#     response = requests.post(
#         "https://api.d-id.com/talks",
#         headers={**headers, "Content-Type": "application/json"},
#         json=payload,
#         timeout=30
#     )

#     if response.status_code != 201:
#         raise Exception(response.text)

#     talk_id = response.json()["id"]

#     # Poll with timeout
#     for _ in range(60):  # max 60 seconds
#         status = requests.get(
#             f"https://api.d-id.com/talks/{talk_id}",
#             headers=headers,
#             timeout=15
#         ).json()

#         if status["status"] == "done":
#             return status["result_url"]

#         if status["status"] == "error":
#             raise Exception(status)

#         time.sleep(1)

#     raise Exception("Avatar generation timeout")

# # --------------------------------------------------
# # UI
# # --------------------------------------------------
# st.set_page_config(page_title="Voice Avatar Assistant", layout="wide")

# st.title("🎤 Voice Avatar Assistant")
# st.caption("Speak → AI responds with a talking avatar")

# left_col, right_col = st.columns([1, 1], gap="large")

# # --------------------------------------------------
# # LEFT
# # --------------------------------------------------
# with left_col:
#     st.subheader("Speak")

#     audio = st.audio_input("Click and record your question")

#     if audio is not None:
#         print("✅ AUDIO OBJECT RECEIVED")
#         print("Audio type:", type(audio))
#         print("Audio size (bytes):", audio.size)

#         try:
#             audio_bytes = audio.getbuffer()
#             print("✅ Audio buffer length:", len(audio_bytes))
#         except Exception as e:
#             print("❌ Failed to read audio buffer:", e)
#             st.error("Audio buffer read failed")
#             st.stop()

#         # Save audio
#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#                 tmp.write(audio_bytes)
#                 audio_path = tmp.name
#             print("✅ Audio saved at:", audio_path)
#         except Exception as e:
#             print("❌ Audio save failed:", e)
#             st.error("Audio save failed")
#             st.stop()

#         # Transcription
#         with st.spinner("Transcribing..."):
#             try:
#                 print("🔁 Starting Whisper transcription")
#                 segments, _ = whisper_model.transcribe(audio_path)
#                 segments = list(segments)
#                 user_text = " ".join(s.text for s in segments).strip()
#                 print("📝 Transcribed text:", user_text)
#             except Exception as e:
#                 print("❌ Whisper failed:", e)
#                 st.error("Transcription failed")
#                 st.stop()

#         if not user_text:
#             print("⚠️ Empty transcription")
#             st.warning("No speech detected.")
#             st.stop()

#         # LLM
#         with st.spinner("Thinking..."):
#             try:
#                 print("🤖 Sending to LLM")
#                 reply = query_llm(user_text)
#                 print("🤖 LLM reply:", reply)
#             except Exception as e:
#                 print("❌ LLM failed:", e)
#                 st.error("LLM error")
#                 st.stop()

#         # Avatar
#         with st.spinner("Generating avatar..."):
#             try:
#                 print("🎤 Generating TTS")
#                 tts_audio = generate_tts_audio(reply)
#                 print("🎥 Sending to D-ID")
#                 video_url = generate_avatar_video_url(tts_audio)
#                 print("✅ Avatar video URL:", video_url)
#                 st.session_state.avatar_video = video_url
#             except Exception as e:
#                 print("❌ Avatar generation failed:", e)
#                 st.error("Avatar failed")
#                 st.audio(tts_audio)

#         st.rerun()

#     st.divider()
#     for role, text in st.session_state.conversation[-6:]:
#         st.markdown(f"**{role}:** {text}")

# # --------------------------------------------------
# # RIGHT
# # --------------------------------------------------
# with right_col:
#     st.subheader("Assistant")
#     if st.session_state.avatar_video:
#         st.video(st.session_state.avatar_video, autoplay=True)
#     else:
#         st.image(AVATAR_IMAGE_URL, width='stretch')

# # --------------------------------------------------
# # CSS
# # --------------------------------------------------
# st.markdown(
#     """
#     <style>
#     video, img {
#         max-height: 70vh;
#         object-fit: contain;
#         border-radius: 16px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )






import os
import time
import base64
import tempfile
import requests
import streamlit as st

from faster_whisper import WhisperModel
from openai import OpenAI
from gtts import gTTS
from dotenv import load_dotenv

# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DID_API_KEY = os.getenv("DID_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_SIZE = "small.en"
AVATAR_IMAGE_URL = "https://storage.googleapis.com/magicpoint/outputs/biometric-photo-output.png"

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "avatar_video" not in st.session_state:
    st.session_state.avatar_video = None

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "last_audio_size" not in st.session_state:
    st.session_state.last_audio_size = None

# --------------------------------------------------
# LOAD WHISPER
# --------------------------------------------------
@st.cache_resource
def load_whisper():
    return WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )

whisper_model = load_whisper()

# --------------------------------------------------
# LLM
# --------------------------------------------------
def query_llm(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer briefly in 1–2 sentences."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content.strip()

# --------------------------------------------------
# TTS
# --------------------------------------------------
def generate_tts_audio(text: str) -> str:
    tts = gTTS(text=text, lang="en")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# --------------------------------------------------
# D-ID AVATAR
# --------------------------------------------------
def generate_avatar_video_url(audio_path: str) -> str:
    auth = base64.b64encode(f"{DID_API_KEY}:".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}

    print("⬆ Uploading audio to D-ID")
    audio_upload = requests.post(
        "https://api.d-id.com/audios",
        headers=headers,
        files={"audio": open(audio_path, "rb")},
        timeout=30
    )

    if audio_upload.status_code != 201:
        raise Exception(audio_upload.text)

    audio_url = audio_upload.json()["url"]

    payload = {
        "source_url": AVATAR_IMAGE_URL,
        "script": {
            "type": "audio",
            "audio_url": audio_url
        }
    }

    print("🎬 Creating talk")
    response = requests.post(
        "https://api.d-id.com/talks",
        headers={**headers, "Content-Type": "application/json"},
        json=payload,
        timeout=30
    )

    if response.status_code != 201:
        raise Exception(response.text)

    talk_id = response.json()["id"]

    for _ in range(60):
        status = requests.get(
            f"https://api.d-id.com/talks/{talk_id}",
            headers=headers,
            timeout=15
        ).json()

        if status["status"] == "done":
            print("✅ Avatar ready")
            return status["result_url"]

        if status["status"] == "error":
            raise Exception(status)

        time.sleep(1)

    raise Exception("Avatar generation timeout")

# --------------------------------------------------
# UI
# --------------------------------------------------
st.set_page_config(page_title="Voice Avatar Assistant", layout="wide")

st.title("Voice Avatar Assistant")
st.caption("Speak → AI responds with a talking avatar")

left_col, right_col = st.columns([1, 1], gap="large")

# --------------------------------------------------
# LEFT COLUMN
# --------------------------------------------------
with left_col:
    st.subheader("Speak")

    audio = st.audio_input("Click and record your question")

    if audio is not None and not st.session_state.is_processing:

        if audio.size == st.session_state.last_audio_size:
            st.stop()

        st.session_state.is_processing = True
        st.session_state.last_audio_size = audio.size

        print("✅ AUDIO RECEIVED | Size:", audio.size)

        audio_bytes = audio.getbuffer()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name

        print("💾 Audio saved:", audio_path)

        with st.spinner("Transcribing..."):
            segments, _ = whisper_model.transcribe(audio_path)
            segments = list(segments)
            user_text = " ".join(s.text for s in segments).strip()

        print("📝 Transcribed:", user_text)

        if user_text:
            st.session_state.conversation.append(("You", user_text))

            with st.spinner("Thinking..."):
                reply = query_llm(user_text)

            print("🤖 LLM Reply:", reply)
            st.session_state.conversation.append(("Assistant", reply))

            with st.spinner("Generating avatar..."):
                try:
                    tts_audio = generate_tts_audio(reply)
                    video_url = generate_avatar_video_url(tts_audio)
                    st.session_state.avatar_video = video_url
                except Exception as e:
                    print("❌ Avatar error:", e)
                    st.error("Avatar generation failed")
                    st.audio(tts_audio)

        st.session_state.is_processing = False

    st.divider()

    for role, text in st.session_state.conversation[-6:]:
        st.markdown(f"**{role}:** {text}")

# --------------------------------------------------
# RIGHT COLUMN
# --------------------------------------------------
with right_col:
    st.subheader("Assistant")

    if st.session_state.avatar_video:
        st.video(st.session_state.avatar_video, autoplay=True)
    else:
        st.image(AVATAR_IMAGE_URL, width='stretch')

# --------------------------------------------------
# CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    video, img {
        max-height: 70vh;
        object-fit: contain;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
