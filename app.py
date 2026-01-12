# import os
# import time
# import base64
# import tempfile
# import requests
# import streamlit as st

# from faster_whisper import WhisperModel
# from openai import OpenAI
# from gtts import gTTS

# # --------------------------------------------------
# # PAGE CONFIG
# # --------------------------------------------------
# st.set_page_config(page_title="Voice Avatar Assistant", layout="wide")

# # --------------------------------------------------
# # SIDEBAR – API KEYS
# # --------------------------------------------------
# st.sidebar.title("API Configuration")

# openai_key_input = st.sidebar.text_input(
#     "OpenAI API Key",
#     type="password",
#     help="Enter your OpenAI API key"
# )

# did_key_input = st.sidebar.text_input(
#     "D-ID API Key",
#     type="password",
#     help="Enter your D-ID API key"
# )

# if "OPENAI_API_KEY" not in st.session_state:
#     st.session_state.OPENAI_API_KEY = None

# if "DID_API_KEY" not in st.session_state:
#     st.session_state.DID_API_KEY = None

# if openai_key_input:
#     st.session_state.OPENAI_API_KEY = openai_key_input

# if did_key_input:
#     st.session_state.DID_API_KEY = did_key_input

# # --------------------------------------------------
# # STOP APP IF KEYS MISSING
# # --------------------------------------------------
# if not st.session_state.OPENAI_API_KEY or not st.session_state.DID_API_KEY:
#     st.title("Voice Avatar Assistant")
#     st.warning("Please enter both OpenAI and D-ID API keys in the sidebar to continue.")
#     st.stop()

# # --------------------------------------------------
# # CLIENTS
# # --------------------------------------------------
# client = OpenAI(api_key=st.session_state.OPENAI_API_KEY)

# MODEL_SIZE = "small.en"
# AVATAR_IMAGE_URL = "https://storage.googleapis.com/magicpoint/outputs/biometric-photo-output.png"

# # --------------------------------------------------
# # SESSION STATE
# # --------------------------------------------------
# if "avatar_video" not in st.session_state:
#     st.session_state.avatar_video = None

# if "conversation" not in st.session_state:
#     st.session_state.conversation = []

# if "is_processing" not in st.session_state:
#     st.session_state.is_processing = False

# if "last_audio_size" not in st.session_state:
#     st.session_state.last_audio_size = None

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
# # TTS
# # --------------------------------------------------
# def generate_tts_audio(text: str) -> str:
#     tts = gTTS(text=text, lang="en")
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#     tts.save(tmp.name)
#     return tmp.name

# # --------------------------------------------------
# # D-ID AVATAR
# # --------------------------------------------------
# def generate_avatar_video_url(audio_path: str) -> str:
#     auth = base64.b64encode(f"{st.session_state.DID_API_KEY}:".encode()).decode()
#     headers = {"Authorization": f"Basic {auth}"}

#     print("⬆ Uploading audio to D-ID")
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

#     print("🎬 Creating talk")
#     response = requests.post(
#         "https://api.d-id.com/talks",
#         headers={**headers, "Content-Type": "application/json"},
#         json=payload,
#         timeout=30
#     )

#     if response.status_code != 201:
#         raise Exception(response.text)

#     talk_id = response.json()["id"]

#     for _ in range(60):
#         status = requests.get(
#             f"https://api.d-id.com/talks/{talk_id}",
#             headers=headers,
#             timeout=15
#         ).json()

#         if status["status"] == "done":
#             print("Avatar ready")
#             return status["result_url"]

#         if status["status"] == "error":
#             raise Exception(status)

#         time.sleep(1)

#     raise Exception("Avatar generation timeout")

# # --------------------------------------------------
# # UI
# # --------------------------------------------------
# st.title("Voice Avatar Assistant")
# st.caption("Speak → AI responds with a talking avatar")

# left_col, right_col = st.columns([1, 1], gap="large")

# # --------------------------------------------------
# # LEFT COLUMN
# # --------------------------------------------------
# with left_col:
#     st.subheader("Speak")

#     audio = st.audio_input("Click and record your question")

#     if audio is not None and not st.session_state.is_processing:

#         if audio.size == st.session_state.last_audio_size:
#             st.stop()

#         st.session_state.is_processing = True
#         st.session_state.last_audio_size = audio.size

#         print("AUDIO RECEIVED | Size:", audio.size)

#         audio_bytes = audio.getbuffer()
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#             tmp.write(audio_bytes)
#             audio_path = tmp.name

#         with st.spinner("Transcribing..."):
#             segments, _ = whisper_model.transcribe(audio_path)
#             segments = list(segments)
#             user_text = " ".join(s.text for s in segments).strip()

#         if user_text:
#             st.session_state.conversation.append(("You", user_text))

#             with st.spinner("Thinking..."):
#                 reply = query_llm(user_text)

#             st.session_state.conversation.append(("Assistant", reply))

#             with st.spinner("Generating avatar..."):
#                 try:
#                     tts_audio = generate_tts_audio(reply)
#                     video_url = generate_avatar_video_url(tts_audio)
#                     st.session_state.avatar_video = video_url
#                 except Exception as e:
#                     st.error("Avatar generation failed")
#                     st.audio(tts_audio)

#         st.session_state.is_processing = False

#     st.divider()
#     for role, text in st.session_state.conversation[-6:]:
#         st.markdown(f"**{role}:** {text}")

# # --------------------------------------------------
# # RIGHT COLUMN
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
from openai import OpenAIError

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Voice Avatar Assistant", layout="wide")

# --------------------------------------------------
# SAFE ERROR HANDLER
# --------------------------------------------------
def safe_error(user_message: str, err: Exception | None = None):
    st.error(user_message)
    if err:
        print("INTERNAL ERROR:", repr(err))

# --------------------------------------------------
# SIDEBAR – API KEYS
# --------------------------------------------------
st.sidebar.title("API Configuration")

openai_key_input = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

did_key_input = st.sidebar.text_input(
    "D-ID API Key",
    type="password"
)

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = None

if "DID_API_KEY" not in st.session_state:
    st.session_state.DID_API_KEY = None

if openai_key_input:
    st.session_state.OPENAI_API_KEY = openai_key_input

if did_key_input:
    st.session_state.DID_API_KEY = did_key_input

# --------------------------------------------------
# STOP IF KEYS MISSING
# --------------------------------------------------
if not st.session_state.OPENAI_API_KEY or not st.session_state.DID_API_KEY:
    st.title("Voice Avatar Assistant")
    st.warning("Please enter both OpenAI and D-ID API keys in the sidebar.")
    st.stop()

# --------------------------------------------------
# CLIENT
# --------------------------------------------------
client = OpenAI(api_key=st.session_state.OPENAI_API_KEY)

MODEL_SIZE = "small.en"
AVATAR_IMAGE_URL = "https://storage.googleapis.com/magicpoint/outputs/biometric-photo-output.png"

# --------------------------------------------------
# API KEY VALIDATION (NEW)
# --------------------------------------------------
def validate_openai_key():
    try:
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
        return True
    except OpenAIError as e:
        msg = str(e).lower()

        if "incorrect api key" in msg or "invalid" in msg:
            safe_error("Your OpenAI API key is invalid. Please check and try again.", e)
        elif "quota" in msg or "billing" in msg or "exceeded" in msg:
            safe_error("Your OpenAI credits are exhausted or expired.", e)
        else:
            safe_error("Unable to verify OpenAI API key right now.", e)

        return False

def validate_did_key():
    try:
        auth = base64.b64encode(f"{st.session_state.DID_API_KEY}:".encode()).decode()
        headers = {"Authorization": f"Basic {auth}"}

        r = requests.get(
            "https://api.d-id.com/credits",
            headers=headers,
            timeout=10
        )

        if r.status_code == 401:
            safe_error("Your D-ID API key is invalid.")
            return False

        if r.status_code == 403:
            safe_error("Your D-ID credits are exhausted or expired.")
            return False

        return True

    except Exception as e:
        safe_error("Unable to verify D-ID API key right now.", e)
        return False

# --------------------------------------------------
# VALIDATE KEYS ONCE
# --------------------------------------------------
if not validate_openai_key() or not validate_did_key():
    st.stop()

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
    auth = base64.b64encode(f"{st.session_state.DID_API_KEY}:".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}

    audio_upload = requests.post(
        "https://api.d-id.com/audios",
        headers=headers,
        files={"audio": open(audio_path, "rb")},
        timeout=30
    )

    if audio_upload.status_code == 401:
        raise Exception("D-ID invalid key")

    if audio_upload.status_code == 403:
        raise Exception("D-ID credits exhausted")

    if audio_upload.status_code != 201:
        raise Exception("Audio upload failed")

    audio_url = audio_upload.json()["url"]

    payload = {
        "source_url": AVATAR_IMAGE_URL,
        "script": {
            "type": "audio",
            "audio_url": audio_url
        }
    }

    response = requests.post(
        "https://api.d-id.com/talks",
        headers={**headers, "Content-Type": "application/json"},
        json=payload,
        timeout=30
    )

    if response.status_code != 201:
        raise Exception("Avatar creation failed")

    talk_id = response.json()["id"]

    for _ in range(60):
        status = requests.get(
            f"https://api.d-id.com/talks/{talk_id}",
            headers=headers,
            timeout=15
        ).json()

        if status.get("status") == "done":
            return status["result_url"]

        if status.get("status") == "error":
            raise Exception("Avatar rendering error")

        time.sleep(1)

    raise Exception("Avatar generation timeout")

# --------------------------------------------------
# UI
# --------------------------------------------------
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
        try:
            if audio.size == st.session_state.last_audio_size:
                st.stop()

            st.session_state.is_processing = True
            st.session_state.last_audio_size = audio.size

            audio_bytes = audio.getbuffer()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                audio_path = tmp.name

            try:
                with st.spinner("Transcribing..."):
                    segments, _ = whisper_model.transcribe(audio_path)
                    segments = list(segments)
                    user_text = " ".join(s.text for s in segments).strip()
            except Exception as e:
                safe_error("Sorry, I couldn't understand your voice.", e)
                st.stop()

            if user_text:
                st.session_state.conversation.append(("You", user_text))

                try:
                    with st.spinner("Thinking..."):
                        reply = query_llm(user_text)
                except Exception as e:
                    safe_error("Your OpenAI credits may be exhausted.", e)
                    st.stop()

                st.session_state.conversation.append(("Assistant", reply))

                try:
                    with st.spinner("Generating avatar..."):
                        tts_audio = generate_tts_audio(reply)
                        video_url = generate_avatar_video_url(tts_audio)
                        st.session_state.avatar_video = video_url
                except Exception as e:
                    safe_error("Avatar could not be generated right now.", e)
                    if "tts_audio" in locals():
                        st.audio(tts_audio)

        finally:
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
