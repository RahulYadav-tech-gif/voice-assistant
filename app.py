import os
import time
import requests
import streamlit as st
from faster_whisper import WhisperModel
from openai import OpenAI
from gtts import gTTS
from dotenv import load_dotenv
import tempfile
from audio_recorder import AudioRecorder

# -------------------------------
# LOAD ENV
# -------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DID_API_KEY = os.getenv("DID_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# CONFIG
# -------------------------------
MODEL_SIZE = "small.en"
AVATAR_IMAGE_URL = "https://storage.googleapis.com/magicpoint/outputs/biometric-photo-output.png"

# -------------------------------
# SESSION STATE INIT
# -------------------------------
if "avatar_video" not in st.session_state:
    st.session_state.avatar_video = None  # None = show image

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "recorder" not in st.session_state:
    st.session_state.recorder = AudioRecorder(
        silence_threshold=500,
        silence_duration=2.0
    )

if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

if "recorded_audio_path" not in st.session_state:
    st.session_state.recorded_audio_path = None

if "last_check_time" not in st.session_state:
    st.session_state.last_check_time = 0

# -------------------------------
# INIT MODELS (CACHED)
# -------------------------------
@st.cache_resource
def load_whisper():
    return WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )

whisper_model = load_whisper()

# -------------------------------
# LLM
# -------------------------------
def query_llm(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer briefly in 1–2 sentences."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content.strip()

# -------------------------------
# TTS
# -------------------------------
def generate_tts_audio(text):
    tts = gTTS(text=text, lang="en")
    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_audio.name)
    return tmp_audio.name

# -------------------------------
# D-ID AVATAR
# -------------------------------
def generate_avatar_video_url(audio_path):
    audio_upload = requests.post(
        "https://api.d-id.com/audios",
        headers={"Authorization": f"Basic {DID_API_KEY}"},
        files={"audio": open(audio_path, "rb")}
    )

    if audio_upload.status_code != 201:
        raise Exception(audio_upload.text)

    audio_url = audio_upload.json()["url"]

    payload = {
        "source_url": AVATAR_IMAGE_URL,
        "script": {
            "type": "audio",
            "audio_url": audio_url
        },
        "config": {"stitch": True}
    }

    response = requests.post(
        "https://api.d-id.com/talks",
        headers={
            "Authorization": f"Basic {DID_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload
    )

    if response.status_code != 201:
        raise Exception(response.text)

    talk_id = response.json()["id"]

    while True:
        status = requests.get(
            f"https://api.d-id.com/talks/{talk_id}",
            headers={"Authorization": f"Basic {DID_API_KEY}"}
        ).json()

        if status["status"] == "done":
            return status["result_url"]

        if status["status"] == "error":
            raise Exception(status)

        time.sleep(1)

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Voice Avatar Assistant", layout="wide")

st.title("Voice Avatar Assistant")
st.caption("Speak → AI responds with a talking avatar")

left_col, right_col = st.columns([1, 1], gap="large")

# -------------------------------
# LEFT SIDE: RECORD + CHAT
# -------------------------------
with left_col:
    # Spacer to push record section downward
    st.markdown("<div style='height: 3.5rem;'></div>", unsafe_allow_html=True)

    # Recording button
    if not st.session_state.is_recording:
        if st.button("🎤 Start Recording", type="secondary", use_container_width=True):
            try:
                # Start recording
                st.session_state.is_recording = True
                st.session_state.recorder.start_recording()
                
                # Show recording status
                with st.spinner("Recording... Speak your question."):
                    # Wait for recording to complete (silence detected)
                    while st.session_state.recorder.is_currently_recording():
                        time.sleep(0.1)
                    
                    # Recording stopped, get the audio file
                    audio_path = st.session_state.recorder.stop_recording()
                
                st.session_state.is_recording = False
                
                if audio_path:
                    with st.spinner("Transcribing..."):
                        segments, _ = whisper_model.transcribe(audio_path)
                        user_text = " ".join(s.text for s in segments).strip()

                    if user_text:
                        st.session_state.conversation.append(("You", user_text))

                        with st.spinner("Thinking..."):
                            llm_reply = query_llm(user_text)

                        st.session_state.conversation.append(("Assistant", llm_reply))

                        with st.spinner("Generating avatar..."):
                            tts_audio = generate_tts_audio(llm_reply)
                            try:
                                video_url = generate_avatar_video_url(tts_audio)
                                st.session_state.avatar_video = video_url
                            except Exception as e:
                                st.warning(f"Avatar generation failed: {str(e)}")
                                st.audio(tts_audio)
                        
                        st.rerun()
                    else:
                        st.warning("No speech detected. Please try again.")
                else:
                    st.error("Recording failed. No audio file was created.")
                    
            except Exception as e:
                st.session_state.is_recording = False
                st.error(f"Recording error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    st.divider()


# -------------------------------
# RIGHT SIDE: AVATAR (FIXED, FULL VIEW)
# -------------------------------
with right_col:
    st.markdown("### Assistant")

    if st.session_state.avatar_video:
        st.video(
            st.session_state.avatar_video,
            autoplay=True,
            loop=False
        )
    else:
        st.image(
            AVATAR_IMAGE_URL,
            width="stretch"
        )

# -------------------------------
# CSS
# -------------------------------
st.markdown(
    """
    <style>
    /* Overall page padding */
    .block-container {
        padding-top: 2.8rem;
    }

    /* Bring Assistant section upward */
    [data-testid="column"]:nth-child(2) {
        padding-top: 0rem;
    }

    /* Reduce Assistant title spacing */
    h3 {
        margin-top: 0.2rem;
        margin-bottom: 0.6rem;
    }

    /* Avatar sizing – no scroll */
    video, img {
        max-height: 68vh;
        width: 100%;
        object-fit: contain;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



