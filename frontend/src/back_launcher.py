# back_launcher.py
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS
from frontend.src.lib import file_manager_milo
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import threading
import time
import shutil
import requests

from lib import transcriber, subsynthetizer, webm_to_wav_converter, tts
from lib import message_queue

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
last_chunk_event = threading.Event()

BASE_DIR = Path(__file__).resolve().parent.parent
FRONT_DIR = BASE_DIR / "front"

# Adresse du RAG FastAPI — modifier si vous exécutez le RAG sur un autre hôte/port
RAG_BASE_URL = "http://127.0.0.1:8000"

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_front(path):
    if path != "" and (FRONT_DIR / path).exists():
        return send_from_directory(FRONT_DIR, path)
    else:
        # Default: serve index_voice or index_text depending on path presence
        # Keep same behavior as before (index.html previously)
        # We try to serve index_voice first (Milo voice UI), fallback to index_text
        if (FRONT_DIR / "index_voice.html").exists():
            return send_from_directory(FRONT_DIR, "index_voice.html")
        return send_from_directory(FRONT_DIR, "index_text.html")


@app.route("/upload-audio", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(file_manager_milo.webm_dir, filename)
    file.save(filepath)

    last_chunk = request.form.get("last_chunk", "false").lower() == "true"

    if last_chunk:
        last_chunk_event.set()

    message_queue.message_queue_handler.publish(
        "Audio_topic", {"filename": filename, "last_chunk": str(last_chunk)}
    )

    return jsonify({"status": "ok", "saved_as": filepath, "last_chunk": last_chunk})


@app.route("/get-audio/<filename>")
def get_audio(filename):
    filename = secure_filename(filename)
    return send_from_directory(file_manager_milo.milo_webm_response_dir, filename)


@app.route("/get-response-audio/<filename>")
def get_response_audio(filename):
    filename = secure_filename(filename)
    # Return wav responses for question/response playback
    return send_from_directory(file_manager_milo.milo_wav_question_response_dir, filename)


@app.route("/start-recording", methods=["POST"])
def start_recording():
    try:
        file_manager_milo.clearDirectory(file_manager_milo.webm_dir)
        file_manager_milo.clearDirectory(file_manager_milo.wav_dir)
        file_manager_milo.clearDirectory(file_manager_milo.milo_wav_response_dir)
        file_manager_milo.clearDirectory(file_manager_milo.milo_webm_response_dir)
        file_manager_milo.clearDirectory(file_manager_milo.sub_resume_dir)
        file_manager_milo.clearDirectory(file_manager_milo.transcript_dir)
        file_manager_milo.create_final_transcript()
        message_queue.clearAllStreams()
        last_chunk_event.clear()
        return {"status": "ok"}, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500


@app.route("/upload-question", methods=["POST"])
def upload_question():
    file_manager_milo.clearDirectory(file_manager_milo.milo_wav_question_dir)
    file_manager_milo.clearDirectory(file_manager_milo.milo_webm_question_dir)
    file_manager_milo.clearDirectory(file_manager_milo.milo_wav_question_response_dir)
    file_manager_milo.clearDirectory(file_manager_milo.milo_webm_question_response_dir)
    file_manager_milo.clearDirectory(file_manager_milo.question_transcript_dir)
    file_manager_milo.clearDirectory(file_manager_milo.milo_response_dir)
    message_queue.clearAllStreams()
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(file_manager_milo.milo_webm_question_dir, filename)
    file.save(filepath)

    message_queue.message_queue_handler.publish(
        "Question_topic", {"filename": filename}
    )
    return {"status": "ok"}, 200


def call_rag_for_text(question_text, matiere=None, enseignant=None, semestre=None, promo=None, threshold=0.35):
    """
    Post a question text to the RAG API and return the parsed JSON response.
    Returns None on error.
    """
    url = f"{RAG_BASE_URL}/api/ask"
    payload = {
        "question": question_text,
        "matiere": matiere,
        "enseignant": enseignant,
        "semestre": semestre,
        "promo": promo,
        "threshold": threshold
    }
    try:
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"[WARN] RAG returned status {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        print(f"[ERROR] calling RAG: {e}")
        return None


def handle_new_audio_file(msg, ObjTranscriber):
    filename = msg["filename"]
    last_chunk = msg.get("last_chunk", "False") == "True"
    wav_file = webm_to_wav_converter.convert_to_wav(
        file_manager_milo.webm_dir,
        file_manager_milo.wav_dir,
        filename
    )
    transcript_file = ObjTranscriber.transcribe_file(Path(wav_file))
    file_manager_milo.append_and_delete_transcript(transcript_file)

    if last_chunk:
        print("Tous les chunks reçus, génération finale...")
        message_queue.message_queue_handler.publish("Transcriber_topic", {"filepath": f"{file_manager_milo.transcript_dir}/transcript_final.txt"})


def handle_new_transcript(msg, ObjLlama=None):
    """
    When a transcript is ready we:
      1) Read transcript file
      2) Call the RAG API with the transcript as the 'question' (or process)
      3) Save the text response to disk, TTS it, convert and emit socket event
      4) Fallback to ObjLlama.generate_from_file if RAG not available
    """
    try:
        file_path = msg.get("filepath") or msg.get("filepath")
        if not file_path:
            print("[WARN] handle_new_transcript: no filepath in message")
            return

        transcript_path = Path(file_path)
        if not transcript_path.exists():
            print(f"[WARN] transcript file not found: {transcript_path}")
            return

        print(f"[INFO] Processing transcript: {transcript_path}")
        with open(transcript_path, "r", encoding="utf-8") as fh:
            transcript_text = fh.read().strip()

        if not transcript_text:
            print("[WARN] transcript empty, skipping")
            return

        # Call RAG: use the transcript as the "question/context" to generate a short spoken resume/answer.
        response_json = call_rag_for_text(transcript_text)

        if response_json and "answer" in response_json:
            answer_text = response_json["answer"]
            # Save answer text into milo_response_dir
            safe_name = transcript_path.stem + "_response.txt"
            answer_path = file_manager_milo.milo_response_dir / safe_name
            file_manager_milo.milo_response_dir.mkdir(parents=True, exist_ok=True)
            with open(answer_path, "w", encoding="utf-8") as out:
                out.write(answer_text)

            # Generate TTS (wav) from the answer text
            milo_wav = tts.myTTS.text_to_speech(answer_path, file_manager_milo.milo_wav_response_dir)

            # Convert wav -> webm for front playback
            milo_webm = webm_to_wav_converter.convert_to_webm(
                milo_wav,
                file_manager_milo.milo_webm_response_dir
            )

            # Emit socket to frontend to let it know new audio ready
            socketio.emit("new_audio", {"filename": os.path.basename(milo_webm)})
            print(f"[INFO] RAG response TTS ready: {milo_webm}")
        else:
            # Fallback: call old ObjLlama behavior if RAG unavailable
            if ObjLlama:
                print("[INFO] RAG unavailable or empty response — falling back to local ObjLlama synthesizer")
                ObjLlama.generate_from_file(Path(file_path))
                # Assume ObjLlama will produce synthesized files and emit socket messages as before
            else:
                print("[ERROR] No ObjLlama provided and RAG call failed.")

        # Backup transcripts and sub-resumes as before
        try:
            backup_dir = file_manager_milo.backup_transcript
            backup_dir.mkdir(exist_ok=True, parents=True)
            for src_dir in [file_manager_milo.transcript_dir, file_manager_milo.sub_resume_dir]:
                for item in src_dir.iterdir():
                    dest = backup_dir / item.name
                    shutil.copy2(item, dest)
        except Exception as e:
            print(f"[WARN] backup step failed: {e}")

    except Exception as e:
        print(f"[ERROR] handle_new_transcript failed: {e}")


def handle_new_question(msg, ObjTranscriber):
    """
    This handles "short question" flow:
    - Convert uploaded question webm -> wav
    - Transcribe
    - Call RAG for answer
    - Generate TTS and emit new_response_audio
    """
    try:
        filename = msg["filename"]
        wav_file = webm_to_wav_converter.convert_to_wav(
            file_manager_milo.milo_webm_question_dir,
            file_manager_milo.milo_wav_question_dir,
            filename
        )
        transcript_filename = ObjTranscriber.transcribe_file(Path(wav_file), file_manager_milo.question_transcript_dir)
        transcript_path = file_manager_milo.question_transcript_dir / transcript_filename

        # Read transcript text
        with open(transcript_path, "r", encoding="utf-8") as fh:
            question_text = fh.read().strip()

        # Call RAG
        response_json = call_rag_for_text(question_text)
        if response_json and "answer" in response_json:
            answer_text = response_json["answer"]
            # Save
            safe_name = transcript_path.stem + "_response.txt"
            answer_path = file_manager_milo.milo_response_dir / safe_name
            file_manager_milo.milo_response_dir.mkdir(parents=True, exist_ok=True)
            with open(answer_path, "w", encoding="utf-8") as out:
                out.write(answer_text)

            # TTS & conversion
            milo_wav = tts.myTTS.text_to_speech(answer_path, file_manager_milo.milo_wav_question_response_dir)
            # convert to webm if you prefer webm for playback
            # milo_webm = webm_to_wav_converter.convert_to_webm(milo_wav, file_manager.milo_webm_question_response_dir)
            socketio.emit("new_response_audio", {"filename": os.path.basename(milo_wav)})
            print("[INFO] Question processed and response emitted via socket")
        else:
            # fallback to old generation if available
            if ObjTranscriber:
                subsynthetizer.mySynthetizer.generate_from_file(transcript_path, isQuestion=True, output_dir=file_manager_milo.milo_response_dir)
                # After that, convert/speak as previous flow if necessary
            else:
                print("[WARN] RAG unavailable and no fallback available for question")

    except Exception as e:
        print(f"[ERROR] handle_new_question failed: {e}")


def handle_new_response(msg, ObjLlama):
    """
    The original implementation expected an LLM file generation stage and then TTS.
    We keep it, but primary RAG-driven flows should happen in handle_new_question / handle_new_transcript above.
    """
    file_path = msg["filepath"]
    output_name = None
    try:
        output_name = ObjLlama.generate_from_file(Path(file_path), True, file_manager_milo.milo_response_dir)
    except Exception as e:
        print(f"[WARN] ObjLlama.generate_from_file failed: {e}")
    if output_name:
        try:
            milo_response_wav = tts.myTTS.text_to_speech(file_manager_milo.milo_response_dir / output_name, file_manager_milo.milo_wav_question_response_dir)
            socketio.emit("new_response_audio", {"filename": os.path.basename(milo_response_wav)})
        except Exception as e:
            print(f"[WARN] TTS on generated response failed: {e}")


def setup_listeners():
    message_queue.message_queue_handler.subscribe("Audio_topic", "Audio_listener", callback=lambda msg: handle_new_audio_file(msg, transcriber.myTranscrib))
    message_queue.message_queue_handler.subscribe("Transcriber_topic", "Transcriber_listener", callback=lambda msg: handle_new_transcript(msg, subsynthetizer.mySynthetizer))
    message_queue.message_queue_handler.subscribe("Question_topic", "Question_listener", callback=lambda msg: handle_new_question(msg, transcriber.myTranscrib))
    message_queue.message_queue_handler.subscribe("Response_topic", "Response_listener", callback=lambda msg: handle_new_response(msg, subsynthetizer.mySynthetizer))

if __name__ == "__main__":
    file_manager_milo.clearAllDirectories()
    message_queue.clearAllStreams()
    file_manager_milo.create_final_transcript()
    setup_listeners()
    transcriber.myTranscrib.load_model()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)
