# Libraries 
import os
import re
import json
import nltk
import whisper
from nltk.tokenize import sent_tokenize
from pyannote.audio import Pipeline
from deep_translator import GoogleTranslator
from collections import defaultdict

nltk.download("punkt")

# Load Models
print("Loading models...")
whisper_model = whisper.load_model("base")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=" "  #huggingfacetoken 
)

# input
target_lang = input("Enter output language (e.g., english, hindi): ").strip().lower()
audio_path =  input("Enter path to audio file (.mp3/.wav): ").strip()

if not os.path.exists(audio_path):
    raise FileNotFoundError("Audio file not found.")

# Speaker Diarization
print("Performing speaker diarization...")
diarization = pipeline(audio_path)

#  Transcription 
print("Transcribing with Whisper...")
transcription = whisper_model.transcribe(audio_path, word_timestamps=True)
segments = transcription["segments"]

# Speaker Attribution
speaker_segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    words = []
    for seg in segments:
        if "words" in seg:
            for word_info in seg["words"]:
                if turn.start <= word_info["start"] <= turn.end:
                    words.append(word_info["word"])
    if words:
        speaker_segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
            "text": " ".join(words)
        })

#Guess Speaker Roles
def guess_roles(speaker_segments):
    speaker_order = []
    for seg in speaker_segments:
        if seg["speaker"] not in speaker_order:
            speaker_order.append(seg["speaker"])
        if len(speaker_order) == 2:
            break
    if len(speaker_order) < 2:
        return {speaker_order[0]: "Support"}
    return {speaker_order[0]: "Support", speaker_order[1]: "Customer"}

role_map = guess_roles(speaker_segments)
name_map = {
    speaker: ("Mira (Support)" if role == "Support" else "Ravi (Customer)")
    for speaker, role in role_map.items()
}

# Group & Translate 
translated_segments = []
prev_speaker = None
buffer_text = ""

print(f"Translating to {target_lang} ...")
for seg in speaker_segments:
    speaker_name = name_map.get(seg["speaker"], seg["speaker"])
    if speaker_name == prev_speaker:
        buffer_text += " " + seg["text"]
    else:
        if prev_speaker:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(buffer_text.strip())
            cleaned_text = re.sub(r'\s+', ' ', translated).strip()
            translated_segments.append({"name": prev_speaker, "text": cleaned_text})
        prev_speaker = speaker_name
        buffer_text = seg["text"]

if prev_speaker and buffer_text:
    translated = GoogleTranslator(source='auto', target=target_lang).translate(buffer_text.strip())
    cleaned_text = re.sub(r'\s+', ' ', translated).strip()
    translated_segments.append({"name": prev_speaker, "text": cleaned_text})

# Display Conversation 
print("\n Translated Conversation ====>")
for seg in translated_segments:
    print(f"{seg['name']}: {seg['text']}")

# Extract Complaint Info
customer_text = " ".join(seg["text"] for seg in translated_segments if "Customer" in seg["name"])
data = {}

data["Customer Name"] = re.search(r"My name is ([A-Z][a-z]+ [A-Z][a-z]+)", customer_text)
data["Customer Name"] = data["Customer Name"].group(1) if data["Customer Name"] else "Not Found"

data["Product Name"] = re.search(r"purchased a (.+?)( last| and)", customer_text)
data["Product Name"] = data["Product Name"].group(1).strip() if data["Product Name"] else "Not Found"

data["Order ID"] = re.search(r"order ID.*?(SBM\d+)", customer_text)
data["Order ID"] = data["Order ID"].group(1) if data["Order ID"] else "Not Found"

data["Purchase Date"] = re.search(r"bought it on ([A-Za-z]+\s\d{1,2})", customer_text)
data["Purchase Date"] = data["Purchase Date"].group(1) if data["Purchase Date"] else "Not Found"

data["Platform of Purchase"] = re.search(r"ordered it (directly from your website|through a third[- ]party platform)", customer_text)
data["Platform of Purchase"] = data["Platform of Purchase"].group(1) if data["Platform of Purchase"] else "Not Found"

# Issue Description (Flexible)
issue_patterns = [
    r"I’m having (?:an )?issue with (.+?)\.?",  # e.g., issue with the fit
    r"problem with (.+?)\.?",                   # e.g., problem with stitching
    r"trouble with the (.+?)\.?",               # e.g., trouble with the size
    r"It keeps (.+?)\.?",                       # e.g., it keeps falling apart
]

issue_description = []

for pattern in issue_patterns:
    match = re.search(pattern, customer_text, re.IGNORECASE)
    if match:
        issue_description.append(match.group(1).strip())

data["Issue Description"] = "; ".join(issue_description) if issue_description else "Not Found"


# Steps Already Taken (Product-agnostic, Flexible)
step_keywords = {
    "tried it with": "Tried with different inputs/devices",
    "reset": "Performed a reset",
    "washed": "Washed the product",
    "reinstalled": "Reinstalled",
    "updated the firmware": "Updated the firmware",
    "tightened the screws": "Tightened the screws",
    "used a different": "Tested with an alternative",
    "recharged": "Recharged the battery",
    "used according to instructions": "Followed product manual",
    "checked manual": "Checked the manual",
    "contacted support earlier": "Previously contacted support",
    "flipped through": "Inspected manually"
}

steps = []
for keyword, description in step_keywords.items():
    if keyword in customer_text.lower():
        steps.append(description)

data["Steps Already Taken"] = steps if steps else ["Not Found"]


# Preferred Resolution (check all text, not just customer)
full_text = " ".join(seg["text"] for seg in translated_segments)

if "offer you a replacement" in full_text.lower() or "replacement" in full_text.lower():
    data["Preferred Resolution"] = "Replacement"
elif "refund" in full_text.lower():
    data["Preferred Resolution"] = "Refund"
elif "repair" in full_text.lower():
    data["Preferred Resolution"] = "Repair"
else:
    data["Preferred Resolution"] = "Not Found"


feedback_match = re.search(r"I really liked (.+?)\. It's (.+?)\. ", customer_text)
data["Additional Product Feedback"] = (
    f"{feedback_match.group(1)}. It's {feedback_match.group(2)}."
    if feedback_match else "Not Found"
)

email_match = re.search(r"email.*? ([\w\.-]+@[\w\.-]+)", customer_text)
data["Email Address"] = email_match.group(1) if email_match else "Not Provided"

# Output 
print("\n Extracted Complaint Info =====>")
print(json.dumps(data, indent=4))

with open("complaint_info.json", "w") as f:
    json.dump(data, f, indent=4)


