# install ffmpeg
# install keybert
#install google-api-python-client


from keybert import KeyBERT
from googleapiclient.discovery import build
from nemo.collections.asr.models import EncDecCTCModel
import os

#convert accented stt
model = EncDecCTCModel.from_pretrained("QuartzNet15x5Base-En")
transcription = model.transcribe(["/home/shrishti/audio/Ml_project_test1.wav"])
print("Prediction:", transcription[0])

#extracting keywords
kw_model = KeyBERT()
text = transcription[0]
keywords = kw_model.extract_keywords(transcription[0].text, keyphrase_ngram_range=(1, 2), stop_words='english')
print(keywords)


#search relevant youtube videos
API_KEY = os.getenv("YT_API_KEY")  
SEARCH_QUERY = keywords[0][0]
print(SEARCH_QUERY)

youtube = build("youtube", "v3", developerKey=API_KEY)

request = youtube.search().list(
    part="snippet",
    q=SEARCH_QUERY,
    type="video",
    maxResults=5
)
response = request.execute()

print("🎥 Top YouTube Results:")
for item in response["items"]:
    video_id = item["id"]["videoId"]
    title = item["snippet"]["title"]
    print(f"{title} — https://www.youtube.com/watch?v={video_id}")
