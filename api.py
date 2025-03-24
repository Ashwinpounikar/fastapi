from fastapi import FastAPI
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
import tempfile
import json
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from deep_translator import GoogleTranslator
from fastapi.responses import FileResponse

app = FastAPI()

# Initialize Sentiment Analyzer & Translator
analyzer = SentimentIntensityAnalyzer()
translator = Translator()

class NewsRequest(BaseModel):
    company: str
    num_articles: int = 10

# Function to get news article URLs using Google Search
def get_news_urls(company, num_articles=10):
    query = f"{company} news"
    search_url = f"https://www.google.com/search?q={query}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    urls = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/url?q=" in href:
            url = href.split("/url?q=")[1].split("&")[0]
            if url.startswith("http") and "google" not in url:
                urls.append(url)
        if len(urls) >= num_articles:
            break

    return urls

# Function to extract news data
def extract_news_data(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string if soup.title else "No Title Found"
        paragraphs = soup.find_all("p")
        summary_text = " ".join([p.get_text() for p in paragraphs[:10]])[:500]

        hindi_summary = translate_to_hindi(summary_text)
        sentiment_result = analyze_sentiment(summary_text)
        audio_file = text_to_speech(hindi_summary)

        return {
            "title": title,
            "summary": summary_text,
            "hindi_summary": hindi_summary,
            "sentiment": sentiment_result["sentiment"],
            "vader_score": sentiment_result["vader_score"],
            "textblob_score": sentiment_result["textblob_score"],
            "url": url,
            "audio_file": audio_file
        }
    except Exception as e:
        return {"error": f"Failed to extract data from {url}: {str(e)}"}

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment_score = analyzer.polarity_scores(text)["compound"]
    blob_score = TextBlob(text).sentiment.polarity

    sentiment = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"

    return {
        "sentiment": sentiment,
        "vader_score": sentiment_score,
        "textblob_score": blob_score
    }

# Function to translate text to Hindi
def translate_to_hindi(text):
    try:
        return GoogleTranslator(source="auto", target="hi").translate(text)
    except:
        return text

# Function to convert text to Hindi speech
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang="hi")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        return temp_audio.name
    except:
        return None

@app.post("/fetch-news")
async def fetch_news(request: NewsRequest):
    urls = get_news_urls(request.company, request.num_articles)
    news_data = [extract_news_data(url) for url in urls]
    
    # Save results to JSON
    json_filename = f"{request.company}_news.json"
    with open(json_filename, "w", encoding="utf-8") as json_file:
        json.dump(news_data, json_file, indent=4, ensure_ascii=False)

    return {"news_data": news_data, "json_file": json_filename}

@app.get("/download-json")
async def download_json(filename: str):
    return FileResponse(filename, media_type="application/json", filename=filename)

