import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import TranscriptsDisabled
from urllib.parse import urlparse, parse_qs
from langdetect import detect
from googletrans import Translator

# Load environment variables
load_dotenv()

# Configure the Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Base summarization prompt
base_prompt = """
You are a YouTube video summarizer. You will create a detailed, point-wise summary of the video content based on the transcript provided. Ensure each point is referenced with a timestamp and explained in simple, clear language. Include every major part of the video to provide a complete overview.
"""

# Function to extract video ID from YouTube URL
def extract_video_id(youtube_url):
    try:
        # Parse the URL
        parsed_url = urlparse(youtube_url)

        # Extract video ID from query parameters
        if parsed_url.query:
            query_params = parse_qs(parsed_url.query)
            video_id = query_params.get("v")
            if video_id:
                return video_id[0]

        # Handle URLs without query parameters (e.g., shortened URLs)
        if "/youtu.be/" in youtube_url:
            video_id = parsed_url.path.strip("/")
            return video_id

        raise ValueError("Invalid YouTube URL format.")
    except Exception as e:
        raise ValueError(f"Error extracting video ID: {e}")

# Function to extract transcript from a YouTube video and translate it if necessary
def extract_transcript_details(youtube_video_url):
    try:
        video_id = extract_video_id(youtube_video_url)

        # Get available transcripts
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try to get English transcript
        try:
            transcript_data = transcripts.find_transcript(['en']).fetch()
        except Exception:
            st.warning("English transcript not found. Checking available languages...")
            # Fetch available language codes
            available_languages = [
                transcript.language_code
                for transcript in transcripts if transcript.is_generated
            ]

            if available_languages:
                # Try fetching a generated transcript in one of the available languages
                language_to_try = available_languages[0]  # Pick the first available language
                st.info(f"Trying auto-generated transcript in '{language_to_try}'...")
                transcript_data = transcripts.find_generated_transcript([language_to_try]).fetch()
            else:
                st.info("This video has only auto-generated captions, and no valid transcripts available. Please try another video.")
                return None, None

        transcript = ""
        detailed_transcript = []

        for item in transcript_data:
            transcript += " " + item["text"]
            detailed_transcript.append({
                "text": item["text"],
                "start": item["start"],
                "duration": item["duration"]
            })

        # If the transcript is not in English, translate it
        detected_language = detect(transcript)
        if detected_language != "en":
            st.warning(f"Transcript is in {detected_language}. Translating to English...")
            translator = Translator()
            try:
                # Attempt translation
                transcript = translator.translate(transcript, src=detected_language, dest="en").text
                for item in detailed_transcript:
                    item["text"] = translator.translate(item["text"], src=detected_language, dest="en").text
            except AttributeError:
                # Handle coroutine-related errors
                st.info("This video has only auto-generated captions, and translation failed. Please try another video.")
                return None, None
            except Exception:
                # Handle other unexpected translation errors
                st.info("Translation failed due to an unexpected issue. Please try another video.")
                return None, None

        return transcript, detailed_transcript
    except Exception as e:
        st.info("This video has only auto-generated captions. Please try another video.")
        return None, None

# Function to generate point-wise summary with timestamps and a short summary
def generate_gemini_summary_with_timestamps(detailed_transcript, prompt, summary_length):
    if summary_length == "Short":
        length_prompt = " Summarize within 100 words."
    elif summary_length == "Medium":
        length_prompt = " Summarize within 250 words."
    elif summary_length == "Long":
        length_prompt = " Summarize within 500 words."
    else:
        length_prompt = f" Summarize within {summary_length} words."

    transcript_for_summary = "\n".join([
        f"[{item['start']:.2f}s]: {item['text']}" for item in detailed_transcript
    ])

    enhanced_prompt = f"""
    {prompt}
    Please ensure the summary is detailed, point-wise, and references each part of the video with timestamps. Use simple and clear language to explain the content. {length_prompt}
    Transcript:
    {transcript_for_summary}
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    detailed_summary = model.generate_content(enhanced_prompt).text

    # Generate short summary
    short_summary_prompt = f"""
    Based on the detailed summary provided below, create a very brief and concise summary in simple language:
    Detailed Summary:
    {detailed_summary}
    """

    short_summary = model.generate_content(short_summary_prompt).text

    return detailed_summary, short_summary

# Function to generate answers based on a question
def generate_answer(transcript_text, question):
    if not transcript_text:
        # If no transcript is available, mention it and use external knowledge
        st.warning("Answer not available in the video transcript. Fetching the answer from external knowledge base.")
        qa_prompt = f"""
        You are an intelligent assistant. The question could not be answered based on the video transcript because the information is not present. Using your general knowledge and resources, answer the following question as accurately as possible:
        Question: {question}
        Answer:
        """
    else:
        # Use the transcript to answer
        qa_prompt = f"""
        You are an intelligent assistant. Based on the following video transcript, answer the question as accurately as possible. If the information is not available in the transcript, provide an answer from your general knowledge and resources:
        Transcript: {transcript_text}
        Question: {question}
        Answer:
        """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(qa_prompt)
    return response.text

# Streamlit UI
def main():
    st.title("YouTube Video Summarizer and Q&A")

    # Initialize session state for summary visibility
    if "show_summary" not in st.session_state:
        st.session_state.show_summary = True

    # Layout setup
    col1, col2 = st.columns([1, 2])

    with col1:
        # Input for YouTube video link
        youtube_link = st.text_input("Enter YouTube Video Link:")

        if youtube_link:
            try:
                video_id = extract_video_id(youtube_link)
                st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)
            except ValueError as e:
                st.error(str(e))

    with col2:
        # Dropdown for summary length selection
        summary_length = st.selectbox(
            "Select Summary Length:",
            options=["Short", "Medium", "Long", "Custom"],
            index=1
        )

        # Input for custom summary length
        custom_length = None
        if summary_length == "Custom":
            custom_length = st.number_input(
                "Enter custom summary length (number of words):",
                min_value=50,
                max_value=1000,
                step=50,
                value=250
            )

        # Button to generate detailed notes
        if st.button("Get Detailed Notes"):
            if youtube_link:
                transcript_text, detailed_transcript = extract_transcript_details(youtube_link)

                if transcript_text and detailed_transcript:
                    # Generate and store the summary in session state
                    summary_limit = custom_length if summary_length == "Custom" else summary_length
                    detailed_summary, short_summary = generate_gemini_summary_with_timestamps(detailed_transcript, base_prompt, summary_limit)
                    st.session_state.detailed_summary = detailed_summary
                    st.session_state.short_summary = short_summary
                    st.session_state.show_summary = True
                else:
                    st.info("This video has only auto-generated captions. Please try another video.")
            else:
                st.error("Please provide a YouTube video link.")

        # Input for question
        question = st.text_input("**Ask a question about the video (optional):**")

        # Button to get answers for the question
        if st.button("Ask Question"):
            if youtube_link and question:
                transcript_text, _ = extract_transcript_details(youtube_link)

                if transcript_text:
                    answer = generate_answer(transcript_text, question)
                    st.markdown("### Answer to Your Question:")
                    st.write(answer)
                else:
                    st.info("This video has only auto-generated captions. Please try another video.")
            else:
                st.error("Please provide a YouTube video link and a question.")

    # Show the summary regardless of question activity
    if "detailed_summary" in st.session_state:
        with st.expander("## Detailed Notes with Timestamps (Click to Hide/Show):", expanded=st.session_state.show_summary):
            st.write(st.session_state.detailed_summary)

        st.markdown("### Short Summary:")
        st.write(st.session_state.short_summary)

if __name__ == "__main__":
    main()
