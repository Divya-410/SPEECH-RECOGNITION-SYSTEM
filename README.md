# SPEECH-RECOGNITION-SYSTEM

COMPANY : CODTECH IT SOLUTIONS

NAME : YATHIRAJULA DIVYA SREE

INTERN ID : CT08DRC

DOMAIN : ARTIFICIAL INTELLIGENCE

DURATION : 12 WEEKS

MENTOR : NEELA SANTOSH

DESCRIPTION :
 
Overview of Wav2Vec2
Wav2Vec2 is a self-supervised learning model that directly processes raw audio waveforms and learns speech representations. Unlike traditional ASR systems, which require manually labeled datasets and phonetic dictionaries, Wav2Vec2 learns from unlabeled speech data and fine-tunes on specific languages or accents.

The key advantage of Wav2Vec2 is that it eliminates the need for feature engineering, as it learns effective representations directly from audio signals. It is widely used for applications such as voice assistants, automated subtitles, call center transcription, and accessibility tools.

Project Implementation
This project involves the following steps:

Preprocessing Audio:

The input audio file (WAV, MP3, or FLAC) is loaded using pydub or soundfile.
If the sample rate is not 16kHz, the audio is resampled using torchaudio.
Long audio files are split into smaller chunks for efficient processing.
Loading the Wav2Vec2 Model:

The Hugging Face Transformers library is used to load a pretrained Wav2Vec2 model.
The model converts raw waveforms into text-based transcriptions.
Inference & Transcription:

The audio waveform is passed through the model.
The model generates predictions using a CTC loss function (Connectionist Temporal Classification).
The output text is decoded and displayed as the final transcription.
Saving & Displaying Results:

The transcriptions are saved in a text file for further analysis.
Research & Sources
This project was developed using various research sources:
 Google: Used for researching speech recognition models, Python libraries, and Wav2Vec2 documentation.
 ChatGPT: Assisted in refining the Python code, debugging issues, and optimizing model performance.
 YouTube: Helpful for tutorials, implementation guides, and understanding real-world use cases of Wav2Vec2.
Research Papers: Provided deep insights into self-supervised learning, speech processing techniques, and improvements over traditional ASR models.

Applications of Speech Recognition
The Wav2Vec2-powered speech recognition system has diverse applications, including:

🔹 Voice Assistants: Used in Google Assistant, Siri, Alexa, and Cortana for hands-free interaction.
🔹 Medical Transcription: Automates clinical documentation for hospitals and healthcare professionals.
🔹 Automated Subtitles: Generates real-time captions for YouTube, online lectures, and media content.
🔹 Call Center Analytics: Converts customer service calls into text for sentiment analysis.
🔹 Accessibility Tools: Helps individuals with hearing impairments by transcribing conversations.
🔹 Legal Transcriptions: Converts court proceedings and meetings into structured text.
