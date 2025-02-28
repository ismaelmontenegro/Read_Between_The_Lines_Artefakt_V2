# Meeting Analysis Dashboard

An advanced video meeting analysis system that provides real-time insights and visualizations about team dynamics, communication patterns, and meeting effectiveness. The system processes video recordings to generate comprehensive analytics about speaker interactions, sentiment, topic distribution, and various team dynamics metrics.

## Features

### Analysis
- Speaker diarization and identification
- Speech-to-text transcription
- Live updating visualizations
- Progressive analysis in time-buffered chunks

### Core Analytics
- **Speaker Analysis**
  - Speaking time distribution
  - Turn-taking patterns
  - Speaker identification and tracking
  - Individual contribution metrics

- **Content Analysis**
  - Transcription
  - Topic modeling and clustering
  - Sentiment analysis
  - Filler word detection
  - Knowledge tracking

- **Team Dynamics**
  - Participation balance
  - Response latency
  - Topic consensus
  - Team sentiment alignment

### Interactive Visualizations
- Team Sentiment Timeline
- Speaker Distribution Polygon
- Participation Balance Charts
- Response Dynamics Tracking
- Topic Distribution
- Turn-taking Patterns
- Meeting Objectivity Metrics
- Knowledge Coherence Graphs
- Filler Word Analysis

## Requirements

### System Requirements
- Python 3.12

### Key Dependencies
- oTree
- PyTorch
- pyannote.audio
- OpenAI Whisper
- TextBlob
- scikit-learn
- numpy
- scipy
- nltk
- Chart.js
- D3.js

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ismaelmontenegro/Read_Between_The_Lines_Artefakt_V2.git
cd Read_Between_The_Lines_Artefakt_V2
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up pyannote.audio:
```bash
# Get your HuggingFace token from https://huggingface.co/settings/tokens
# Replace YOUR-HF-TOKEN in videoTranscriberV2.py with your actual token
```

5. Download NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

1. Change Directory to the Demo folder:
```bash
cd Demo
```

2. Start the oTree development server:
```bash
otree devserver
```

3. Navigate to the application URL (typically `http://localhost:8000`)

4. Upload a video file through the interface

5. The system will automatically:
   - Extract audio from the video
   - Process the content in real-time
   - Generate interactive visualizations
   - Update the dashboard continuously

## Dashboard Interface

The dashboard provides both overview and detailed analysis views:

- **Main Video Player**: Watch the meeting recording with synchronized analytics
- **Active Visualization**: Large, detailed view of the selected metric
- **Minimized Views**: Quick access to other available visualizations
- **Transcript Panel**: Transcript with speaker identification
- **Analysis Controls**: Select different visualization types and control playback

## Configuration

Key configuration options can be modified in `videoTranscriberV2.py`:

```python
# Buffer size for processing (in seconds)
buffer_size=12

# Processing overlap (in seconds)
overlap=0

# Speaker embedding threshold
embedding_threshold=0.65

# Topic modeling update frequency
topic_update_frequency=3

# Sleep Interval after processing each buffer to simulate real-time processing (modify accordingly depending on CPU computing capacity)
time.sleep (2.22)

```

## Performance Optimization

For optimal performance:

1. Adjust buffer size based on available system resources
2. Process videos with clear audio quality
3. Ensure proper microphone placement in recorded meetings
4. Use recommended video formats (MP4 with H.264 encoding)


## Acknowledgments

- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [oTree](https://www.otree.org/) for web framework
- [Chart.js](https://www.chartjs.org/) and [D3.js](https://d3js.org/) for visualizations

## Author
Ismael Montenegro

## Future Development

Planned features and improvements:

- Multi-language support
- Custom visualization templates
- Advanced meeting analytics export
- Integration with video conferencing platforms
- Enhanced topic modeling algorithms
- Real-time meeting monitoring capabilities

## Troubleshooting

If the application appears to freeze on the loading screen:
1. Check your terminal/console for error messages
2. Common issues include:
   - Missing model authentication tokens
   - Insufficient system resources
   - Network connectivity problems when downloading models for first use
   - SSL Certificate verification on macOS
