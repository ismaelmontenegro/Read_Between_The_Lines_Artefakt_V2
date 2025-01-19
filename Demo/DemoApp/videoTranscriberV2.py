import os
from collections import defaultdict
import numpy as np
from moviepy import VideoFileClip
from pyannote.audio import Pipeline
from pyannote.audio import Model
import whisper
from textblob import TextBlob
import json
import soundfile as sf
import time
from pyannote.audio import Inference
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine
import torch
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class LiveMeetingAnalyzer:
    def __init__(self, video_path, buffer_size=12):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.step_size = buffer_size

        self.audio_path = "temp_audio.wav"
        self.current_position = 0
        self.total_duration = None
        self.stop_processing = False

        self.device = torch.device("cpu")

        # Initialize models
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token="YOUR-HF-TOKEN"
        ).to(self.device)
        self.embedding_model = Model.from_pretrained(
            "pyannote/embedding", use_auth_token="YOUR-HF-TOKEN"
        ).to(self.device)
        self.inference = Inference(self.embedding_model, window="whole").to(self.device)
        self.transcription_model = whisper.load_model("tiny",device="cuda")

        # Speaker tracking
        self.speaker_embeddings = {}  # {speaker_id: embedding_vector}
        self.embedding_threshold = 0.65  # Similarity threshold for speaker matching
        self.accumulated_results = {
            'meeting_duration': 0,
            'speaker_statistics': {},
            'transcript': [],
            'turn_taking': {
                'window_entropies': [],
                'entropy_peaks': []
            }
        }
        self.team_metrics = {
            'interaction_balance': [],
            'topic_consensus': {},
            'response_latency': [],
            'sentiment_alignment': [],
        }
        self.vectorizer = None
        self.overall_model = None
        self.overall_topics = []
        self.accumulated_texts = []
        self.speaker_accumulated_texts = {}

        # Add new attributes for improved topic modeling
        self.topic_update_frequency = 3  # Number of buffers before updating topics
        self.buffer_count = 0
        self.current_buffer_texts = []  # Texts from current set of buffers
        self.all_buffer_texts = []  # All accumulated texts
        self.speaker_buffer_texts = {}  # Texts per speaker for current set of buffers
        self.all_speaker_texts = {}  # All accumulated texts per speaker
        # Add vectorizer configuration
        self.vectorizer_config = {
            'max_df': 1.0,  # Ignore terms that appear in all documents
            'min_df': 1,  # Ignore terms that appear no documents
            'max_features': 1000,
            'stop_words': self.get_custom_stop_words(),
            'token_pattern': r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ characters
        }

        # Add tracking for cumulative texts
        self.buffer_count = 0
        self.update_frequency = 3  # Number of buffers before updating topics
        self.cumulative_texts = []  # All texts collected so far
        self.cumulative_speaker_texts = {}  # All texts per speaker

        self.filler_word_stats = {
            'overall': {
                'total_count': 0,
                'per_minute': [],
                'most_common': {}
            },
            'per_speaker': {}
        }

        # Basic filler words from last group plus common ones from NLTK
        self.filler_words = {
            'hesitation': ["um", "umm", "uum", "ah", "aah", "ahh", "uh", "uuh", "uhh", "er", "eer", "err", "ähm", "äh",
                           "öhm", "hmm"],
            'discourse': ["so", "soo", "okay", "well", "right", "and so", "you know", "i think", "like", "sort of",
                          "kind of", "basically", "actually", "literally"],
            'repair': ["sorry", "i mean", "what i meant", "rather"],
            'pause': ["..."]
        }

        # Add to accumulated results structure
        self.accumulated_results['filler_analysis'] = {
            'overall_stats': {
                'total_count': 0,
                'per_minute': [],
                'by_category': {},
                'most_common': {}
            },
            'speaker_stats': {}
        }

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

    def extract_audio(self):
        """Extract full audio from video file"""
        print("Extracting audio from video...")
        video = VideoFileClip(self.video_path)
        self.total_duration = video.duration
        video.audio.write_audiofile(self.audio_path, fps=16000)
        return self.audio_path

    def get_audio_segment(self, start_time, end_time):
        """Extract audio segment for the current buffer"""
        audio, sr = sf.read(self.audio_path, start=int(start_time * 16000),
                            frames=int((end_time - start_time) * 16000))
        return audio, sr

    def process_buffer(self, start_time, end_time):
        """Process a single buffer of audio with speaker embedding tracking"""
        try:
            audio_segment, sr = self.get_audio_segment(start_time, end_time)

            if len(audio_segment.shape) > 1:
                audio_segment = audio_segment.mean(axis=1)
            audio_segment = audio_segment.astype(np.float32)

            temp_buffer_path = "temp_buffer.wav"
            sf.write(temp_buffer_path, audio_segment, sr)

            diarization = self.diarization_pipeline(
                temp_buffer_path, min_speakers=1, max_speakers=5
            )

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                abs_start = start_time + turn.start
                abs_end = start_time + turn.end
                speaker_id = self.match_speaker_embedding(temp_buffer_path, turn)

                # Only add segments if the speaker has meaningful transcription
                if speaker_id != "SPEAKER_UNKNOWN":
                    segments.append({
                        'start': abs_start,
                        'end': abs_end,
                        'speaker': speaker_id,
                        'duration': turn.end - turn.start
                    })

            transcription = self.transcription_model.transcribe(
                audio_segment, language="english", verbose=False, fp16=False
            )

            # Apply speaker clustering periodically
            if self.buffer_count % 10 == 0:  # Every 10 buffers
                self.cluster_speakers()

            # Filter segments based on speaker confidence
            filtered_segments = [
                segment for segment in segments
                if self.calculate_speaker_confidence(segment['speaker'],
                                                     segment['end'] - segment['start'])
            ]

            # Filter out ghost speakers
            buffer_results = self.process_transcription(filtered_segments, transcription, start_time)
            buffer_results = [result for result in buffer_results if result['text'].strip()]

            # Calculate turn-taking entropy
            window_entropies = self.calculate_turn_taking_entropy(segments)
            entropy_peaks = self.detect_entropy_peaks(window_entropies)
            self.calculate_team_interaction_balance(segments)
            self.calculate_topic_consensus(buffer_results)
            self.calculate_response_latency(segments)

            # Update turn-taking data in accumulated results
            self.accumulated_results['turn_taking']['window_entropies'].extend(window_entropies)
            self.accumulated_results['turn_taking']['entropy_peaks'].extend(entropy_peaks)

            self.update_results(buffer_results, segments)
            self.calculate_recent_topic_alignment(buffer_results)
            self.save_results()

            if os.path.exists(temp_buffer_path):
                os.remove(temp_buffer_path)

            return True

        except Exception as e:
            print(f"Error processing buffer: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def generate_embedding(self, audio_path, turn):
        """Generate speaker embedding for a given audio segment"""
        try:
            # Extract audio segment for the current turn
            start = int(turn.start * 16000)
            end = int(turn.end * 16000)
            audio, sr = sf.read(audio_path, start=start, frames=end - start)

            # Check if audio is too short
            if len(audio) < 16000:  # Minimum 1 second
                return None

            # Save temporary audio segment
            temp_audio_path = "temp_segment.wav"
            sf.write(temp_audio_path, audio, sr)

            try:
                # Generate embedding using pyannote's Inference
                embedding = self.inference(temp_audio_path)
                # Reshape to 2D array for distance calculations
                embedding = embedding.reshape(1, -1)
                return embedding

            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None

    def match_speaker_embedding(self, audio_path, turn):
        """Enhanced speaker matching with historical context"""
        try:
            # Generate current embedding
            embedding = self.generate_embedding(audio_path, turn)
            if embedding is None:
                return "SPEAKER_UNKNOWN"

            # Initialize tracking of recent embeddings if not exists
            if not hasattr(self, 'recent_embeddings'):
                self.recent_embeddings = defaultdict(list)
                self.max_history = 5  # Keep last 5 embeddings per speaker

            # Compare with existing speakers using average of recent embeddings
            best_match = None
            best_similarity = float('inf')

            for speaker_id, recent_embs in self.recent_embeddings.items():
                if recent_embs:
                    # Calculate average similarity across recent embeddings
                    similarities = [
                        cdist(embedding, ref_emb, metric="cosine")[0, 0]
                        for ref_emb in recent_embs
                    ]
                    avg_similarity = np.mean(similarities)

                    if avg_similarity < best_similarity and avg_similarity < self.embedding_threshold:
                        best_similarity = avg_similarity
                        best_match = speaker_id

            # If match found, update historical embeddings
            if best_match:
                self.recent_embeddings[best_match].append(embedding)
                if len(self.recent_embeddings[best_match]) > self.max_history:
                    self.recent_embeddings[best_match].pop(0)
                return best_match

            # If no match, create new speaker
            new_id = f"SPEAKER_{len(self.recent_embeddings):02d}"
            self.recent_embeddings[new_id].append(embedding)
            return new_id

        except Exception as e:
            print(f"Error in speaker matching: {e}")
            return "SPEAKER_UNKNOWN"

    def process_transcription(self, segments, transcription, buffer_start):
        results = []

        for segment in transcription['segments']:
            start_time = buffer_start + segment['start']
            end_time = buffer_start + segment['end']
            text = segment['text']

            speaker = None
            for s in segments:
                if (start_time >= s['start'] and start_time < s['end']) or \
                        (end_time > s['start'] and end_time <= s['end']):
                    speaker = s['speaker']
                    break

            if speaker and text.strip():
                # Add filler word analysis
                self.analyze_filler_words(text, speaker, start_time)

                sentiment = self.analyze_sentiment(text)
                results.append({
                    'start': start_time,
                    'end': end_time,
                    'speaker': speaker,
                    'text': text,
                    'sentiment': sentiment
                })

        return results

    def analyze_filler_words(self, text, speaker_id, timestamp):
        """Analyze text for filler words and update statistics"""
        filler_counts = {category: 0 for category in self.filler_words.keys()}
        specific_fillers = {}

        words = text.lower().split()

        # Check for single word fillers
        for word in words:
            for category, fillers in self.filler_words.items():
                if word in fillers:
                    filler_counts[category] += 1
                    specific_fillers[word] = specific_fillers.get(word, 0) + 1

        # Check for multi-word fillers (phrases)
        text_lower = text.lower()
        for category, fillers in self.filler_words.items():
            for filler in fillers:
                if ' ' in filler and filler in text_lower:
                    filler_counts[category] += text_lower.count(filler)
                    specific_fillers[filler] = specific_fillers.get(filler, 0) + text_lower.count(filler)

        total_fillers = sum(filler_counts.values())

        # Update overall statistics
        self.accumulated_results['filler_analysis']['overall_stats']['total_count'] += total_fillers

        # Update per-minute statistics
        minute = int(timestamp // 60)
        while len(self.accumulated_results['filler_analysis']['overall_stats']['per_minute']) <= minute:
            self.accumulated_results['filler_analysis']['overall_stats']['per_minute'].append(0)
        self.accumulated_results['filler_analysis']['overall_stats']['per_minute'][minute] += total_fillers

        # Update category statistics
        for category, count in filler_counts.items():
            if category not in self.accumulated_results['filler_analysis']['overall_stats']['by_category']:
                self.accumulated_results['filler_analysis']['overall_stats']['by_category'][category] = 0
            self.accumulated_results['filler_analysis']['overall_stats']['by_category'][category] += count

        # Update most common fillers
        for filler, count in specific_fillers.items():
            if filler not in self.accumulated_results['filler_analysis']['overall_stats']['most_common']:
                self.accumulated_results['filler_analysis']['overall_stats']['most_common'][filler] = 0
            self.accumulated_results['filler_analysis']['overall_stats']['most_common'][filler] += count

        # Update per-speaker statistics
        if speaker_id not in self.accumulated_results['filler_analysis']['speaker_stats']:
            self.accumulated_results['filler_analysis']['speaker_stats'][speaker_id] = {
                'total_count': 0,
                'per_minute': [],
                'by_category': {},
                'most_common': {}
            }

        speaker_stats = self.accumulated_results['filler_analysis']['speaker_stats'][speaker_id]
        speaker_stats['total_count'] += total_fillers

        # Update speaker's per-minute stats
        while len(speaker_stats['per_minute']) <= minute:
            speaker_stats['per_minute'].append(0)
        speaker_stats['per_minute'][minute] += total_fillers

        # Update speaker's category stats
        for category, count in filler_counts.items():
            if category not in speaker_stats['by_category']:
                speaker_stats['by_category'][category] = 0
            speaker_stats['by_category'][category] += count

        # Update speaker's most common fillers
        for filler, count in specific_fillers.items():
            if filler not in speaker_stats['most_common']:
                speaker_stats['most_common'][filler] = 0
            speaker_stats['most_common'][filler] += count

    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    def is_significant_speaker(self, speaker, stats, total_time):
        """
        Determine if a speaker is significant based on both participation percentage
        and minimum number of utterances.
        """
        # Calculate participation percentage
        participation_percentage = (stats['speaking_time'] / total_time * 100) if total_time > 0 else 0

        # Count number of utterances for this speaker
        utterance_count = sum(1 for segment in self.accumulated_results['transcript']
                              if segment['speaker'] == speaker)

        # Speaker must meet BOTH criteria to be considered significant
        return (participation_percentage >= 1.0 and utterance_count >= 3)

    def save_results(self):
        # Calculate total time once
        total_time = sum(s['speaking_time'] for s in self.accumulated_results['speaker_statistics'].values())

        # Identify significant speakers using both criteria
        significant_speakers = {
            speaker: stats for speaker, stats in self.accumulated_results['speaker_statistics'].items()
            if self.is_significant_speaker(speaker, stats, total_time)
        }

        # Rest of the save_results method remains the same, using significant_speakers for filtering
        ...

    def get_custom_stop_words(self):
        """Get extended stop words list including common filler words"""
        custom_stops = [
            'like', 'just', 'um', 'uh', 'well', 'sort', 'kind', 'yeah', 'yes', 'know',
            'mean', 'right', 'going', 'get', 'got', 'gonna', 'would', 'could', 'should',
            'really', 'say', 'saying', 'said', 'way', 'thing', 'things', 'think', 'thinking',
            'actually', 'basically', 'certainly', 'definitely', 'probably', 'possibly',
            'ok', 'okay', 'hey', 'oh', 'ow', 'wow', 'ah', 'uhm', 'like', 'so', 'also'
        ]
        custom_stops.extend(stopwords.words('english'))
        return list(dict.fromkeys(custom_stops))

    def clean_word(self, word):
        """Clean and validate a word"""
        cleaned = ''.join(c for c in word.lower() if c.isalpha())
        if len(cleaned) < 3:
            return None
        return cleaned

    def generate_topic_name(self, top_words, word_weights):
        """Generate a descriptive name for a topic based on its top words and weights"""
        significant_words = []
        custom_stops = self.get_custom_stop_words()

        for word, weight in zip(top_words, word_weights):
            cleaned_word = self.clean_word(word)
            if cleaned_word and cleaned_word not in custom_stops:
                formatted_word = cleaned_word.capitalize()
                if len(significant_words) < 3 and formatted_word not in significant_words:
                    significant_words.append(formatted_word)

        if len(significant_words) < 2:
            return "General Discussion"

        if len(significant_words) == 2:
            return f"{significant_words[0]} & {significant_words[1]}"
        return f"{significant_words[0]}, {significant_words[1]} & {significant_words[2]}"

    def should_update_topics(self):
        """Check if we should perform topic modeling based on buffer count"""
        return self.buffer_count % self.topic_update_frequency == 0

    def update_topic_texts(self, buffer_results):
        """Update text collections for topic modeling"""
        # Update current buffer texts
        new_texts = [result['text'] for result in buffer_results if len(result['text'].strip()) > 10]
        self.current_buffer_texts.extend(new_texts)

        # Update speaker-specific texts
        for result in buffer_results:
            if len(result['text'].strip()) > 10:
                speaker = result['speaker']
                # Update current buffer texts per speaker
                if speaker not in self.speaker_buffer_texts:
                    self.speaker_buffer_texts[speaker] = []
                self.speaker_buffer_texts[speaker].append(result['text'])

                # Update all-time texts per speaker
                if speaker not in self.all_speaker_texts:
                    self.all_speaker_texts[speaker] = []
                self.all_speaker_texts[speaker].append(result['text'])

    def perform_periodic_topic_modeling(self):
        """Perform topic modeling on accumulated texts when appropriate"""
        if not self.should_update_topics():
            return

        # Add current buffer texts to all-time collection
        self.all_buffer_texts.extend(self.current_buffer_texts)

        # Perform topic modeling on all accumulated texts
        if len(self.all_buffer_texts) >= 10:  # Minimum threshold for meaningful topics
            # Create fresh vectorizer for this round
            vectorizer = CountVectorizer(**self.vectorizer_config)

            # Update overall topics
            overall_topics, _, _ = self.perform_topic_modeling(
                self.all_buffer_texts,
                vectorizer=vectorizer,
                update_overall=True,
                num_topics=1  # Dynamic topic count
            )

            if overall_topics:
                self.accumulated_results['topics'] = {
                    'overall': overall_topics,
                    'per_speaker': {}
                }

                # Update per-speaker topics
                for speaker, texts in self.all_speaker_texts.items():
                    if len(texts) >= 5:  # Minimum threshold for speaker topics
                        speaker_topics, _, speaker_model = self.perform_topic_modeling(
                            texts,
                            vectorizer=vectorizer,  # Use same vectorizer for consistency
                            num_topics= 1  # Dynamic topic count
                        )

                        if speaker_topics and speaker_model:
                            topic_similarities = self.calculate_topic_similarity(
                                speaker_model, speaker_topics
                            )
                            self.accumulated_results['topics']['per_speaker'][speaker] = {
                                'topics': speaker_topics,
                                'topic_similarities': topic_similarities
                            }

        # Reset current buffer collections
        self.current_buffer_texts = []
        self.speaker_buffer_texts = {}

    def perform_topic_modeling(self, texts, vectorizer=None, update_overall=False, num_topics=1, num_words=15):
        """Perform topic modeling with adjusted parameters for small text volumes"""
        # Ensure we have enough meaningful text to analyze
        filtered_texts = [text for text in texts if len(text.split()) >= 2]  # Changed from 3 to 2

        if not filtered_texts:
            print("No valid texts for topic modeling")
            return [], None, None

        try:
            # Dynamically adjust vectorizer parameters based on text volume
            current_config = self.vectorizer_config.copy()
            if len(filtered_texts) < 5:
                current_config.update({
                    'min_df': 1,
                    'max_df': 1.0,
                })
            elif len(filtered_texts) < 10:
                current_config.update({
                    'min_df': 1,
                    'max_df': 0.99,
                })

            # Use provided vectorizer or create new one
            if vectorizer is None:
                vectorizer = CountVectorizer(**current_config)

            # Transform texts
            doc_term_matrix = vectorizer.fit_transform(filtered_texts)

            # Check if we have enough terms
            if doc_term_matrix.shape[1] < 2:  # Changed from 3 to 2
                print(f"Insufficient terms found ({doc_term_matrix.shape[1]} terms)")
                return [], None, None

            # Adjust number of topics based on available terms
            adjusted_num_topics = min(
                num_topics,
                doc_term_matrix.shape[1] // 2,  # Changed from 3 to 2
                len(filtered_texts) // 2  # Changed from 3 to 2
            )

            if adjusted_num_topics < 1:
                adjusted_num_topics = 1

            # Create and fit LDA model with adjusted parameters
            lda_model = LatentDirichletAllocation(
                n_components=adjusted_num_topics,
                random_state=42,
                max_iter=10,
                learning_method='online',
                n_jobs=-1,
                doc_topic_prior=0.1,  # Changed from 0.9 to be less restrictive
                topic_word_prior=0.1  # Changed from 0.9 to be less restrictive
            )

            # Fit the model
            doc_topic_distributions = lda_model.fit_transform(doc_term_matrix)

            # Extract topics
            topics = []
            feature_names = vectorizer.get_feature_names_out()

            for topic_idx, topic in enumerate(lda_model.components_):
                top_words_idx = topic.argsort()[:-num_words - 1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                word_weights = [topic[i] for i in top_words_idx]

                topic_name = self.generate_topic_name(top_words, word_weights)
                topic_weight = float(np.mean(doc_topic_distributions[:, topic_idx]))

                topics.append({
                    'topic_id': topic_idx,
                    'name': topic_name,
                    'top_words': top_words,
                    'weight': topic_weight
                })

            if update_overall:
                self.overall_model = lda_model
                self.overall_topics = topics

            return topics, doc_topic_distributions, lda_model

        except Exception as e:
            print(f"Warning: Topic modeling error - {str(e)}")
            import traceback
            traceback.print_exc()
            return [], None, None

    def calculate_recent_topic_alignment(self, buffer_results, window_size=5):
        """
        Calculate topic alignment based on recent utterances with enhanced warning system.

        Args:
            buffer_results: Results from current buffer
            window_size: Number of minutes to look back for recent utterances
        """
        try:
            # Initialize recent utterances tracking if not exists
            if not hasattr(self, 'recent_utterances'):
                self.recent_utterances = {
                    'overall': [],
                    'per_speaker': {}
                }

            # Initialize warning tracking if not exists
            if not hasattr(self, 'off_topic_tracking'):
                self.off_topic_tracking = {
                    'consecutive_warnings': {},
                    'last_status': {}
                }

            # Define warning level function
            def get_warning_level(score):
                if score >= 0.65:
                    return None
                elif score >= 0.5:
                    return {
                        'severity': 'mild',
                        'message': 'Speaker is slightly off-topic',
                        'score': score
                    }
                elif score >= 0.3:
                    return {
                        'severity': 'moderate',
                        'message': 'Speaker is moderately off-topic',
                        'score': score
                    }
                else:
                    return {
                        'severity': 'severe',
                        'message': 'Speaker is severely off-topic',
                        'score': score
                    }

            def get_topic_suggestion(speaker_topics, overall_topics, similarity_matrix):
                # Find the overall topics that are least similar to speaker's current topics
                least_discussed = []
                for j in range(len(overall_topics)):
                    col_max = np.max(similarity_matrix[:, j])
                    least_discussed.append((j, col_max))

                # Sort by similarity ascending (least similar first)
                least_discussed.sort(key=lambda x: x[1])

                # Get the top 2 least discussed topics
                suggestions = []
                for idx, _ in least_discussed[:2]:
                    topic = overall_topics[idx]
                    suggestions.append({
                        'topic_name': topic['name'],
                        'key_terms': topic['top_words'][:5]  # First 5 key terms
                    })

                return suggestions

            # Get current time from the latest utterance
            current_time = max(result['end'] for result in buffer_results)
            cutoff_time = current_time - (window_size * 60)  # Convert minutes to seconds

            # Update recent utterances with new buffer results
            for result in buffer_results:
                if len(result['text'].strip()) > 10:  # Only consider meaningful utterances
                    # Add to overall recent utterances
                    self.recent_utterances['overall'].append({
                        'text': result['text'],
                        'time': result['end']
                    })

                    # Add to speaker-specific recent utterances
                    speaker = result['speaker']
                    if speaker not in self.recent_utterances['per_speaker']:
                        self.recent_utterances['per_speaker'][speaker] = []

                    self.recent_utterances['per_speaker'][speaker].append({
                        'text': result['text'],
                        'time': result['end']
                    })

            # Clean up old utterances
            self.recent_utterances['overall'] = [
                u for u in self.recent_utterances['overall']
                if u['time'] > cutoff_time
            ]

            for speaker in self.recent_utterances['per_speaker']:
                self.recent_utterances['per_speaker'][speaker] = [
                    u for u in self.recent_utterances['per_speaker'][speaker]
                    if u['time'] > cutoff_time
                ]

            # Initialize vectorizer with current configuration
            vectorizer = CountVectorizer(**self.vectorizer_config)

            # Get recent overall texts and fit vectorizer
            overall_texts = [u['text'] for u in self.recent_utterances['overall']]

            if len(overall_texts) < 3:  # Need minimum number of utterances for meaningful analysis
                return

            # Fit vectorizer on all texts to maintain consistent vocabulary
            vectorizer.fit(overall_texts)

            # Perform topic modeling on overall recent texts
            overall_dtm = vectorizer.transform(overall_texts)

            # Adjust number of topics based on available data
            n_topics = max(1, min(3, len(overall_texts) // 5))

            overall_lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10,
                learning_method='online',
                n_jobs=-1,
                doc_topic_prior=0.1,
                topic_word_prior=0.1
            )

            overall_doc_topics = overall_lda.fit_transform(overall_dtm)
            overall_topics = self.extract_topics(overall_lda, vectorizer, overall_doc_topics)

            # Calculate speaker alignments
            alignments = {}

            for speaker, utterances in self.recent_utterances['per_speaker'].items():
                speaker_texts = [u['text'] for u in utterances]

                if len(speaker_texts) < 2:  # Skip if too few utterances
                    continue

                # Transform speaker texts using same vectorizer
                speaker_dtm = vectorizer.transform(speaker_texts)

                # Create and fit speaker LDA
                speaker_lda = LatentDirichletAllocation(
                    n_components=max(1, min(2, len(speaker_texts) // 3)),
                    random_state=42,
                    max_iter=10,
                    learning_method='online',
                    n_jobs=-1,
                    doc_topic_prior=0.1,
                    topic_word_prior=0.1
                )

                speaker_doc_topics = speaker_lda.fit_transform(speaker_dtm)
                speaker_topics = self.extract_topics(speaker_lda, vectorizer, speaker_doc_topics)

                # Calculate similarity between speaker and overall topics
                similarity_matrix = cosine_similarity(
                    speaker_lda.components_,
                    overall_lda.components_
                )

                # Calculate average similarity across all topic pairs
                avg_similarity = float(np.mean(similarity_matrix))

                # Calculate warning level and track consecutive warnings
                warning_info = get_warning_level(avg_similarity)

                # Initialize tracking for this speaker if needed
                if speaker not in self.off_topic_tracking['consecutive_warnings']:
                    self.off_topic_tracking['consecutive_warnings'][speaker] = 0
                    self.off_topic_tracking['last_status'][speaker] = None

                # Update consecutive warnings
                if warning_info:
                    if self.off_topic_tracking['last_status'][speaker]:
                        self.off_topic_tracking['consecutive_warnings'][speaker] += 1
                    topic_suggestions = get_topic_suggestion(speaker_topics, overall_topics, similarity_matrix)
                else:
                    self.off_topic_tracking['consecutive_warnings'][speaker] = 0
                    topic_suggestions = []

                # Update last status
                self.off_topic_tracking['last_status'][speaker] = warning_info is not None

                # Find best matching topics
                topic_matches = []
                for i, speaker_topic in enumerate(speaker_topics):
                    best_match_idx = np.argmax(similarity_matrix[i])
                    similarity_score = float(similarity_matrix[i][best_match_idx])

                    topic_matches.append({
                        'speaker_topic': speaker_topic['name'],
                        'overall_topic': overall_topics[best_match_idx]['name'],
                        'similarity_score': similarity_score
                    })

                # Store results with enhanced warning information
                alignments[speaker] = {
                    'recent_alignment_score': avg_similarity,
                    'timestamp': current_time,
                    'speaker_topics': speaker_topics,
                    'topic_matches': topic_matches,
                    'warning_info': warning_info,
                    'consecutive_warnings': self.off_topic_tracking['consecutive_warnings'][speaker],
                    'topic_suggestions': topic_suggestions if warning_info else []
                }

            # Update accumulated results with recent alignment data
            if 'recent_topic_alignment' not in self.accumulated_results:
                self.accumulated_results['recent_topic_alignment'] = {
                    'timeline': {},
                    'current_scores': {}
                }

            # Update timeline and current scores
            for speaker, alignment in alignments.items():
                if speaker not in self.accumulated_results['recent_topic_alignment']['timeline']:
                    self.accumulated_results['recent_topic_alignment']['timeline'][speaker] = []

                self.accumulated_results['recent_topic_alignment']['timeline'][speaker].append({
                    'time': alignment['timestamp'],
                    'alignment_score': alignment['recent_alignment_score'],
                    'topic_matches': alignment['topic_matches'],
                    'warning_info': alignment['warning_info'],
                    'consecutive_warnings': alignment['consecutive_warnings'],
                    'topic_suggestions': alignment['topic_suggestions']
                })

                self.accumulated_results['recent_topic_alignment']['current_scores'][speaker] = {
                    'score': alignment['recent_alignment_score'],
                    'topics': alignment['speaker_topics'],
                    'matches': alignment['topic_matches'],
                    'warning_info': alignment['warning_info'],
                    'consecutive_warnings': alignment['consecutive_warnings'],
                    'topic_suggestions': alignment['topic_suggestions']
                }

        except Exception as e:
            print(f"Error in recent topic alignment calculation: {str(e)}")
            import traceback
            traceback.print_exc()

    def cluster_speakers(self, threshold=0.02):
        """Merge speakers with very similar embeddings and low participation."""
        # Get all speakers with their stats
        speakers = self.accumulated_results['speaker_statistics']

        # Add missing speaking_time_percentage for any speaker
        total_time = sum(s['speaking_time'] for s in speakers.values())
        for speaker, stats in speakers.items():
            if 'speaking_time_percentage' not in stats:
                stats['speaking_time_percentage'] = (
                    (stats['speaking_time'] / total_time * 100) if total_time > 0 else 0
                )

        # Filter out speakers with significant participation
        major_speakers = {s: stats for s, stats in speakers.items()
                          if stats.get('speaking_time_percentage', 0) >= threshold * 100}
        minor_speakers = {s: stats for s, stats in speakers.items()
                          if stats.get('speaking_time_percentage', 0) < threshold * 100}

        # For each minor speaker, try to merge with the closest major speaker
        for minor_speaker, minor_stats in minor_speakers.items():
            if minor_speaker in self.speaker_embeddings:
                best_match = None
                best_similarity = float('inf')

                # Compare with major speakers
                for major_speaker in major_speakers:
                    if major_speaker in self.speaker_embeddings:
                        similarity = cdist(
                            self.speaker_embeddings[minor_speaker],
                            self.speaker_embeddings[major_speaker],
                            metric="cosine"
                        )[0, 0]

                        if similarity < best_similarity:
                            best_similarity = similarity
                            best_match = major_speaker

                # Merge if a good match is found
                if best_match and best_similarity < self.embedding_threshold:
                    self.merge_speakers(minor_speaker, best_match)

    def calculate_speaker_confidence(self, speaker_id, segment_duration):
        """Calculate confidence score for speaker identification."""
        if not hasattr(self, 'speaker_confidence'):
            self.speaker_confidence = defaultdict(float)

        # Base confidence on speaking time and consistency
        if speaker_id in self.accumulated_results['speaker_statistics']:
            stats = self.accumulated_results['speaker_statistics'][speaker_id]

            # Ensure 'speaking_time_percentage' is calculated
            total_time = sum(s['speaking_time'] for s in self.accumulated_results['speaker_statistics'].values())
            if 'speaking_time_percentage' not in stats:
                stats['speaking_time_percentage'] = (
                    (stats['speaking_time'] / total_time * 100) if total_time > 0 else 0
                )

            speaking_percentage = stats['speaking_time_percentage']

            # Higher confidence for speakers with more speaking time
            time_confidence = min(speaking_percentage / 10, 1.0)  # Max out at 10% speaking time

            # Consider embedding consistency if available
            embedding_confidence = 0.0
            if speaker_id in self.recent_embeddings and len(self.recent_embeddings[speaker_id]) > 1:
                similarities = []
                embeddings = self.recent_embeddings[speaker_id]
                for i in range(len(embeddings) - 1):
                    sim = 1 - cdist(embeddings[i], embeddings[i + 1], metric="cosine")[0, 0]
                    similarities.append(sim)
                embedding_confidence = np.mean(similarities)

            # Combine confidence scores
            confidence = 0.7 * time_confidence + 0.3 * embedding_confidence
            self.speaker_confidence[speaker_id] = confidence

            # Filter out low confidence speakers
            if confidence < 0.1:  # Threshold for considering valid speaker
                return False

        return True

    def calculate_topic_similarity(self, speaker_model, speaker_topics):
        """Calculate similarity between speaker topics and overall meeting topics"""
        if not speaker_model or not self.overall_model:
            return []

        similarities = []
        speaker_topic_term = speaker_model.components_
        overall_topic_term = self.overall_model.components_

        similarity_matrix = cosine_similarity(speaker_topic_term, overall_topic_term)

        for i, speaker_topic in enumerate(speaker_topics):
            best_match_idx = np.argmax(similarity_matrix[i])
            similarity_score = float(similarity_matrix[i][best_match_idx])

            similarities.append({
                'speaker_topic': speaker_topic['name'],
                'overall_topic': self.overall_topics[best_match_idx]['name'],
                'similarity_score': similarity_score
            })

        return similarities

    def extract_topics(self, lda_model, vectorizer, doc_topics, num_words=15):
        """Extract topics from LDA model using consistent feature names"""
        topics = []
        feature_names = vectorizer.get_feature_names_out()

        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[:-num_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            word_weights = [topic[i] for i in top_words_idx]

            topic_name = self.generate_topic_name(top_words, word_weights)
            topic_weight = float(np.mean(doc_topics[:, topic_idx]))

            topics.append({
                'topic_id': topic_idx,
                'name': topic_name,
                'top_words': top_words,
                'weight': topic_weight
            })

        return topics

    def calculate_turn_taking_entropy(self, segments, buffer_size=12):
        """Calculate turn-taking entropy for the current buffer, maintaining a sliding history"""
        if not hasattr(self, 'transition_history'):
            self.transition_history = []

        # Add new transitions from current buffer
        previous_speaker = None
        sorted_segments = sorted(segments, key=lambda x: x['start'])

        for segment in sorted_segments:
            current_speaker = segment['speaker']
            if previous_speaker and current_speaker != previous_speaker:
                self.transition_history.append({
                    'time': segment['start'],
                    'from': previous_speaker,
                    'to': current_speaker
                })
            previous_speaker = current_speaker

        # Calculate entropy for current buffer
        if len(self.transition_history) > 1:
            transition_counts = {}
            for t in self.transition_history:
                key = (t['from'], t['to'])
                transition_counts[key] = transition_counts.get(key, 0) + 1

            total_transitions = sum(transition_counts.values())
            probabilities = [count / total_transitions for count in transition_counts.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities)
        else:
            entropy = 0

        # Keep only transitions within recent history (e.g., last 60 seconds)
        current_time = max(segment['end'] for segment in segments)
        self.transition_history = [t for t in self.transition_history
                                   if current_time - t['time'] <= 60]

        return [{
            'start_time': min(segment['start'] for segment in segments),
            'end_time': max(segment['end'] for segment in segments),
            'entropy': entropy
        }]

    def detect_entropy_peaks(self, window_entropies):
        """Detect peaks in entropy values indicating critical instabilities."""
        if not window_entropies:
            return []

        # Get current entropy value
        current_entropy = window_entropies[-1]['entropy']

        # Track previous entropy for comparison
        if not hasattr(self, 'previous_entropy'):
            self.previous_entropy = 0

        # Detect if this is a peak (higher than previous and threshold)
        peaks = []
        if current_entropy > self.previous_entropy and current_entropy > 0.5:  # Adjust threshold as needed
            peaks.append({
                'time': window_entropies[-1]['start_time'],
                'entropy_value': current_entropy
            })

        self.previous_entropy = current_entropy
        return peaks

    def calculate_knowledge_tracking(self, buffer_results):
        """Calculate knowledge tracking metrics showing how each speaker's language converges with group semantics"""

        # Skip if not enough content
        if not buffer_results:
            return

        # Get all texts from current buffer
        buffer_texts = [result['text'] for result in buffer_results if result['text'].strip()]

        # Skip if no meaningful text in buffer
        if not buffer_texts:
            return

        # Initialize knowledge tracking results if not present
        if 'knowledge_tracking' not in self.accumulated_results:
            self.accumulated_results['knowledge_tracking'] = {
                'group_centroid': None,
                'speaker_tracking': {},
                'vocabulary': None,
                'vectorizer': None,
                'team_coherence': {
                    'values': [],
                    'final_centroid': None
                }
            }

        try:
            # First, gather all texts to create a consistent vocabulary
            all_texts = self.cumulative_texts + buffer_texts
            if len(all_texts) == 0:
                return

            # Create new vectorizer only if vocabulary needs updating
            needs_new_vectorizer = False
            if self.accumulated_results['knowledge_tracking']['vectorizer'] is None:
                needs_new_vectorizer = True
            else:
                # Check if we have new terms that aren't in our current vocabulary
                current_vocab = set(self.accumulated_results['knowledge_tracking']['vocabulary'])
                new_terms = set(' '.join(buffer_texts).split())
                if not new_terms.issubset(current_vocab):
                    needs_new_vectorizer = True

            if needs_new_vectorizer:
                # Create new vectorizer and fit on all available text
                vectorizer = CountVectorizer(**self.vectorizer_config)
                vectorizer.fit(all_texts)

                # Store the vectorizer and vocabulary
                self.accumulated_results['knowledge_tracking']['vectorizer'] = vectorizer
                self.accumulated_results['knowledge_tracking']['vocabulary'] = vectorizer.get_feature_names_out()

                # If we had previous speaker centroids, we need to recalculate them with new vocabulary
                if self.accumulated_results['knowledge_tracking']['speaker_tracking']:
                    for speaker in self.accumulated_results['knowledge_tracking']['speaker_tracking'].keys():
                        speaker_texts = []
                        for result in self.accumulated_results['transcript']:
                            if result['speaker'] == speaker:
                                speaker_texts.append(result['text'])

                        if speaker_texts:
                            speaker_dtm = vectorizer.transform(speaker_texts)
                            speaker_centroid = np.array(speaker_dtm.mean(axis=0)).flatten()
                            self.accumulated_results['knowledge_tracking']['speaker_tracking'][speaker][
                                'running_centroid'] = speaker_centroid.tolist()

            # Use the stored vectorizer
            vectorizer = self.accumulated_results['knowledge_tracking']['vectorizer']

            # Calculate current group centroid and transform all texts
            current_group_dtm = vectorizer.transform(all_texts)
            current_group_centroid = np.array(current_group_dtm.mean(axis=0)).flatten()
            self.accumulated_results['knowledge_tracking']['group_centroid'] = current_group_centroid.tolist()

            # Update final centroid
            self.accumulated_results['knowledge_tracking']['team_coherence'][
                'final_centroid'] = current_group_centroid.tolist()

            # Calculate team's coherence with its final state
            if self.accumulated_results['knowledge_tracking']['team_coherence']['values']:
                previous_values = self.accumulated_results['knowledge_tracking']['team_coherence']['values']
                last_time = previous_values[-1]['time']
            else:
                last_time = 0

            # Add current coherence value
            current_time = max(result['end'] for result in buffer_results)
            if current_time > last_time:
                self.accumulated_results['knowledge_tracking']['team_coherence']['values'].append({
                    'time': current_time,
                    'coherence': 1.0
                })

                # Recalculate previous coherence values
                if len(previous_values) > 0:
                    epsilon = 1e-10
                    final_centroid = current_group_centroid + epsilon

                    for prev_value in previous_values[:-1]:
                        time = prev_value['time']
                        historical_texts = [
                            r['text'] for r in self.accumulated_results['transcript']
                            if r['end'] <= time and r['text'].strip()
                        ]
                        if historical_texts:
                            historical_dtm = vectorizer.transform(historical_texts)
                            historical_centroid = np.array(historical_dtm.mean(axis=0)).flatten() + epsilon
                            prev_value['coherence'] = float(1 - cosine(historical_centroid, final_centroid))

            # Calculate per-speaker tracking
            for result in buffer_results:
                speaker = result['speaker']
                text = result['text']

                if not text.strip():
                    continue

                # Initialize speaker tracking if needed
                if speaker not in self.accumulated_results['knowledge_tracking']['speaker_tracking']:
                    self.accumulated_results['knowledge_tracking']['speaker_tracking'][speaker] = {
                        'coherence_values': [],
                        'running_centroid': None
                    }

                # Transform speaker's text using the same vocabulary
                speaker_vector = vectorizer.transform([text]).toarray()[0]

                # Update speaker's running centroid
                speaker_tracking = self.accumulated_results['knowledge_tracking']['speaker_tracking'][speaker]
                if speaker_tracking['running_centroid'] is None:
                    speaker_tracking['running_centroid'] = speaker_vector
                else:
                    prev_centroid = np.array(speaker_tracking['running_centroid'])
                    n = len(speaker_tracking['coherence_values']) + 1
                    speaker_tracking['running_centroid'] = (
                            (prev_centroid * (n - 1) + speaker_vector) / n
                    ).tolist()

                # Calculate coherence with group centroid
                epsilon = 1e-10
                speaker_centroid = np.array(speaker_tracking['running_centroid']) + epsilon
                group_centroid_eps = current_group_centroid + epsilon

                coherence = 1 - cosine(speaker_centroid, group_centroid_eps)
                speaker_tracking['coherence_values'].append({
                    'time': result['start'],
                    'coherence': float(coherence)
                })

        except Exception as e:
            print(f"Error calculating knowledge tracking: {str(e)}")
            import traceback
            traceback.print_exc()

    def calculate_gini(self, values):
        """
        Calculate Gini coefficient using mean absolute difference formula
        Args:
            values: List or array of participation values
        """
        arr = np.array(values)
        # Return 0 if there's only one speaker or empty array
        if len(arr) <= 1:
            return 0

        # Calculate mean absolute difference
        mad = np.abs(np.subtract.outer(arr, arr)).mean()
        # Calculate Gini coefficient
        gini = mad / (2 * np.mean(arr))
        return gini

    def calculate_team_interaction_balance(self, segments):
        """Calculate balance of participation among team members using Gini coefficient"""
        if not segments:
            return 0.0

        # Calculate speaking time for each speaker
        speaker_times = {}
        total_time = 0

        for segment in segments:
            speaker = segment['speaker']
            duration = segment['duration']

            if speaker != "SPEAKER_UNKNOWN":
                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
                total_time += duration

        if total_time == 0 or len(speaker_times) <= 1:
            return 0.0

        # Calculate participation percentages
        participation = [time / total_time for time in speaker_times.values()]

        # For debugging
        print(f"Speaker times: {speaker_times}")
        print(f"Participation percentages: {participation}")

        # Calculate balance score (1 - Gini coefficient)
        gini = self.calculate_gini(participation)
        balance_score = 1.0 - gini

        # For debugging
        print(f"Gini coefficient: {gini}")
        print(f"Balance score: {balance_score}")

        self.team_metrics['interaction_balance'].append({
            'time': segments[-1]['end'],
            'score': float(balance_score)
        })

        return balance_score

    def calculate_topic_consensus(self, buffer_results):
        """Calculate team consensus around topics"""
        if not buffer_results or 'topics' not in self.accumulated_results:
            return

        # Get current topics
        current_topics = self.accumulated_results['topics'].get('overall', [])

        for topic in current_topics:
            topic_name = topic['name']

            # Find utterances related to this topic
            topic_words = set(topic['top_words'])
            topic_sentiments = []

            for result in buffer_results:
                # Check if utterance contains topic keywords
                words = set(result['text'].lower().split())
                if words.intersection(topic_words):
                    topic_sentiments.append(result['sentiment']['polarity'])

            if topic_sentiments:
                # Calculate consensus score (1 - normalized variance)
                variance = np.var(topic_sentiments) if len(topic_sentiments) > 1 else 0
                consensus_score = 1.0 - min(1.0, variance * 2)  # Normalize variance

                if topic_name not in self.team_metrics['topic_consensus']:
                    self.team_metrics['topic_consensus'][topic_name] = []

                self.team_metrics['topic_consensus'][topic_name].append({
                    'time': buffer_results[-1]['end'],
                    'score': float(consensus_score)
                })

    def calculate_response_latency(self, segments):
        """Calculate average response time between speakers"""
        if len(segments) < 2:
            return

        latencies = []
        for i in range(1, len(segments)):
            prev_end = segments[i - 1]['end']
            curr_start = segments[i]['start']

            # Only count if it's a different speaker
            if segments[i]['speaker'] != segments[i - 1]['speaker']:
                latency = curr_start - prev_end
                if latency < 5.0:  # Only count responses within 5 seconds
                    latencies.append(latency)

        if latencies:
            avg_latency = np.mean(latencies)
            normalized_score = max(0.0,min(1.0, 1.0 - (avg_latency / 5.0)))

            self.team_metrics['response_latency'].append({
                'time': segments[-1]['end'],
                'score': float(normalized_score)
            })


    def calculate_sentiment_alignment(self, buffer_results):
        """Calculate sentiment alignment using existing Gini coefficient approach"""
        if not buffer_results:
            return 0.0

        # Group sentiments by speaker
        speaker_sentiments = defaultdict(list)
        for result in buffer_results:
            if result['text'].strip():
                speaker_sentiments[result['speaker']].append(
                    result['sentiment']['polarity']
                )

        # Need at least 2 speakers with utterances
        if len(speaker_sentiments) < 2:
            return 0.0

        # Calculate average sentiment per speaker
        avg_sentiments = {
            speaker: np.mean(sentiments)
            for speaker, sentiments in speaker_sentiments.items()
            if sentiments  # Only include speakers who spoke
        }

        if not avg_sentiments:
            return 0.0

        # Normalize sentiments to positive values for Gini calculation
        sentiments = np.array(list(avg_sentiments.values()))
        sentiments = sentiments + abs(min(sentiments)) + 0.01  # Ensure all positive

        # Calculate alignment as 1 - Gini coefficient
        gini = self.calculate_gini(sentiments)
        alignment_score = float(1.0 - gini)

        self.team_metrics['sentiment_alignment'].append({
            'time': buffer_results[-1]['end'],
            'score': max(0.0, min(1.0, alignment_score))
        })

        return alignment_score


    def update_results(self, buffer_results, segments):
        """Update accumulated results with buffer results"""
        # Update buffer count and text collections
        self.buffer_count += 1

        # Add new texts to cumulative collections
        new_texts = [result['text'] for result in buffer_results if len(result['text'].strip()) > 10]
        if new_texts:
            # Add to cumulative collection
            self.cumulative_texts.extend(new_texts)

            # Add to cumulative speaker collections
            for result in buffer_results:
                if len(result['text'].strip()) > 10:
                    speaker = result['speaker']
                    if speaker not in self.cumulative_speaker_texts:
                        self.cumulative_speaker_texts[speaker] = []
                    self.cumulative_speaker_texts[speaker].append(result['text'])

        # Calculate team metrics
        balance_score = self.calculate_team_interaction_balance(segments)
        self.calculate_topic_consensus(buffer_results)
        latency_score = self.calculate_response_latency(segments)
        alignment_score = self.calculate_sentiment_alignment(buffer_results)

        # Update accumulated team metrics
        metrics_data = {
            'interaction_balance': {
                'timeline': self.team_metrics['interaction_balance'],
                'current_score': balance_score,
                'average_score': np.mean([entry['score'] for entry in self.team_metrics['interaction_balance']]) if
                self.team_metrics['interaction_balance'] else 0.0
            },
            'topic_consensus': {
                'per_topic': self.team_metrics['topic_consensus'],
                'current_score': list(self.team_metrics['topic_consensus'].values())[-1][-1]['score'] if
                self.team_metrics['topic_consensus'] else 0.0,
                'average_score': np.mean(
                    [entry['score'] for topic in self.team_metrics['topic_consensus'].values() for entry in topic]) if
                self.team_metrics['topic_consensus'] else 0.0
            },
            'response_latency': {
                'timeline': self.team_metrics['response_latency'],
                'current_score': latency_score,
                'average_score': np.mean([entry['score'] for entry in self.team_metrics['response_latency']]) if
                self.team_metrics['response_latency'] else 0.0
            },
            'sentiment_alignment': {
                'timeline': self.team_metrics['sentiment_alignment'],
                'current_score': alignment_score,
                'average_score': np.mean([entry['score'] for entry in self.team_metrics['sentiment_alignment']]) if
                self.team_metrics['sentiment_alignment'] else 0.0
            },
        }


        self.accumulated_results['team_metrics'] = metrics_data

        # Update transcript
        self.accumulated_results['transcript'].extend(buffer_results)

        # Update speaker statistics
        speaker_durations = {}
        total_duration = 0

        # Calculate durations for current segments
        for segment in segments:
            speaker = segment['speaker']
            duration = segment['duration']

            if speaker != "SPEAKER_UNKNOWN" and len(
                    [result for result in buffer_results if result['speaker'] == speaker]) > 0:
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
                total_duration += duration

        # Update speaker statistics
        for speaker, duration in speaker_durations.items():
            if speaker not in self.accumulated_results['speaker_statistics']:
                self.accumulated_results['speaker_statistics'][speaker] = {
                    'speaking_time': 0,
                    'sentiment': {'polarity_sum': 0, 'subjectivity_sum': 0, 'count': 0}
                }

            self.accumulated_results['speaker_statistics'][speaker]['speaking_time'] += duration

        # Update sentiment statistics
        for result in buffer_results:
            speaker = result['speaker']
            sentiment = result['sentiment']
            if speaker != "SPEAKER_UNKNOWN" and len(
                    [r for r in buffer_results if r['speaker'] == speaker and r['text'].strip()]) > 0:
                stats = self.accumulated_results['speaker_statistics'][speaker]
                stats['sentiment']['polarity_sum'] += sentiment['polarity']
                stats['sentiment']['subjectivity_sum'] += sentiment['subjectivity']
                stats['sentiment']['count'] += 1

        # Update meeting duration
        self.accumulated_results['meeting_duration'] = max(
            self.accumulated_results['meeting_duration'],
            max(segment['end'] for segment in segments)
        )
        if self.buffer_count % self.update_frequency == 0:
            print(f"\nUpdating topics after {self.buffer_count} buffers")
            print(f"Total accumulated sentences: {len(self.cumulative_texts)}")

            try:
                # Create and fit vectorizer on ALL texts (overall + speaker texts)
                all_texts = self.cumulative_texts.copy()
                for speaker_texts in self.cumulative_speaker_texts.values():
                    all_texts.extend(speaker_texts)

                if len(all_texts) < 5:  # Minimum threshold for any analysis
                    print("Not enough texts for topic modeling")
                    return

                vectorizer = CountVectorizer(**self.vectorizer_config)
                vectorizer.fit(all_texts)  # Fit once on all texts

                # Perform topic modeling on overall texts using transform
                doc_term_matrix = vectorizer.transform(self.cumulative_texts)


                # Create and fit LDA model for overall topics
                overall_lda = LatentDirichletAllocation(
                    n_components=max(1,min(1, len(self.cumulative_texts) // 10)),
                    random_state=42,
                    max_iter=10,
                    learning_method='online',
                    n_jobs=-1
                )

                overall_doc_topics = overall_lda.fit_transform(doc_term_matrix)
                overall_topics = self.extract_topics(overall_lda, vectorizer, overall_doc_topics)

                self.overall_model = overall_lda
                self.overall_topics = overall_topics

                # Initialize or update topics in accumulated_results
                if 'topics' not in self.accumulated_results:
                    self.accumulated_results['topics'] = {
                        'overall': [],
                        'per_speaker': {}
                    }

                self.accumulated_results['topics']['overall'] = overall_topics
                print(f"Updated overall topics: {len(overall_topics)} topics found")

                # Process each speaker's cumulative texts using same vectorizer
                for speaker, texts in self.cumulative_speaker_texts.items():
                    if len(texts) >= 5:  # Minimum threshold for speaker topics
                        try:
                            # Transform speaker texts using same vectorizer
                            speaker_doc_term_matrix = vectorizer.transform(texts)

                            # Create and fit speaker LDA model
                            speaker_lda = LatentDirichletAllocation(
                                n_components=max(1,min(1, len(texts) // 5)),
                                random_state=42,
                                max_iter=10,
                                learning_method='online',
                                n_jobs=-1
                            )

                            speaker_doc_topics = speaker_lda.fit_transform(speaker_doc_term_matrix)
                            speaker_topics = self.extract_topics(speaker_lda, vectorizer, speaker_doc_topics)

                            # Calculate similarities
                            similarities = []
                            similarity_matrix = cosine_similarity(
                                speaker_lda.components_,
                                overall_lda.components_
                            )

                            for i, speaker_topic in enumerate(speaker_topics):
                                best_match_idx = np.argmax(similarity_matrix[i])
                                similarity_score = float(similarity_matrix[i][best_match_idx])

                                similarities.append({
                                    'speaker_topic': speaker_topic['name'],
                                    'overall_topic': overall_topics[best_match_idx]['name'],
                                    'similarity_score': similarity_score
                                })

                            self.accumulated_results['topics']['per_speaker'][speaker] = {
                                'topics': speaker_topics,
                                'topic_similarities': similarities
                            }

                        except Exception as e:
                            print(f"Warning: Error processing speaker {speaker}: {str(e)}")
                            continue

            except Exception as e:
                print(f"Warning: Error in topic modeling: {str(e)}")
                import traceback
                traceback.print_exc()

        # Perform topic modeling if enough buffers have been processed
        if self.buffer_count % self.update_frequency == 0:
            self.perform_periodic_topic_modeling()

        # Update knowledge tracking
        self.calculate_knowledge_tracking(buffer_results)

    def save_results(self):
        """Save current results to JSON file"""
        # Calculate percentages and averages
        total_time = sum(s['speaking_time'] for s in self.accumulated_results['speaker_statistics'].values())

        # Create set of significant speakers
        significant_speakers = {
            speaker: stats for speaker, stats in self.accumulated_results['speaker_statistics'].items()
            if self.is_significant_speaker(speaker, stats, total_time)
        }

        output_results = {'meeting_duration': self.accumulated_results['meeting_duration'], 'speaker_statistics': {},
                          'transcript': self.accumulated_results['transcript'], 'topics': {
                'overall': self.overall_topics,
                'per_speaker': {}
            }, 'turn_taking': {
                'window_entropies': self.accumulated_results['turn_taking']['window_entropies'],
                'entropy_peaks': self.accumulated_results['turn_taking']['entropy_peaks'],
                'entropy_statistics': {
                    'mean_entropy': np.mean(
                        [w['entropy'] for w in self.accumulated_results['turn_taking']['window_entropies']]) if
                    self.accumulated_results['turn_taking']['window_entropies'] else 0,
                    'peak_count': len(self.accumulated_results['turn_taking']['entropy_peaks']),
                    'peak_times': [peak['time'] for peak in self.accumulated_results['turn_taking']['entropy_peaks']]
                }
            }, 'team_metrics': {
                'interaction_balance': {
                    'timeline': self.team_metrics['interaction_balance'],
                    'current_score': self.team_metrics['interaction_balance'][-1]['score'] if self.team_metrics[
                        'interaction_balance'] else 0.0,
                    'average_score': np.mean([entry['score'] for entry in self.team_metrics['interaction_balance']]) if
                    self.team_metrics['interaction_balance'] else 0.0

                },
                'topic_consensus': {
                    'per_topic': self.team_metrics['topic_consensus'],
                    'average_score': np.mean([entry['score'] for topic in self.team_metrics['topic_consensus'].values()
                                              for entry in topic]) if self.team_metrics['topic_consensus'] else 0.0,
                    'current_score': list(self.team_metrics['topic_consensus'].values())[-1][-1]['score'] if
                    self.team_metrics['topic_consensus'] else 0.0

                },
                'response_latency': {
                    'timeline': self.team_metrics['response_latency'],
                    'current_score': self.team_metrics['response_latency'][-1]['score'] if self.team_metrics[
                        'response_latency'] else 0.0,
                    'average_score': np.mean([entry['score'] for entry in self.team_metrics['response_latency']]) if
                    self.team_metrics['response_latency'] else 0.0
                },
                'sentiment_alignment': {
                    'timeline': self.team_metrics['sentiment_alignment'],
                    'current_score': self.team_metrics['sentiment_alignment'][-1]['score'] if self.team_metrics[
                        'sentiment_alignment'] else 0.0,
                    'average_score': np.mean([entry['score'] for entry in self.team_metrics['sentiment_alignment']]) if
                    self.team_metrics['sentiment_alignment'] else 0.0
                },
            }, 'filler_analysis': {
                'overall': {
                    'total_count': self.accumulated_results['filler_analysis']['overall_stats']['total_count'],
                    'rate_per_minute': (
                        self.accumulated_results['filler_analysis']['overall_stats']['total_count'] /
                        (self.accumulated_results['meeting_duration'] / 60)
                        if self.accumulated_results['meeting_duration'] > 0 else 0
                    ),
                    'per_minute_timeline': [
                        {
                            'minute': i,
                            'count': count
                        } for i, count in enumerate(
                            self.accumulated_results['filler_analysis']['overall_stats']['per_minute']
                        )
                    ],
                    'by_category': {
                        category: {
                            'count': count,
                            'percentage': (
                                count / self.accumulated_results['filler_analysis']['overall_stats'][
                                    'total_count'] * 100
                                if self.accumulated_results['filler_analysis']['overall_stats'][
                                       'total_count'] > 0 else 0
                            )
                        }
                        for category, count in
                        self.accumulated_results['filler_analysis']['overall_stats']['by_category'].items()
                    },
                    'most_common': [
                        {
                            'filler': filler,
                            'count': count,
                            'percentage': (
                                count / self.accumulated_results['filler_analysis']['overall_stats'][
                                    'total_count'] * 100
                                if self.accumulated_results['filler_analysis']['overall_stats'][
                                       'total_count'] > 0 else 0
                            )
                        }
                        for filler, count in sorted(
                            self.accumulated_results['filler_analysis']['overall_stats']['most_common'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:10]  # Top 10 most common fillers
                    ]
                },
                'per_speaker': {}

            },
                          # In your output_results dictionary
            'recent_topic_alignment': {  # Added this section
                'timeline': {},
                'current_scores': {},
                'summary': {
                    'per_speaker': {},
                    'team_stats': {}
                }
        }

        }
        # Add recent topic alignment data
        if 'recent_topic_alignment' in self.accumulated_results:
            topic_data = self.accumulated_results['recent_topic_alignment']

            # Copy timeline and current scores with filtering
            output_results['recent_topic_alignment']['timeline'] = {
                speaker: timeline for speaker, timeline in topic_data['timeline'].items()
                if speaker in significant_speakers
            }
            output_results['recent_topic_alignment']['current_scores'] = {
                speaker: scores for speaker, scores in topic_data['current_scores'].items()
                if speaker in significant_speakers
            }

            # Generate summary statistics
            summary = {}
            per_speaker_summary = {}
            all_alignment_scores = []
            team_warnings = {
                'total': 0,
                'mild': 0,
                'moderate': 0,
                'severe': 0
            }

            # Only process significant speakers
            for speaker, timeline in topic_data['timeline'].items():
                if speaker in significant_speakers and timeline:  # Add significant speaker check
                    # Rest of your existing code remains exactly the same
                    alignment_scores = [entry['alignment_score'] for entry in timeline]
                    all_alignment_scores.extend(alignment_scores)
                    warnings = [entry for entry in timeline if entry.get('warning_info')]

                    speaker_warnings = {
                        'mild': len([w for w in warnings if w['warning_info']['severity'] == 'mild']),
                        'moderate': len([w for w in warnings if w['warning_info']['severity'] == 'moderate']),
                        'severe': len([w for w in warnings if w['warning_info']['severity'] == 'severe'])
                    }

                    team_warnings['mild'] += speaker_warnings['mild']
                    team_warnings['moderate'] += speaker_warnings['moderate']
                    team_warnings['severe'] += speaker_warnings['severe']
                    team_warnings['total'] += sum(speaker_warnings.values())

                    per_speaker_summary[speaker] = {
                        'average_alignment': float(np.mean(alignment_scores)),
                        'total_warnings': len(warnings),
                        'warning_breakdown': speaker_warnings,
                        'max_consecutive_warnings': max(entry['consecutive_warnings'] for entry in timeline)
                    }

            # Team stats calculation remains the same since it's already using filtered data
            team_stats = {
                'average_team_alignment': float(np.mean(all_alignment_scores)) if all_alignment_scores else 0.0,
                'alignment_std': float(np.std(all_alignment_scores)) if all_alignment_scores else 0.0,
                'warning_counts': team_warnings,
                'warning_percentages': {
                    'mild': (team_warnings['mild'] / team_warnings['total'] * 100) if team_warnings['total'] > 0 else 0,
                    'moderate': (team_warnings['moderate'] / team_warnings['total'] * 100) if team_warnings[
                                                                                                  'total'] > 0 else 0,
                    'severe': (team_warnings['severe'] / team_warnings['total'] * 100) if team_warnings[
                                                                                              'total'] > 0 else 0
                },
                'most_off_topic_speaker': max(
                    per_speaker_summary.items(),
                    key=lambda x: x[1]['max_consecutive_warnings']
                )[0] if per_speaker_summary else None
            }

            output_results['recent_topic_alignment']['summary']['per_speaker'] = per_speaker_summary
            output_results['recent_topic_alignment']['summary']['team_stats'] = team_stats

        # Add per-speaker filler analysis - only for significant speakers
        output_results['filler_analysis']['per_speaker'] = {}
        for speaker, stats in self.accumulated_results['filler_analysis']['speaker_stats'].items():
            if speaker in significant_speakers:
                speaker_duration = self.accumulated_results['speaker_statistics'][speaker]['speaking_time']
                speaker_duration_minutes = speaker_duration / 60 if speaker_duration > 0 else 0

                output_results['filler_analysis']['per_speaker'][speaker] = {
                    'total_count': stats['total_count'],
                    'rate_per_minute': (
                        stats['total_count'] / speaker_duration_minutes
                        if speaker_duration_minutes > 0 else 0
                    ),
                    'per_minute_timeline': [
                        {
                            'minute': i,
                            'count': count
                        } for i, count in enumerate(stats['per_minute'])
                    ],
                    'by_category': {
                        category: {
                            'count': count,
                            'percentage': (
                                count / stats['total_count'] * 100
                                if stats['total_count'] > 0 else 0
                            )
                        }
                        for category, count in stats['by_category'].items()
                    },
                    'most_common': [
                        {
                            'filler': filler,
                            'count': count,
                            'percentage': (
                                count / stats['total_count'] * 100
                                if stats['total_count'] > 0 else 0
                            )
                        }
                        for filler, count in sorted(
                            stats['most_common'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:10]
                    ]
                }

        # Add knowledge tracking information if available
        if 'knowledge_tracking' in self.accumulated_results:
            output_results['knowledge_tracking'] = {
                'group_centroid': self.accumulated_results['knowledge_tracking']['group_centroid'],
                'team_coherence': {
                    'timeline': self.accumulated_results['knowledge_tracking']['team_coherence']['values'],
                    'statistics': {
                        'convergence_rate': float(
                            len(self.accumulated_results['knowledge_tracking']['team_coherence']['values'])
                        ) if self.accumulated_results['knowledge_tracking']['team_coherence']['values'] else 0
                    }
                },
                'speaker_tracking': {}
            }

            # Process speaker tracking data - only for significant speakers
            for speaker, tracking in self.accumulated_results['knowledge_tracking']['speaker_tracking'].items():
                if speaker in significant_speakers:  # Add this check
                    coherence_values = [v['coherence'] for v in tracking['coherence_values']]
                    output_results['knowledge_tracking']['speaker_tracking'][speaker] = {
                        'coherence_timeline': tracking['coherence_values'],
                        'statistics': {
                            'average_coherence': float(np.mean(coherence_values)) if coherence_values else 0,
                            'max_coherence': float(np.max(coherence_values)) if coherence_values else 0,
                            'min_coherence': float(np.min(coherence_values)) if coherence_values else 0,
                            'final_coherence': float(coherence_values[-1]) if coherence_values else 0,
                            'convergence_rate': float(
                                (coherence_values[-1] - coherence_values[0]) / len(coherence_values)
                            ) if len(coherence_values) > 1 else 0
                        }
                    }

        # Continue with existing speaker statistics
        for speaker, stats in significant_speakers.items():
            sentiment_count = stats['sentiment']['count']
            output_results['speaker_statistics'][speaker] = {
                'speaking_time_percentage': (stats['speaking_time'] / total_time * 100) if total_time > 0 else 0,
                'sentiment': {
                    'average_polarity': stats['sentiment'][
                                            'polarity_sum'] / sentiment_count if sentiment_count > 0 else 0,
                    'average_subjectivity': stats['sentiment'][
                                                'subjectivity_sum'] / sentiment_count if sentiment_count > 0 else 0
                }
            }

        if 'topics' in self.accumulated_results and 'per_speaker' in self.accumulated_results['topics']:
            for speaker in significant_speakers:
                speaker_topics = self.accumulated_results['topics']['per_speaker'].get(speaker, {})
                if speaker_topics:
                    output_results['topics']['per_speaker'][speaker] = {
                        'topics': speaker_topics.get('topics', []),
                        'topic_similarities': speaker_topics.get('topic_similarities', [])
                    }

        # Save to file
        video_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   '_static', 'DemoApp', f'meeting_analysis_{video_filename}.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_results, f, indent=4, ensure_ascii=False)

    def analyze_meeting(self):
        """Perform live meeting analysis"""
        try:
            # Extract full audio first
            self.extract_audio()

            print(f"Starting live analysis of {self.total_duration:.2f} seconds of content...")

            # Process video in buffers
            while self.current_position < self.total_duration and not self.stop_processing:
                buffer_end = min(self.current_position + self.buffer_size, self.total_duration)
                print(f"\nProcessing segment {self.current_position:.1f}s to {buffer_end:.1f}s")

                # Process current buffer
                success = self.process_buffer(self.current_position, buffer_end)
                if not success:
                    print(f"Error processing buffer at {self.current_position:.1f}s")

                # Advance position by step size
                self.current_position += self.step_size

                # Simulate real-time processing
                time.sleep(5.5)

            print("\nAnalysis complete!")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            # Cleanup
            if os.path.exists(self.audio_path):
                os.remove(self.audio_path)


def main():
    video_path = input("Enter the path to your video file: ")
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        return

    try:
        analyzer = LiveMeetingAnalyzer(video_path)
        analyzer.analyze_meeting()
    except KeyboardInterrupt:
        print("\nAnalysis stopped by user")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()