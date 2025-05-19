# modules/extractor.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import difflib
from collections import Counter
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import spacy
import pdfplumber
import nltk
nltk.download('punkt')
from modules.model_cache import model_cache

def sent_tokenize1(text):
    # Basic sentence splitter using punctuation (approximate)
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def clean_name(name):
    return re.sub(r'[^a-zA-Z\s\-]', '', name).strip()


class BookFeatureExtractor:
    """
    Extracts emotional and thematic features from books for recommendation systems.
    Implements the methodology described in the project documentation.
    """

    def __init__(self, emotion_lexicon_path=None, model_path=None):
        """
        Initialize the feature extractor with necessary resources.

        Args:
            emotion_lexicon_path (str): Path to the emotion lexicon file
            model_path (str): Path to pre-trained or fine-tuned BERT model
        """
        self.device = model_cache.device


        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Load emotion lexicon (sample format: word,emotion1,emotion2)
        self.emotion_lexicon = self._load_emotion_lexicon(emotion_lexicon_path)

        self.tokenizer = model_cache.get_tokenizer("joeddav/distilbert-base-uncased-go-emotions-student")
        self.model = model_cache.get_transformer_model("joeddav/distilbert-base-uncased-go-emotions-student")
        self.nlp = model_cache.get_spacy_model("en_core_web_sm")

        print(f"[DEBUG] Model loaded from {model_path if model_path else 'default'}")

        # Define Plutchik's 8 emotions
        self.plutchik_emotions = [
            "joy", "trust", "fear", "surprise",
            "sadness", "disgust", "anger", "anticipation"
        ]

        # Official GoEmotions labels
        emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]

        self.emotion_labels = emotion_labels


    def _load_emotion_lexicon(self, lexicon_path):
        """
        Load emotion lexicon from file.

        Args:
            lexicon_path (str): Path to lexicon file

        Returns:
            dict: Word to emotions mapping
        """
        if not lexicon_path:
            # Return a small sample lexicon for testing
            return {
                "happy": ["joy"], "sad": ["sadness"],
                "angry": ["anger"], "afraid": ["fear"]
            }

        lexicon = {}
        with open(lexicon_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                word = parts[0]
                emotions = parts[1:]
                lexicon[word] = emotions

        return lexicon

    def clean_gutenberg_text(self, text):
        """
        Remove Project Gutenberg's header and footer boilerplate, accounting for variations.

        Args:
            text (str): Raw book text

        Returns:
            str: Cleaned book text
        """
        import re

        # Define robust patterns to detect start and end
        start_pattern = re.compile(r'\*+\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*', re.IGNORECASE)
        end_pattern = re.compile(r'\*+\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*', re.IGNORECASE)

        lines = text.splitlines()
        start_idx, end_idx = 0, len(lines)

        for i, line in enumerate(lines):
            if start_pattern.search(line):
                start_idx = i + 1  # skip the matched line
                break

        for i in range(len(lines)-1, -1, -1):
            if end_pattern.search(lines[i]):
                end_idx = i
                break

        return "\n".join(lines[start_idx:end_idx]).strip()

    def preprocess_text(self, text):
        """
        Clean and preprocess text for analysis.

        Args:
            text (str): Raw text content

        Returns:
            list: Processed tokens
            list: Processed sentences
        """
        # Convert to lowercase
        text = text.lower()

        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Tokenize into sentences
        sentences = sent_tokenize1(text)

        # Tokenize and lemmatize words, remove stopwords
        processed_tokens = []
        for sentence in sentences:
            tokens = [token.text for token in self.nlp(sentence)]
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                     if token not in self.stop_words]
            processed_tokens.extend(tokens)

        return processed_tokens, sentences

    def segment_by_chapter(self, text):
        """
        Split book text into chapters using multiple detection methods.
        Falls back to paragraph-based segmentation if no clear chapter structure is found.
        
        Args:
            text (str): Book text
            
        Returns:
            list: List of chapter texts
        """
        import re

        # Normalize line endings and clean up whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize multiple newlines
        
        # Try multiple chapter heading patterns
        patterns = [
            # Roman numerals with optional "Chapter" and title
            r'(?:^|\n)(?:Chapter|CHAPTER)?\s*([IVXLCDM]+)(?:\.|\s+)([A-Z][^\n]+)',
            # Standard "Chapter X" patterns
            r'(?:^|\n)(?:Chapter|CHAPTER)\s+(?:[A-Z]+|\d+|[IVXLCDM]+)\b',
            # Roman numerals with optional "Chapter"
            r'(?:^|\n)(?:[IVXLCDM]+)(?:\s*\.|\s*$|\s*[A-Z])',
            # Numbered sections
            r'(?:^|\n)(?:\d+\.\s*[A-Z][^\n]+)',
            # Common book section markers
            r'(?:^|\n)(?:BOOK|PART|SECTION)\s+(?:[A-Z]+|\d+|[IVXLCDM]+)\b',
            # All caps titles that look like chapter headings
            r'(?:^|\n)(?:[A-Z][A-Z\s]{5,})(?:\n|$)'
        ]
        
        # Try each pattern
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if len(matches) >= 3:  # Require at least 3 matches to consider it a valid chapter structure
                chapters = []
                for i in range(len(matches)):
                    start = matches[i].start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                    chapter_text = text[start:end].strip()
                    if len(chapter_text) > 100:  # Skip very short chapters
                        chapters.append(chapter_text)
                
                if chapters:
                    print(f"[INFO] Found {len(chapters)} chapters using pattern: {pattern}")
                    return chapters
        
        # Fallback 1: Try to split by double newlines if they seem to indicate chapters
        paragraphs = text.split('\n\n')
        if len(paragraphs) >= 10:  # Only use if we have enough paragraphs
            avg_len = sum(len(p) for p in paragraphs) / len(paragraphs)
            if avg_len > 500:  # If paragraphs are long enough to be chapters
                print("[INFO] Using paragraph-based chapter detection")
                return paragraphs
        
        # Fallback 2: Split into roughly equal chunks based on sentence count
        print("[INFO] Using sentence-based chapter detection")
        sentences = sent_tokenize1(text)
        target_chapters = max(5, min(20, len(sentences) // 50))  # Aim for 5-20 chapters
        chunk_size = len(sentences) // target_chapters
        
        chapters = []
        for i in range(0, len(sentences), chunk_size):
            chapter_text = ' '.join(sentences[i:i + chunk_size])
            if len(chapter_text) > 100:  # Skip very short chapters
                chapters.append(chapter_text)
        
        return chapters


    def extract_lexicon_based_emotions(self, tokens):
        """
        Extract emotions based on lexicon matching.

        Args:
            tokens (list): Processed word tokens

        Returns:
            dict: Emotion counts and normalized scores
        """
        emotion_counts = {emotion: 0 for emotion in self.emotion_labels}

        for token in tokens:
            if token in self.emotion_lexicon:
                for emotion in self.emotion_lexicon[token]:
                    if emotion in emotion_counts:
                        emotion_counts[emotion] += 1

        # Normalize counts
        total = sum(emotion_counts.values())
        emotion_scores = emotion_counts.copy()

        if total > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] = emotion_scores[emotion] / total

        return {
            "counts": emotion_counts,
            "scores": emotion_scores
        }

    def chunk_sentences(self, sentences, chunk_size=5):
      """Yield successive chunks from list of sentences."""
      for i in range(0, len(sentences), chunk_size):
          yield sentences[i:i + chunk_size]

    def extract_bert_emotions(self, text, batch_size=4):
        """
        Use fine-tuned BERT to classify text into Plutchik's 8 emotions.

        Args:
            text (str): Text to analyze
            batch_size (int): Batch size for processing

        Returns:
            dict: Emotion classification results
        """
        # This is a simplified implementation assuming the model is fine-tuned
        # for multi-label classification of Plutchik's emotions

        sentences = sent_tokenize1(text)
        results = {emotion: 0 for emotion in self.emotion_labels}
        all_probs = []


        # Process in batches
        for batch in self.chunk_sentences(sentences, chunk_size=5):
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )

            # Manually move tensors to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}


            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
        if not all_probs:
            return {label: 0.0 for label in self.emotion_labels}

        all_probs_tensor = torch.cat(all_probs, dim=0)
        top_k = torch.topk(all_probs_tensor, k=min(3, all_probs_tensor.size(0)), dim=0)[0]
        avg_probs = top_k.mean(dim=0).numpy()

        num_output_labels = avg_probs.shape[0]
        used_labels = self.emotion_labels[:num_output_labels]
        emotion_scores = {label: float(avg_probs[i]) for i, label in enumerate(used_labels)}

        #print(f"[DEBUG] Emotion probabilities shape: {avg_probs.shape}")  # Should be (28,)
        #print(f"[DEBUG] Top 5 emotions: {sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:5]}")


        return emotion_scores


    def extract_polarity_with_textblob(self, sentences):
        """
        Extract sentiment polarity using TextBlob.

        Args:
            sentences (list): List of sentences

        Returns:
            list: Polarity scores for each sentence
        """
        polarity_scores = []

        for sentence in sentences:
            analysis = TextBlob(sentence)
            polarity_scores.append(analysis.sentiment.polarity)

        return polarity_scores

    def extract_character_emotions(self, text, max_paragraphs_per_character=12):
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        #print(f"[DEBUG] Paragraphs available: {len(paragraphs)}")

        titles = {'mr', 'mrs', 'miss', 'ms', 'dr', 'professor', 'prof', 'sir', 'madam', 'lady', 'lord'}
        character_mentions = {}
        character_paragraphs = {}

        for paragraph in paragraphs:
            words = paragraph.split()  # Keep case
            potential_names = set()

            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 1:
                    name_parts = [word]
                    j = i + 1
                    while j < len(words) and words[j][0].isupper():
                        name_parts.append(words[j])
                        j += 1
                    name = ' '.join(name_parts)
                    potential_names.add(name)

            clean_names = set()
            common_false_positives = {
                'the', 'and', 'but', 'he', 'she', 'they', 'you', 'i', 'we', 'it',
                'him', 'her', 'them', 'his', 'hers', 'their', 'your', 'my', 'our'
            }

            for name in potential_names:
                parts = name.split()
                if parts[0].lower() in titles:
                    parts = parts[1:]
                cleaned = clean_name(' '.join(parts))
                if (
                    cleaned
                    and len(cleaned.split()) <= 4
                    and len(cleaned) > 2
                    and cleaned.lower() not in common_false_positives
                    and cleaned.isalpha()
                ):
                    clean_names.add(cleaned)

            #print(f"[DEBUG] Cleaned candidate names: {clean_names}")

            for name in clean_names:
                if name not in character_mentions:
                    character_mentions[name] = 0
                    character_paragraphs[name] = []
                character_mentions[name] += 1
                character_paragraphs[name].append(paragraph)

        #print(f"[DEBUG] Character mention counts: {character_mentions}")

        character_scores = {}
        for name, mentions in character_mentions.items():
            if mentions >= 3:
                paras = character_paragraphs[name]
                score = mentions * (1 + len(set(tuple(p.split()[:10]) for p in paras)) / len(paras))
                character_scores[name] = score
        #print(f"[DEBUG] Detected character names: {list(character_mentions.keys())}")

        top_characters = sorted(character_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        #print(f"[DEBUG] Top characters: {top_characters}")

        character_emotions = {}
        for name, _ in top_characters:
            paras = character_paragraphs[name]
            selected = [p for p in paras if name.lower() in p.lower()][:max_paragraphs_per_character]
            context_text = " ".join(selected)
            #print(f"[DEBUG] Selected paragraphs for {name}: {len(selected)}")

            try:
                emotion_scores = self.extract_bert_emotions(context_text)     
                print(f"[DEBUG] Raw emotions for {name}: {emotion_scores}")
                emotion_scores = {k: v for k, v in emotion_scores.items() if v > 0.03}
                if emotion_scores:
                    character_emotions[name] = emotion_scores
            except Exception as e:
                print(f"[ERROR] Failed emotion analysis for {name}: {e}")

        return character_emotions


    def get_custom_stopwords_from_entities(self, text, nlp, top_n=5):
          doc = nlp(text)
          people = [ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"]
          most_common = [name for name, _ in Counter(people).most_common(top_n)]
          return set(most_common)


    def extract_tfidf_features(self, chapters):
        """
        Extract TF-IDF features from book chapters with improved robustness.
        Handles empty chapters, very short chapters, and improves feature selection.

        Args:
            chapters (list): List of book chapters

        Returns:
            array: TF-IDF matrix
            list: Feature names
        """
        try:
            # Filter out empty or very short chapters
            valid_chapters = [ch for ch in chapters if len(ch.strip()) > 100]
            if not valid_chapters:
                raise ValueError("No valid chapters found for analysis")
            
            if len(valid_chapters) < 2:
                # If only one chapter, duplicate it to allow vectorization
                valid_chapters = [valid_chapters[0], valid_chapters[0]]

            # Get common character names and custom stopwords
            full_text = " ".join(valid_chapters)
            custom_stopwords = self.get_custom_stopwords_from_entities(full_text, self.nlp)
            
            # Add common book-specific stopwords
            book_stopwords = {
                'said', 'says', 'say', 'saying', 'chapter', 'book', 'page',
                'read', 'reading', 'reader', 'author', 'title', 'edition',
                'copyright', 'published', 'publisher', 'year', 'volume'
            }
            custom_stopwords.update(book_stopwords)

            processed_chapters = []
            for chapter in valid_chapters:
                try:
                    # Lowercase and basic cleaning
                    chapter = chapter.lower()
                    chapter = re.sub(r'[^a-zA-Z\s]', ' ', chapter)
                    chapter = re.sub(r'\s+', ' ', chapter).strip()

                    # Tokenize with spaCy
                    doc = self.nlp(chapter)
                    
                    # More sophisticated token filtering
                    filtered_tokens = []
                    for token in doc:
                        # Skip if token is in stopwords or custom stopwords
                        if (token.text.lower() in self.stop_words or 
                            token.text.lower() in custom_stopwords or
                            len(token.text) < 2):  # Skip very short tokens
                            continue
                            
                        # Keep nouns, adjectives, and important verbs
                        if (token.pos_ in {"NOUN", "ADJ"} or 
                            (token.pos_ == "VERB" and token.lemma_ not in {"be", "have", "do"})):
                            
                            # Lemmatize and clean
                            lemma = self.lemmatizer.lemmatize(token.text.lower())
                            if len(lemma) > 2:  # Skip very short lemmas
                                filtered_tokens.append(lemma)

                    # Join tokens and add to processed chapters
                    if filtered_tokens:  # Only add if we have tokens
                        processed_chapters.append(" ".join(filtered_tokens))
                    
                except Exception as e:
                    print(f"Warning: Error processing chapter: {str(e)}")
                    continue

            if not processed_chapters:
                raise ValueError("No valid processed chapters after filtering")

            # Configure CountVectorizer with improved parameters
            vectorizer = CountVectorizer(
                max_df=0.95,      # Remove terms that appear in >95% of chapters
                min_df=1,         # Remove terms that appear in <2 chapters
                max_features=5000,# Limit vocabulary size
                stop_words='english',
                ngram_range=(1, 2),  # Use both unigrams and bigrams
                token_pattern=r'(?u)\b\w\w+\b',  # Better token pattern
                strip_accents='unicode',
                analyzer='word'
            )

            try:
                # Fit and transform
                tfidf_matrix = vectorizer.fit_transform(processed_chapters)
                
                # Get feature names and filter out any remaining problematic terms
                feature_names = vectorizer.get_feature_names_out()
                valid_features = [f for f in feature_names if len(f) > 2 and not f.isdigit()]
                
                # Reindex matrix to keep only valid features
                if len(valid_features) < len(feature_names):
                    feature_indices = [i for i, f in enumerate(feature_names) if f in valid_features]
                    tfidf_matrix = tfidf_matrix[:, feature_indices]
                    feature_names = valid_features

                return tfidf_matrix, feature_names

            except Exception as e:
                print(f"Error in vectorization: {str(e)}")
                # Fallback to simpler vectorization if the main one fails
                vectorizer = CountVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 1)
                )
                tfidf_matrix = vectorizer.fit_transform(processed_chapters)
                return tfidf_matrix, vectorizer.get_feature_names_out()

        except Exception as e:
            print(f"Critical error in TF-IDF extraction: {str(e)}")
            # Return minimal valid output to prevent pipeline failure
            dummy_matrix = np.zeros((len(chapters), 1))
            return dummy_matrix, ["unknown"]



    def extract_topics_with_lda(self, tfidf_matrix, feature_names, num_topics=10):
        """Extract topics using Latent Dirichlet Allocation."""
        # Additional book-specific stopwords
        book_stopwords = {
            'said', 'say', 'says', 'saying', 'chapter', 'book', 'read', 'reading',
            'page', 'pages', 'author', 'story', 'stories', 'novel', 'fiction',
            'character', 'characters', 'plot', 'scene', 'scenes', 'setting',
            'theme', 'themes', 'narrative', 'narrator', 'dialogue', 'dialog',
            'paragraph', 'paragraphs', 'sentence', 'sentences', 'word', 'words',
            'text', 'texts', 'write', 'writes', 'writing', 'written', 'wrote',
            'begin', 'begins', 'began', 'beginning', 'end', 'ends', 'ended', 'ending',
            'start', 'starts', 'started', 'starting', 'finish', 'finishes', 'finished',
            'finishing', 'continue', 'continues', 'continued', 'continuing',
            'stop', 'stops', 'stopped', 'stopping', 'pause', 'pauses', 'paused',
            'pausing', 'resume', 'resumes', 'resumed', 'resuming'
        }
        
        # Filter out stopwords from feature names
        valid_indices = [i for i, word in enumerate(feature_names) 
                        if word.lower() not in book_stopwords and len(word) > 2]
        
        if not valid_indices:
            return {}, np.zeros((tfidf_matrix.shape[0], 1))
        
        filtered_matrix = tfidf_matrix[:, valid_indices]
        filtered_features = [feature_names[i] for i in valid_indices]

        num_topics = min(12, max(3, len(filtered_features) // 20))


        # Create and fit LDA model with better parameters
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=30,
            learning_method='online',
            learning_offset=20.0,
            random_state=42,
            n_jobs=-1
        )

                
        try:
            lda_output = lda.fit_transform(filtered_matrix)
            row_sums = lda_output.sum(axis=1)
            print("Row sums (should be ~1):", row_sums[:5])
            
            # Get topic words
            topics = {}
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[:-11:-1]
                top_words = [filtered_features[i] for i in top_indices]
                top_weights = [topic[i] for i in top_indices]
                relevance = np.sum(top_weights[:5]) / (np.sum(top_weights) + 1e-8)

                if len(set(top_words)) < 5 or relevance < 0.2:
                    continue
                
                topics[f"topic_{topic_idx+1}"] = top_words

            return topics, lda_output

        except Exception as e:
            print(f"[ERROR] LDA failed: {e}")
            return {}, np.zeros((filtered_matrix.shape[0], 1))

    def create_sentiment_trajectory(self, chapters):
        """
        Create a sentiment trajectory for the book.

        Args:
            chapters (list): List of book chapters

        Returns:
            dict: Sentiment trajectory data
        """
        trajectory = {
            "polarity": [],
            "emotions": {emotion: [] for emotion in self.emotion_labels},
            "chapter_boundaries": []
        }

        current_position = 0

        for chapter in chapters:
            tokens, sentences = self.preprocess_text(chapter)

            # Get polarity
            polarity_scores = self.extract_polarity_with_textblob(sentences)
            avg_polarity = sum(polarity_scores) / len(polarity_scores) if polarity_scores else 0
            trajectory["polarity"].append(avg_polarity)

            # Get emotions
            #emotions = self.extract_lexicon_based_emotions(tokens)
            chapter_text = " ".join(sentences)
            emotions = self.extract_bert_emotions(chapter_text)

            for emotion in self.emotion_labels:
                #trajectory["emotions"][emotion].append(emotions["scores"].get(emotion, 0))
                trajectory["emotions"][emotion].append(emotions.get(emotion, 0))

            # Track chapter boundary
            current_position += len(sentences)
            trajectory["chapter_boundaries"].append(current_position)

        return trajectory

    def apply_rolling_average(self, trajectory, window=3):
        """
        Apply rolling average smoothing to the sentiment trajectory.

        Args:
            trajectory (dict): Sentiment trajectory
            window (int): Window size for rolling average

        Returns:
            dict: Smoothed trajectory
        """
        smoothed = trajectory.copy()

        # Smooth polarity
        polarity = np.array(trajectory["polarity"])
        smoothed["polarity"] = np.convolve(polarity, np.ones(window)/window, mode='valid').tolist()

        # Smooth emotions
        for emotion in self.emotion_labels:
            values = np.array(trajectory["emotions"][emotion])
            smoothed["emotions"][emotion] = np.convolve(values, np.ones(window)/window, mode='valid').tolist()

        # Adjust chapter boundaries
        offset = (window - 1) // 2
        smoothed["chapter_boundaries"] = trajectory["chapter_boundaries"][offset:len(smoothed["polarity"])+offset]

        return smoothed
    from scipy.ndimage import gaussian_filter1d

    def apply_gaussian_smoothing(self, trajectory, sigma=1):
        """
        Apply Gaussian smoothing to the sentiment trajectory.

        Args:
            trajectory (dict): Sentiment trajectory
            sigma (int): Standard deviation for Gaussian kernel

        Returns:
            dict: Smoothed trajectory
        """
        smoothed = {
            "polarity": [],
            "emotions": {emotion: [] for emotion in self.emotion_labels},
            "chapter_boundaries": trajectory["chapter_boundaries"]  # optional: leave as-is
        }

        # Smooth polarity
        from scipy.ndimage import gaussian_filter1d
        polarity_array = np.array(trajectory["polarity"])
        smoothed["polarity"] = gaussian_filter1d(polarity_array, sigma=sigma).tolist()

        # Smooth emotions
        for emotion in self.emotion_labels:
            values = np.array(trajectory["emotions"][emotion])
            smoothed["emotions"][emotion] = gaussian_filter1d(values, sigma=sigma).tolist()

        return smoothed
    
    def infer_genre_from_profile(self, profile, title=None):
        """
        Infer book genre using analysis of topics, emotions, characters, and title.

        Args:
            profile (dict): Book profile with emotional and thematic features
            title (str): Optional book title

        Returns:
            str: Inferred genre (with fallback and confidence check)
        """
        import collections
        from difflib import SequenceMatcher

        # === Emotion remapping (GoEmotions -> Plutchik-like) ===
        emotion_map = {
            'admiration': 'joy', 'amusement': 'joy', 'approval': 'trust',
            'caring': 'love', 'confusion': 'surprise', 'curiosity': 'anticipation',
            'desire': 'anticipation', 'disappointment': 'sadness', 'disapproval': 'disgust',
            'disgust': 'disgust', 'excitement': 'joy', 'fear': 'fear', 'gratitude': 'joy',
            'grief': 'sadness', 'joy': 'joy', 'love': 'love', 'nervousness': 'fear',
            'optimism': 'anticipation', 'pride': 'joy', 'realization': 'surprise',
            'relief': 'joy', 'remorse': 'sadness', 'sadness': 'sadness',
            'surprise': 'surprise'
        }

        # === Genre definitions ===
        genre_patterns = {
            "romance": {
                "keywords": {"love", "heart", "marriage", "desire", "romance", "passion"},
                "emotions": {"love", "joy", "desire"},
                "title_patterns": ["love", "romance", "heart", "affair"],
                "weight": 1.5
            },
            "mystery": {
                "keywords": {"murder", "detective", "clue", "crime", "investigation"},
                "emotions": {"fear", "surprise", "anticipation"},
                "title_patterns": ["mystery", "case", "murder"],
                "weight": 1.5
            },
            "fantasy": {
                "keywords": {"magic", "dragon", "kingdom", "quest", "wizard"},
                "emotions": {"surprise", "anticipation", "joy", "fear"},
                "title_patterns": ["magic", "dragon", "kingdom", "fantasy"],
                "weight": 1.5
            },
            "science fiction": {
                "keywords": {"space", "robot", "future", "alien", "galaxy"},
                "emotions": {"anticipation", "curiosity", "fear"},
                "title_patterns": ["space", "future", "robot"],
                "weight": 1.4
            },
            "historical fiction": {
                "keywords": {"war", "kingdom", "castle", "revolution", "empire"},
                "emotions": {"pride", "remorse", "admiration"},
                "title_patterns": ["history", "century", "era", "ancient"],
                "weight": 1.3
            },
            "adventure": {
                "keywords": {"journey", "expedition", "treasure", "mission"},
                "emotions": {"joy", "excitement", "anticipation", "fear"},
                "title_patterns": ["adventure", "quest", "exploration"],
                "weight": 1.3
            },
            "horror": {
                "keywords": {"ghost", "haunted", "monster", "terror", "nightmare"},
                "emotions": {"fear", "disgust"},
                "title_patterns": ["horror", "terror", "night"],
                "weight": 1.4
            },
            "literary fiction": {
                "keywords": {"consciousness", "truth", "identity", "existence"},
                "emotions": {"remorse", "realization", "grief"},
                "title_patterns": ["soul", "mind", "life"],
                "weight": 1.2
            }
        }

        genre_scores = collections.defaultdict(float)

        # === Weighted topic frequency (based on topic distribution) ===
        topic_words = []
        for topic_name, words in profile.get('topics', {}).items():
            topic_idx = int(topic_name.split("_")[-1]) - 1
            topic_weight = np.mean([dist[topic_idx] for dist in profile["chapter_topic_distribution"]])
            topic_words.extend(words * int(topic_weight * 10 + 1))  # weight by prominence

        topic_counts = collections.Counter(topic_words)

        # === Remap dominant emotions to broader categories ===
        emotion_scores = profile["overall_emotions"]
        remapped_emotions = collections.Counter()
        for emotion, score in emotion_scores.items():
            if emotion in emotion_map:
                remapped_emotions[emotion_map[emotion]] += score

        top_remapped = set([e for e, _ in remapped_emotions.most_common(5)])

        # === Character emotion cues ===
        char_emotions = profile.get("character_emotions", {})
        char_top = []
        for emotions in char_emotions.values():
            if emotions:
                top = max(emotions.items(), key=lambda x: x[1])[0]
                if top in emotion_map:
                    char_top.append(emotion_map[top])
        char_top_set = set(char_top)

        # === Genre scoring ===
        for genre, pattern in genre_patterns.items():
            # Topic keyword match
            keyword_match = {
                kw for kw in pattern["keywords"]
                for topic_word in topic_counts
                if SequenceMatcher(None, kw, topic_word).ratio() > 0.8
            }

            genre_scores[genre] += sum(topic_counts[k] for k in keyword_match) * pattern["weight"]

            # Emotion match
            emotion_match = pattern["emotions"] & top_remapped
            genre_scores[genre] += len(emotion_match) * pattern["weight"]

            # Character emotions
            char_match = pattern["emotions"] & char_top_set
            genre_scores[genre] += len(char_match) * 0.5

            # Title pattern
            if title:
                for pattern_str in pattern["title_patterns"]:
                    if pattern_str in title.lower():
                        genre_scores[genre] += 2.0

        # === Signal from character count ===
        if len(char_emotions) > 10:
            genre_scores["mystery"] += 0.3
        elif len(char_emotions) < 4:
            genre_scores["romance"] += 0.2

        # === Fallback if all are low ===
        if max(genre_scores.values(), default=0) < 2.0:
            genre_scores["literary fiction"] += 1.0

        # === Get best genre(s) ===
        sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        best_genre, best_score = sorted_genres[0]
        total_score = sum(score for _, score in sorted_genres)
        confidence = best_score / total_score if total_score > 0 else 0

        if confidence < 0.3:
            best_genre = f"general {best_genre}"

        # Optionally return top 3 genres if needed
        profile["top_genre_candidates"] = sorted_genres[:3]

        return best_genre

    def chunk_text(self, sentences, chunk_size=40):
        """Chunk sentences into fixed-size groups for LDA topic modeling."""
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i + chunk_size])
            if len(chunk.strip().split()) > 20:  # Ensure non-trivial chunk
                chunks.append(chunk)
        return chunks


    def extract_book_profile(self, title, text):
        """
        Extract a comprehensive book profile with all features.

        Args:
            text (str): Book text

        Returns:
            dict: Book profile with emotional and thematic features
        """
        # Clean Gutenberg metadata
        text = self.clean_gutenberg_text(text)
        print(f"[DEBUG] Cleaned text length: {len(text)} characters")
        # Segment book into chapters
        chapters = self.segment_by_chapter(text)
        print(f"[DEBUG] Number of chapters: {len(chapters)}")
        # Process full text
        all_tokens, all_sentences = self.preprocess_text(text)
        print(f"[DEBUG] Total tokens: {len(all_tokens)}")
        # Extract overall emotion profile
        emotion_profile = self.extract_lexicon_based_emotions(all_tokens)
        print(f"[DEBUG] Overall emotions: {emotion_profile['scores']}")
        # Extract sentiment trajectory
        trajectory = self.create_sentiment_trajectory(chapters)
        print(f"[DEBUG] Sentiment trajectory length: {len(trajectory['polarity'])} chapters")
        # Apply smoothing
        #smoothed_trajectory = self.apply_rolling_average(trajectory)
        smoothed_trajectory = self.apply_gaussian_smoothing(trajectory, sigma=1)
        print(f"[DEBUG] Smoothed trajectory length: {len(smoothed_trajectory['polarity'])} chapters")
        # Extract TF-IDF features and topics
        sentences = sent_tokenize1(text)
        chunks = self.chunk_text(sentences, chunk_size=40)  # <-- new method
        tfidf_matrix, feature_names = self.extract_tfidf_features(chunks)
        topics, doc_topic_dist = self.extract_topics_with_lda(tfidf_matrix, feature_names)
        print(f"[DEBUG] Extracted topics: {len(topics)}")

        print(f"[DEBUG] Extracted topics: {len(topics)}")
        # Extract character emotions
        character_emotions = self.extract_character_emotions(text)
        #print(f"[DEBUG] Character emotions: {len(character_emotions)} characters")
        # Compile book profile
        book_profile = {
            "overall_emotions": emotion_profile["scores"],
            "sentiment_trajectory": {
                "raw": trajectory,
                "smoothed": smoothed_trajectory
            },
            "topics": topics,
            "chapter_topic_distribution": doc_topic_dist.tolist(),
            "character_emotions": character_emotions
        }

        genre = self.infer_genre_from_profile(book_profile, title=title)

        book_profile = {
            "title": title,
            "genre": genre,
            "overall_emotions": emotion_profile["scores"],
            "sentiment_trajectory": {
                "raw": trajectory,
                "smoothed": smoothed_trajectory
            },
            "topics": topics,
            "chapter_topic_distribution": doc_topic_dist.tolist(),
            "character_emotions": character_emotions
        }

        return book_profile

