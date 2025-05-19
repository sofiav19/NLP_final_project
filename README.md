# Book Analysis and Recommendation System

A web application that analyzes books to extract emotional and thematic features, generates visualizations, and provides book recommendations based on emotional profiles.

## Features

- 📚 Book Analysis
  - Emotional trajectory analysis
  - Character emotion detection
  - Topic modeling and visualization
  - Genre inference
  - Sentiment analysis

- 🔍 Book Recommendations
  - Similar profile matching
  - Emotion-based recommendations
  - Cluster-based recommendations

- 📊 Visualizations
  - Sentiment polarity trajectory
  - Emotional composition
  - Character emotion profiles
  - Topic heatmaps

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/book-analysis-system.git
cd book-analysis-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLP models:
```bash
python -m nltk.downloader punkt stopwords wordnet
python -m spacy download en_core_web_sm
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5050
```

3. Upload a book (in .txt format) or select from the stored books to analyze.

## Project Structure

```
book-analysis-system/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── modules/
│   ├── database.py       # Profile storage and management
│   ├── extractor.py      # Book feature extraction
│   ├── model_cache.py    # ML model caching
│   ├── plotting.py       # Visualization generation
│   ├── process_text.py   # Text processing pipeline
│   └── recommend.py      # Book recommendation system
├── static/
│   ├── plots/           # Generated visualizations
│   ├── profiles/        # Book profiles
│   └── style.css        # CSS styles
└── templates/           # HTML templates
```

## Technical Details

### Book Analysis Pipeline

1. Text Preprocessing
   - Gutenberg metadata removal
   - Chapter detection
   - Text cleaning and normalization

2. Feature Extraction
   - Emotional analysis using BERT
   - Sentiment analysis with TextBlob
   - Character detection with spaCy
   - Topic modeling with LDA

3. Visualization Generation
   - Sentiment trajectories
   - Emotional composition
   - Character emotion profiles
   - Topic distributions

### Recommendation System

- Profile-based: Cosine similarity on emotional profiles
- Emotion-based: Matching dominant emotions
- Cluster-based: K-means clustering of emotional profiles

## Error Handling

The system includes robust error handling for:
- File upload issues
- Text processing errors
- Model loading failures
- Analysis pipeline errors

Users are notified of errors through a user-friendly interface.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses the GoEmotions model for emotion detection
- Implements spaCy for NLP tasks
- Leverages scikit-learn for topic modeling
- Built with Flask for the web interface