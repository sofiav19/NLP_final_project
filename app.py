from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import threading
import uuid
import traceback 
from modules.process_text import generate_plots_and_summary, extract_book_profile
from modules.recommend import get_recommendations
from modules.database import load_profiles, profiles
import logging
import os
from werkzeug.exceptions import HTTPException
import sys
import time
from datetime import datetime
from threading import Thread

# Silence TensorFlow, urllib3, and Hugging Face logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 3 = fatal only

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("nltk").setLevel(logging.WARNING)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
for handler in log.handlers[:]:
    log.removeHandler(handler)

results = {}

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

# Progress tracking dict
progress = {}

# Add after app initialization
class AnalysisError(Exception):
    """Custom exception for analysis errors"""
    pass

def handle_error(error):
    """Global error handler"""
    if isinstance(error, AnalysisError):
        flash(str(error), 'error')
        return render_template('error.html', error=str(error)), 400
    elif isinstance(error, HTTPException):
        return error
    else:
        # Log the full error for debugging
        print(f"Unexpected error: {str(error)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        flash("An unexpected error occurred. Please try again later.", 'error')
        return render_template('error.html', error="An unexpected error occurred"), 500

# Register error handler
app.register_error_handler(Exception, handle_error)

# Add after the progress dict
tasks = {}  # Store task information

def get_task(task_id):
    """
    Get task information from the tasks dictionary.
    
    Args:
        task_id (str): Unique task identifier
        
    Returns:
        dict: Task information including status, profile, and results
    """
    return tasks.get(task_id)

def save_task(task_id, **kwargs):
    """
    Save or update task information.
    
    Args:
        task_id (str): Unique task identifier
        **kwargs: Task information to store
    """
    if task_id not in tasks:
        tasks[task_id] = {}
    tasks[task_id].update(kwargs)

# -----------------------------
# ✅ ROUTE: Home page
# -----------------------------
@app.route('/')
def index():
    book_list = [{"title": v.get("title", k), "filename": k, "genre": v.get("genre", "Unknown")} for k, v in profiles.items()]
    return render_template('index.html', books=book_list)

# -----------------------------
# ✅ ROUTE: Analyze book
# -----------------------------
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    try:
        if request.method == 'POST':
            stored = request.form.get("stored")
            if stored:
                # Load from precomputed profile
                if stored not in profiles:
                    return render_template('error.html', error="Selected book profile not found.")

                task_id = str(uuid.uuid4())
                profile = profiles[stored]

                # Build expected plot paths
                base = os.path.splitext(stored)[0]
                plots = [
                    f"plots/{base}_polarity_trajectory.png",
                    f"plots/{base}_emotion_trajectory.png",
                    f"plots/{base}_emotion_composition.png",
                    f"plots/{base}_character_emotions.png",
                    f"plots/{base}_topic_heatmap.png"
                ]
                plots = [p for p in plots if os.path.exists(os.path.join("static", p))]

                # Create summary
                summary = generate_analysis_summary(profile, stored)

                # Save directly into task
                save_task(task_id,
                    status='completed',
                    profile=profile,
                    plots=plots,
                    summary=summary
                )

                print(f"[DEBUG] Loaded precomputed profile for {stored} as task {task_id}")
                return redirect(url_for('result', task_id=task_id))

            # Process uploaded book
            if 'book' not in request.files:
                return render_template('error.html', error="No file uploaded")

            file = request.files['book']
            if file.filename == '':
                return render_template('error.html', error="No file selected")
            if not file.filename.endswith('.txt'):
                return render_template('error.html', error="Only .txt files are supported")

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            task_id = str(uuid.uuid4())
            save_task(task_id, 
                    status= 'processing',
                    start_time=datetime.now().isoformat(),
                    filepath=filepath
                )

            thread = Thread(target=run_analysis_task, args=(task_id, filepath))
            thread.daemon = True
            thread.start()

            return redirect(url_for('result', task_id=task_id))

        # GET request – show available books
        book_list = [{"title": v.get("title", k), "filename": k, "genre": v.get("genre", "Unknown")} 
                    for k, v in profiles.items()]
        return render_template("analyze.html", books=book_list)

    except Exception as e:
        print(f"Error in analyze route: {str(e)}")
        return render_template('error.html', error="Error processing request. Please try again.")

# -----------------------------
# ✅ THREAD TASK: Background analysis
# -----------------------------
def run_analysis_task(task_id, filepath):
    """Run book analysis in background."""
    try:
        print(f"[DEBUG] Starting analysis for task {task_id}")
        # Generate plots and summary
        profile, plots, summary = generate_plots_and_summary(filepath)
        
        print(f"[DEBUG] Finished analysis for task {task_id}")
        # Update task with results
        save_task(task_id,
            status='completed',
            profile=profile,
            plots=plots,
            summary=summary
        )
        print(f"[DEBUG] Saved results for task {task_id}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        # Update task with error
        save_task(task_id,
            status='error',
            error=f"Error analyzing book content: {str(e)}"
        )

# -----------------------------
# ✅ ROUTE: Track progress
# -----------------------------
from flask import jsonify  # add this at the top

@app.route('/progress/<task_id>')
def get_progress(task_id):
    return jsonify({"progress": progress.get(task_id, 0)})

def generate_analysis_summary(profile, filename):
    """
    Generate a comprehensive summary of the book analysis.
    
    Args:
        profile (dict): Book profile with analysis results
        filename (str): Name of the book file
        
    Returns:
        str: Generated summary
    """
    # Get basic information
    title = filename.replace("_", " ").replace(".txt", "")
    genre = profile.get('genre', 'general fiction')
    num_chapters = len(profile.get('chapters', []))
    
    # Get emotional profile
    emotions = profile.get('overall_emotions', {})
    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    emotion_desc = ", ".join(f"{emotion} ({score:.2f})" for emotion, score in top_emotions)
    
    # Get character information
    characters = profile.get('character_emotions', {})
    main_characters = sorted(characters.items(), 
                           key=lambda x: sum(x[1].values()) if x[1] else 0, 
                           reverse=True)[:3]
    
    # Get thematic information
    topics = profile.get('topics', {})
    main_topics = []
    for topic_words in topics.values():
        if topic_words:
            main_topics.extend(topic_words[:3])
    main_topics = list(set(main_topics))[:5]  # Get unique topics
    
    # Generate summary
    summary_parts = []
    
    # Introduction
    summary_parts.append(f"'{title}' is a {genre} novel that unfolds across {num_chapters} chapters. ")
    
    # Emotional journey
    if top_emotions:
        summary_parts.append(f"The narrative is characterized by strong {emotion_desc} emotions, ")
        if len(top_emotions) > 1:
            summary_parts.append("creating a complex emotional landscape. ")
        else:
            summary_parts.append("setting the tone for the story. ")
    
    # Character analysis
    if main_characters:
        char_names = [char for char, _ in main_characters]
        summary_parts.append(f"The story revolves around {', '.join(char_names[:-1])} and {char_names[-1]}, ")
        char_emotions = []
        for char, emotions in main_characters:
            if emotions:
                top_emotion = max(emotions.items(), key=lambda x: x[1])
                char_emotions.append(f"{char} (primarily {top_emotion[0]})")
        if char_emotions:
            summary_parts.append(f"whose emotional journeys are marked by {', '.join(char_emotions)}. ")
    
    # Thematic analysis
    if main_topics:
        summary_parts.append(f"The narrative explores themes of {', '.join(main_topics)}, ")
        summary_parts.append("weaving these elements throughout the story. ")
    
    # Overall assessment
    if emotions:
        dominant_emotion = top_emotions[0][0]
        summary_parts.append(f"The overall emotional tone of the book is predominantly {dominant_emotion}, ")
        summary_parts.append("creating a compelling and engaging reading experience.")
    
    return "".join(summary_parts)

@app.route('/result/<task_id>')
def result(task_id):
    task = get_task(task_id)
    if not task:
        return render_template('error.html', error="Task not found. Please try analyzing the book again.")
    
    if task['status'] == 'error':
        return render_template('error.html', error=task.get('error', 'An error occurred during analysis.'))
    
    if task['status'] == 'processing':
        # ✅ Return progress page with the "Analyzing your book" text (important for JS detection)
        return render_template('progress.html', task_id=task_id)

    # ✅ Task is completed
    profile = task['profile']
    plots = task.get('plots', [])
    summary = task.get('summary', '')

    polarity_trajectory_plot = next((p for p in plots if 'polarity_trajectory' in p), None)
    emotion_trajectory_plot = next((p for p in plots if 'emotion_trajectory' in p), None)
    emotion_composition_plot = next((p for p in plots if 'emotion_composition' in p), None)
    character_emotions_plot = next((p for p in plots if 'character_emotions' in p), None)
    topic_heatmap_plot = next((p for p in plots if 'topic_heatmap' in p), None)

    return render_template('analysis_result.html',
        title=profile.get('title', 'Unknown Title'),
        genre=profile.get('genre', 'Unknown Genre'),
        summary=summary,
        polarity_trajectory_plot=polarity_trajectory_plot,
        emotion_trajectory_plot=emotion_trajectory_plot,
        emotion_composition_plot=emotion_composition_plot,
        character_emotions_plot=character_emotions_plot,
        topic_heatmap_plot=topic_heatmap_plot
    )

# -----------------------------
# ✅ ROUTE: Recommend
# -----------------------------
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        # ✅ Reuse stored book
        if 'stored' in request.form:
            filename = request.form['stored']
            profile = profiles[filename]
            mode = request.form.get('mode', 'profile')
            recommendations = get_recommendations(profile, mode)
            return render_template("recommend_result.html", filename=filename, recommendations=recommendations, mode=mode)

        # ✅ Upload new book
        file = request.files['file']
        mode = request.form.get('mode', 'profile')
        if file.filename == '':
            flash("No file selected")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        profile = extract_book_profile(filepath)
        recommendations = get_recommendations(profile, mode)
        return render_template("recommend_result.html", filename=filename, recommendations=recommendations, mode=mode)

    return render_template("recommend.html", books=[{"title": v.get("title", k), "filename": k, "genre": v.get("genre", "Unknown")} for k, v in profiles.items()])

# -----------------------------
# ✅ MAIN
# -----------------------------
if __name__ == '__main__':
    load_profiles()
    print("🔥 Flask app launching from", __file__)
    app.run(debug=True, host="0.0.0.0", port=5050)
