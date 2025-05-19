import os
from modules.extractor import BookFeatureExtractor
from modules.database import save_profile, save_profiles, add_profile
from modules.plotting import (
    plot_sentiment_trajectory,
    plot_character_emotions,
    plot_topic_heatmap
)
from modules.database import is_duplicate
extractor = BookFeatureExtractor()
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

PROFILE_DIR = os.path.join(BASE_DIR, "..", "static", "profiles")
PLOT_DIR = os.path.join(BASE_DIR, "..", "static", "plots")
def generate_plots_and_summary(filepath):
    filename = os.path.basename(filepath)
    base = os.path.splitext(filename)[0]
    output_dir = os.path.join("static", "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Load book
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Analyze with extractor
    print(f"Extracting profile for {base}...")
    profile = extractor.extract_book_profile(base, text)
    #print(f"Generated profile for {base}: {profile}")

    

    profile_path = os.path.join(PROFILE_DIR, f"{base}.json")
    os.makedirs(PLOT_DIR, exist_ok=True)
    # Always overwrite the profile and plots
    save_profile(profile, profile_path)  # Overwrite JSON
    add_profile(base, profile)           # Update in-memory dict
    save_profiles()                      # Rewrite profiles.json
         
    # Generate plots
    plot_paths = []
    
    # Generate sentiment trajectory plot
    try:
        plot_sentiment_trajectory(profile, output_dir, base)
        plot_paths.append(f"plots/{base}_polarity_trajectory.png")
        plot_paths.append(f"plots/{base}_emotion_trajectory.png")
        plot_paths.append(f"plots/{base}_emotion_composition.png")
    except Exception as e:
        print(f"Warning: Could not generate sentiment plots: {str(e)}")

    # Generate character emotions plot
    try:
        plot_character_emotions(profile, output_dir, base)
        plot_paths.append(f"plots/{base}_character_emotions.png")
    except Exception as e:
        print(f"Warning: Could not generate character emotions plot: {str(e)}")

    # Generate topic heatmap plot only if we have topic distribution data
    try:
        if 'chapter_topic_distribution' in profile:
            plot_topic_heatmap(profile, output_dir, base)
            plot_paths.append(f"plots/{base}_topic_heatmap.png")
    except Exception as e:
        print(f"Warning: Could not generate topic heatmap: {str(e)}")

    # Return summary and paths to plots
    summary = f"Book genre: {profile['genre']}. Top emotions: {sorted(profile['overall_emotions'].items(), key=lambda x: -x[1])[:3]}"

    return profile, plot_paths, summary

def extract_book_profile(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    base = os.path.splitext(os.path.basename(filepath))[0]
    profile = extractor.extract_book_profile(base, text)
    return profile
