import os
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_sentiment_trajectory(profile, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)

    trajectory = profile["sentiment_trajectory"].get("smoothed", {})
    polarity = trajectory.get("polarity", [])
    emotions = trajectory.get("emotions", {})
    if not emotions or not polarity:
        raise ValueError("No sentiment data found in sentiment trajectory.")

    # Plot polarity
    plt.figure(figsize=(12, 6))
    plt.plot(polarity, color="blue")
    plt.title("Sentiment Polarity Over Chapters")
    plt.xlabel("Chapter")
    plt.ylabel("Polarity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_polarity_trajectory.png"))
    plt.close()

    # Plot top 5 emotions
    avg_intensity = {e: np.mean(v) for e, v in emotions.items()}
    top_emotions = sorted(avg_intensity.items(), key=lambda x: x[1], reverse=True)[:5]
    plt.figure(figsize=(12, 8))
    for emotion, _ in top_emotions:
        plt.plot(emotions[emotion], label=emotion.capitalize())
    plt.title("Top 5 Emotional Trajectories")
    plt.xlabel("Chapter")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_emotion_trajectory.png"))
    plt.close()

    # Stacked area chart of top 5
    plt.figure(figsize=(14, 8))
    arrays = [emotions[e] for e, _ in top_emotions]
    plt.stackplot(range(len(arrays[0])), arrays, labels=[e.capitalize() for e, _ in top_emotions], alpha=0.8)
    plt.title("Stacked Emotional Composition")
    plt.xlabel("Chapter")
    plt.ylabel("Proportion")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_emotion_composition.png"))
    plt.close()

def plot_character_emotions(profile, output_dir, prefix):
    emotions_by_character = profile.get("character_emotions", {})
    if not emotions_by_character:
        return

    os.makedirs(output_dir, exist_ok=True)
    top_chars = sorted(emotions_by_character.items(), key=lambda x: sum(x[1].values()), reverse=True)[:5]
    emotions = list(next(iter(emotions_by_character.values())).keys())
    bar_width = 0.15
    r = np.arange(len(emotions))

    plt.figure(figsize=(14, 8))
    for i, (char, char_emotions) in enumerate(top_chars):
        values = [char_emotions.get(e, 0) for e in emotions]
        plt.bar(r + i * bar_width, values, width=bar_width, label=char)

    plt.xlabel('Emotions')
    plt.ylabel('Intensity')
    plt.title('Top Character Emotion Distribution')
    plt.xticks(r + bar_width * 2, [e.capitalize() for e in emotions], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_character_emotions.png"))
    plt.close()

def plot_topic_heatmap(profile, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    topic_dist = np.array(profile["chapter_topic_distribution"])
    topic_labels = []

    # Generate readable multi-line topic labels (top 3 words per topic)
    for i in range(topic_dist.shape[1]):
        words = profile["topics"].get(f"topic_{i+1}", [])[:3]  # use topic_1, topic_2, ...
        label = "\n".join(words) if words else f"Topic {i+1}"
        topic_labels.append(label)

    df = pd.DataFrame(topic_dist, columns=topic_labels)

    # Optional: clip values to improve color contrast
    df = df.clip(lower=0, upper=0.3)

    # Plot
    plt.figure(figsize=(min(22, len(topic_labels) * 1.2), 10))
    sns.heatmap(
        df,
        cmap="YlOrRd",
        cbar_kws={"label": "Topic Relevance"},
        vmin=0
    )

    # Comic-style aesthetics
    plt.title("Topic Distribution Across Chapters")
    plt.xlabel("Topics", fontsize=14)
    plt.ylabel("Chapters", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_topic_heatmap.png")
    plt.savefig(output_path)
    plt.close()

    
