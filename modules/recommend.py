from modules.database import profiles
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np

def get_dominant_emotion(emotion_score):
    """Get the dominant emotion from emotion scores"""
    return max(emotion_score.items(), key=lambda x: x[1])[0]

def get_top_emotions(emotion_score, top_n=3):
    """Get top N emotions from emotion scores"""
    return sorted(emotion_score.items(), key=lambda x: x[1], reverse=True)[:top_n]

def get_top_topics(profile, top_n=3):
    """Get top N topics from profile"""
    topics = []
    for topic_words in profile.get('topics', {}).values():
        if topic_words:
            topics.extend(topic_words[:2])  # Get top 2 words from each topic
    return list(set(topics))[:top_n]  # Remove duplicates and get top N

def flatten_emotions(emotion_dict):
    """Convert emotion dictionary to flat list"""
    return [float(v["score"]) if isinstance(v, dict) else float(v) for v in emotion_dict.values()]

def calculate_similarity_score(similarity):
    """Convert similarity score to percentage"""
    return round(similarity * 100)

def get_recommendations(new_profile, mode='profile', top_k=3):
    """Get book recommendations with detailed similarity information"""
    new_vector = np.array(flatten_emotions(new_profile["overall_emotions"])).reshape(1, -1)
    all_vectors, all_names, all_profiles = [], [], []

    # Collect all profiles
    for name, profile in profiles.items():
        if name != new_profile.get('title'):  # Skip the source book
            all_names.append(name)
            all_vectors.append(flatten_emotions(profile["overall_emotions"]))
            all_profiles.append(profile)

    if not all_vectors:
        return []

    all_vectors = np.array(all_vectors)
    recommendations = []

    if mode == 'profile':
        # Calculate similarities
        similarities = cosine_similarity(new_vector, all_vectors).flatten()
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        for idx in top_indices:
            profile = all_profiles[idx]
            similarity = similarities[idx]
            
            # Get dominant emotions and topics
            dominant_emotion = get_dominant_emotion(profile["overall_emotions"])
            top_emotions = get_top_emotions(profile["overall_emotions"])
            top_topics = get_top_topics(profile)
            
            recommendations.append({
                'title': all_names[idx],
                'genre': profile.get('genre', 'Unknown'),
                'similarity_type': 'profile',
                'similarity_score': calculate_similarity_score(similarity),
                'confidence': calculate_similarity_score(similarity),
                'dominant_emotion': dominant_emotion,
                'top_emotions': [emotion for emotion, _ in top_emotions],
                'themes': top_topics
            })

    elif mode == 'emotion':
        target_emotion = get_dominant_emotion(new_profile["overall_emotions"])
        
        # Calculate emotion similarities
        emotion_scores = []
        for profile in all_profiles:
            profile_emotion = get_dominant_emotion(profile["overall_emotions"])
            if profile_emotion == target_emotion:
                emotion_scores.append(1.0)
            else:
                emotion_scores.append(0.0)
        
        # Get top matches
        top_indices = np.argsort(emotion_scores)[::-1][:top_k]
        
        for idx in top_indices:
            profile = all_profiles[idx]
            recommendations.append({
                'title': all_names[idx],
                'genre': profile.get('genre', 'Unknown'),
                'similarity_type': 'emotion',
                'dominant_emotion': target_emotion,
                'confidence': calculate_similarity_score(emotion_scores[idx]),
                'top_emotions': [emotion for emotion, _ in get_top_emotions(profile["overall_emotions"])]
            })

    elif mode == 'cluster':
        # Perform clustering
        kmeans = KMeans(n_clusters=min(3, len(all_vectors)), random_state=42).fit(all_vectors)
        new_label = kmeans.predict(new_vector)[0]
        
        # Get books in the same cluster
        cluster_indices = np.where(kmeans.labels_ == new_label)[0]
        
        # Calculate similarities within cluster
        cluster_vectors = all_vectors[cluster_indices]
        cluster_similarities = cosine_similarity(new_vector, cluster_vectors).flatten()
        
        # Get top matches from cluster
        top_indices = cluster_indices[np.argsort(cluster_similarities)[::-1][:top_k]]
        
        for idx in top_indices:
            profile = all_profiles[idx]
            similarity = cluster_similarities[np.where(cluster_indices == idx)[0][0]]
            
            recommendations.append({
                'title': all_names[idx],
                'genre': profile.get('genre', 'Unknown'),
                'similarity_type': 'cluster',
                'similarity_score': calculate_similarity_score(similarity),
                'confidence': calculate_similarity_score(similarity),
                'top_emotions': [emotion for emotion, _ in get_top_emotions(profile["overall_emotions"])],
                'themes': get_top_topics(profile)
            })

    return recommendations
