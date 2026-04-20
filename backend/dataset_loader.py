"""
Dataset Loader Module
Handles loading and processing of classification and research articles datasets.
"""

import pandas as pd
import os
from typing import List, Dict, Tuple

# Dataset paths relative to backend directory
CLASSIFICATION_DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "data.csv")
RESEARCH_ARTICLES_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "research_articles.csv")


def load_classification_dataset() -> pd.DataFrame:
    """
    Load the classification dataset for training the text classifier.
    
    Returns:
        pd.DataFrame: DataFrame with 'text' and 'label' columns
    """
    if not os.path.exists(CLASSIFICATION_DATASET_PATH):
        raise FileNotFoundError(
            f"Classification dataset not found at {CLASSIFICATION_DATASET_PATH}. "
            "Please ensure the dataset exists."
        )
    
    df = pd.read_csv(CLASSIFICATION_DATASET_PATH)
    
    # Validate required columns
    required_columns = ['text', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Drop rows with missing values
    df = df.dropna(subset=['text', 'label'])
    
    return df


def load_research_articles() -> pd.DataFrame:
    """
    Load the research articles dataset for the recommendation system.
    
    Returns:
        pd.DataFrame: DataFrame with 'title', 'content', and 'type' columns
    """
    if not os.path.exists(RESEARCH_ARTICLES_PATH):
        raise FileNotFoundError(
            f"Research articles dataset not found at {RESEARCH_ARTICLES_PATH}. "
            "Please ensure the dataset exists."
        )
    
    df = pd.read_csv(RESEARCH_ARTICLES_PATH)
    
    # Validate required columns
    required_columns = ['title', 'content', 'type']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Drop rows with missing values
    df = df.dropna(subset=['title', 'content', 'type'])
    
    return df


def get_label_mapping() -> Dict[str, int]:
    """
    Get the mapping between label names and integer IDs.
    
    Returns:
        Dict[str, int]: Mapping from label name to integer ID
    """
    return {
        "human": 0,
        "ai": 1,
        "humanized": 2
    }


def get_id_to_label_mapping() -> Dict[int, str]:
    """
    Get the reverse mapping from integer IDs to label names.
    
    Returns:
        Dict[int, str]: Mapping from integer ID to label name
    """
    return {
        0: "human",
        1: "ai",
        2: "humanized"
    }


def filter_articles_by_type(articles: pd.DataFrame, article_type: str) -> pd.DataFrame:
    """
    Filter research articles by type.
    
    Args:
        articles: DataFrame of research articles
        article_type: Type to filter by ('ai', 'human', 'humanized', or 'all')
    
    Returns:
        pd.DataFrame: Filtered articles
    """
    if article_type and article_type.lower() != "all":
        return articles[articles['type'].str.lower() == article_type.lower()]
    return articles