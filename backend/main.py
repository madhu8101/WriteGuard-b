"""
FastAPI Backend for AI vs Human vs Humanized Text Classification
and Research Recommendation System.

This module provides REST API endpoints for:
- User authentication (simple validation)
- Text classification
- Document upload and analysis
- Research article recommendations
"""

import os
import io
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import uvicorn

# Import local modules
from predict import predict_text, TextClassifier
from dataset_loader import load_research_articles, filter_articles_by_type
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI(
    title="AI vs Human Text Classification API",
    description="Classify text as human-written, AI-generated, or humanized AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
_classifier = None
_recommendation_model = None
_articles_df = None
_articles_embeddings = None

# Pydantic models for request/response validation
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    success: bool
    message: str

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict

class UploadResponse(BaseModel):
    filename: str
    extracted_text: str 
    classification: dict

class RecommendationResponse(BaseModel):
    recommendations: list

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool


def get_classifier():
    """
    Get or initialize the text classifier.
    Uses lazy loading to avoid blocking server startup.
    """
    global _classifier
    if _classifier is None:
        model_path = os.path.join(os.path.dirname(__file__), "model", "classifier_v2")
        _classifier = TextClassifier(model_path=model_path)
        try:
            _classifier.load_model()
        except FileNotFoundError as e:
            print(f"Warning: Model not loaded. {e}")
            print("The /predict endpoint will not work until the model is trained.")
    return _classifier


def get_recommendation_model():
    """
    Get or initialize the sentence transformer for recommendations.
    """
    global _recommendation_model, _articles_df, _articles_embeddings
    
    if _recommendation_model is None:
        print("Loading sentence transformer for recommendations...")
        _recommendation_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence transformer loaded!")
    
    if _articles_df is None:
        try:
            _articles_df = load_research_articles()
            print(f"Loaded {_articles_df.shape[0]} research articles")
        except FileNotFoundError as e:
            print(f"Warning: Research articles not loaded. {e}")
            _articles_df = []
    
    if _articles_embeddings is None and len(_articles_df) > 0:
        print("Computing embeddings for research articles...")
        contents = _articles_df['content'].tolist()
        _articles_embeddings = _recommendation_model.encode(contents, convert_to_numpy=True, show_progress_bar=True)
        print(f"Computed embeddings for {_articles_embeddings.shape[0]} articles")
    
    return _recommendation_model, _articles_df, _articles_embeddings


@app.get("/", response_model=HealthCheck)
async def root():
    """
    Health check endpoint.
    """
    model_loaded = False
    try:
        classifier = get_classifier()
        model_loaded = classifier.model is not None
    except:
        pass
    
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }


@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Simple authentication endpoint.
    
    This is a basic implementation without database.
    In production, use proper authentication with hashed passwords and JWT tokens.
    """
    # Simple validation (in production, check against database)
    if request.email and request.password:
        # Basic email format validation is handled by Pydantic
        return {
            "success": True,
            "message": f"Welcome back, {request.email}!"
        }
    
    raise HTTPException(
        status_code=401,
        detail="Invalid email or password"
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Classify text as human-written, AI-generated, or humanized AI.
    
    Args:
        text: The text to classify
        
    Returns:
        Classification result with label and confidence score
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )
    
    try:
        classifier = get_classifier()
        if classifier.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not trained. Please run train.py first."
            )
        
        result = classifier.predict(request.text)
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {str(e)}"
        )


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and analyze a document (PDF or TXT).
    
    Extracts text from the file and classifies it.
    
    Args:
        file: PDF or TXT file to analyze
        
    Returns:
        Extracted text and classification result
    """
    # Validate file type
    allowed_extensions = {".pdf", ".txt"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file_ext}' not supported. Use PDF or TXT."
        )
    
    try:
        # Read file content
        content = await file.read()
        extracted_text = ""
        
        if file_ext == ".pdf":
            # Extract text from PDF
            import PyPDF2
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        elif file_ext == ".txt":
            # Read text file
            extracted_text = content.decode("utf-8")
        
        # Classify the extracted text
        classification = None
        if extracted_text.strip():
            classifier = get_classifier()
            if classifier.model is not None:
                classification = classifier.predict(extracted_text)
        
        return {
            "filename": file.filename,
            "extracted_text": extracted_text[:5000],  # Limit text length
            "classification": classification
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@app.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    text: str,
    filter_type: Optional[str] = None,
    top_k: int = 5
):
    """
    Get research article recommendations based on semantic similarity.
    
    Args:
        text: Text to find similar articles for
        filter_type: Filter by article type (ai, human, humanized, all)
        top_k: Number of recommendations to return (default: 5)
        
    Returns:
        List of recommended articles with similarity scores
    """
    if not text or len(text.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )
    
    try:
        model, articles_df, embeddings = get_recommendation_model()
        
        if articles_df is None or len(articles_df) == 0:
            raise HTTPException(
                status_code=404,
                detail="No research articles available"
            )
        
        # Filter articles if type is specified
        if filter_type and filter_type.lower() != "all":
            filtered_df = filter_articles_by_type(articles_df, filter_type)
            if len(filtered_df) == 0:
                return {"recommendations": []}
            
            # Get indices of filtered articles
            filtered_indices = filtered_df.index.tolist()
            filtered_embeddings = embeddings[filtered_indices]
        else:
            filtered_df = articles_df
            filtered_indices = list(range(len(articles_df)))
            filtered_embeddings = embeddings
        
        # Encode input text
        text_embedding = model.encode([text], convert_to_numpy=True)
        
        # Compute cosine similarity
        similarities = np.dot(filtered_embeddings, text_embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build recommendations list
        recommendations = []
        for idx in top_indices:
            original_idx = filtered_indices[idx]
            recommendations.append({
                "title": articles_df.iloc[original_idx]['title'],
                "content": articles_df.iloc[original_idx]['content'][:500],  # Snippet
                "type": articles_df.iloc[original_idx]['type'],
                "similarity": round(float(similarities[idx]), 4)
            })
        
        return {"recommendations": recommendations}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )