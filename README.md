# AI vs Human vs Humanized Text Classification and Research Recommendation System

A complete end-to-end system for classifying text as human-written, AI-generated, or humanized AI, with semantic similarity-based research article recommendations.

## 🚀 Features

### Text Classification
- **Human-written**: Text created by humans
- **AI-generated**: Text produced by AI models
- **Humanized AI**: AI text that has been edited by humans

### Research Recommendations
- Semantic similarity-based article recommendations
- Filter by article type (AI, Human, Humanized)
- Top 5 most relevant research articles

### Document Analysis
- Upload PDF and TXT files
- Extract text from documents
- Classify uploaded documents
- Get recommendations based on document content

### Modern UI
- Clean, responsive React frontend
- Login system with validation
- Real-time classification results
- Loading indicators and error handling

## 🛠️ Tech Stack

### Backend
- **Python** - Core language
- **FastAPI** - Web framework
- **PyTorch** - Deep learning framework
- **Transformers** - HuggingFace models
- **Sentence Transformers** - Semantic similarity
- **PyPDF2** - PDF text extraction

### Frontend
- **React.js** - UI framework
- **Axios** - HTTP client
- **React Router** - Client-side routing

### Machine Learning
- **distilgpt2** - Fine-tuned for classification
- **Sentence Transformers** - All-MiniLM-L6-v2 for embeddings
- **CPU-only training** - Optimized for CPU performance

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+ (for frontend)

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd tri
   ```

2. **Set up Python environment**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create directories**
   ```bash
   mkdir -p backend/model/classifier
   mkdir -p dataset
   ```

### Frontend Setup

1. **Navigate to frontend**
   ```bash
   cd ../frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

## 🏗️ Project Structure

```
tri/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── train.py             # Model training script
│   ├── predict.py           # Prediction module
│   ├── dataset_loader.py    # Dataset utilities
│   ├── requirements.txt     # Python dependencies
│   └── model/               # Trained model directory
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main app component
│   │   ├── index.js        # App entry point
│   │   ├── pages/          # Login and Dashboard
│   │   ├── components/     # Reusable components
│   │   └── index.css       # Global styles
│   ├── public/
│   │   └── logo.png        # App logo
│   └── package.json        # Frontend dependencies
├── dataset/
│   ├── data.csv            # Classification dataset
│   └── research_articles.csv # Research articles dataset
└── README.md               # This file
```

## 🎯 Usage

### 1. Train the Model

```bash
cd backend
python train.py
```

This will:
- Load the classification dataset
- Fine-tune distilgpt2 on your data
- Save the trained model to `backend/model/classifier/`
- Train for 3 epochs with batch size 1 (CPU-optimized)

### 2. Start the Backend Server

```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`

### 3. Start the Frontend

```bash
cd frontend
npm start
```

The React app will be available at `http://localhost:3000`

### 4. Access the Application

1. Open your browser and go to `http://localhost:3000`
2. Log in with any valid email and password
3. Use the dashboard to classify text or upload documents

## 🔧 API Endpoints

### Authentication
- `POST /login` - User authentication

### Text Classification
- `POST /predict` - Classify text
- `POST /upload` - Upload and analyze documents

### Research Recommendations
- `GET /recommendations` - Get article recommendations

### Health Check
- `GET /` - System health check

## 📊 Datasets

### Classification Dataset (`dataset/data.csv`)
- **Format**: `text,label`
- **Labels**: `human`, `ai`, `humanized`
- **Size**: 1000+ examples
- **Content**: Mixed human-written and AI-generated text samples

### Research Articles Dataset (`dataset/research_articles.csv`)
- **Format**: `title,content,type`
- **Types**: `ai`, `human`, `humanized`
- **Content**: Research papers and articles for recommendations

## 🤖 Machine Learning Details

### Classification Model
- **Base Model**: distilgpt2 (CPU-optimized)
- **Training**: 3 epochs, batch size 1
- **Labels**: 3 classes (human, ai, humanized)
- **Output**: Classification + confidence score

### Recommendation System
- **Model**: All-MiniLM-L6-v2
- **Method**: Cosine similarity
- **Output**: Top 5 most similar articles

## 🎨 Frontend Features

### Login Page
- Email and password validation
- Demo credentials (any valid email/password)
- Error handling and loading states

### Dashboard
- Text input for classification
- File upload (PDF/TXT)
- Filter recommendations by type
- Real-time results display
- Extracted text preview

### Components
- **LoadingIndicator**: Spinner with customizable size
- **ClassificationResult**: Detailed classification results
- **RecommendationList**: Research article recommendations

## 🚀 Performance Optimizations

### CPU Training
- Small batch size (1-2) for memory efficiency
- distilgpt2 for faster inference
- No CUDA usage for CPU compatibility

### Model Serving
- Lazy loading of models
- Efficient inference with CPU optimization
- Caching for repeated requests

### Frontend
- Responsive design for all screen sizes
- Optimized state management
- Efficient API calls with loading states

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**
   - Run `python train.py` first to train the model
   - Check that `backend/model/classifier/` exists

2. **Dataset not found**
   - Ensure `dataset/data.csv` and `dataset/research_articles.csv` exist
   - Check file paths are correct

3. **Frontend can't connect to backend**
   - Ensure backend is running on `http://localhost:8000`
   - Check CORS settings in `main.py`

4. **Memory issues during training**
   - The system is optimized for CPU with small batch sizes
   - Reduce batch size further if needed in `train.py`

### Dependencies

If you encounter dependency issues:

```bash
# Backend
pip install --upgrade torch transformers datasets

# Frontend
npm install react-router-dom axios
```

## 🔒 Security Notes

- Simple authentication for demo purposes
- CORS enabled for development
- Input validation on all endpoints
- File upload restrictions (PDF/TXT only)

## 📈 Future Enhancements

- Database integration for user management
- Model versioning and A/B testing
- Advanced analytics and reporting
- Multi-language support
- Real-time collaboration features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace for transformers and datasets
- FastAPI team for excellent web framework
- React community for amazing ecosystem
- All contributors to open-source ML libraries

---

**Built with ❤️ for AI text classification and research discovery**