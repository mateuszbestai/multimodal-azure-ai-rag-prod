# Knowledge Assistant - Production Ready Application

A multimodal RAG (Retrieval Augmented Generation) application using Azure OpenAI, Azure Search, and React.

## 🏗️ Architecture

### Backend (Flask)
- **Blueprint-based architecture** for modular API endpoints
- **Production-ready** with Gunicorn WSGI server
- **Error handling** and custom exceptions
- **Health checks** for Kubernetes/Azure readiness
- **Configuration management** for different environments
- **Rate limiting** and caching support

### Frontend (React + TypeScript)
- **Modern UI** with Tailwind CSS
- **Dark mode** support
- **Streaming responses** for better UX
- **Chat history** with local storage
- **Image viewer** with proxy support

## 📁 Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/          # Blueprint modules
│   │   ├── config/       # Configuration
│   │   ├── core/         # Core business logic
│   │   ├── models/       # Data models
│   │   └── utils/        # Utilities
│   ├── run.py           # Development server
│   └── wsgi.py          # Production entry point
├── frontend/             # React application
├── docker/              # Docker configurations
└── scripts/             # Utility scripts
```

## 🚀 Deployment

### Local Development

1. **Clone and setup environment:**
```bash
# Clone repository
git clone <your-repo-url>
cd knowledge-assistant

# Setup backend
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env
# Edit .env with your Azure credentials

# Setup frontend
cd ../frontend
npm install
```

2. **Run development servers:**
```bash
# Terminal 1 - Backend
cd backend
python run.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### Docker Development

```bash
# Build and run with docker-compose
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### Production Deployment

#### Using Docker

1. **Build production images:**
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build
```

2. **Run production stack:**
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

#### Deploy to Azure Web App

1. **Build and push to Azure Container Registry:**
```bash
# Login to Azure
az login

# Create container registry (if not exists)
az acr create --resource-group myResourceGroup --name myregistry --sku Basic

# Login to registry
az acr login --name myregistry

# Build and push images
docker build -f docker/backend.Dockerfile -t myregistry.azurecr.io/knowledge-assistant-backend:latest .
docker build -f docker/frontend.Dockerfile -t myregistry.azurecr.io/knowledge-assistant-frontend:latest .

#for testing purposes
docker run -p 5000:80 --env-file .env

docker push myregistry.azurecr.io/knowledge-assistant-backend:latest
docker push myregistry.azurecr.io/knowledge-assistant-frontend:latest
```

2. **Create Azure Web App:**
```bash
# Create App Service Plan
az appservice plan create --name myAppServicePlan --resource-group myResourceGroup --sku B1 --is-linux

# Create Web App for Containers
az webapp create --resource-group myResourceGroup --plan myAppServicePlan --name myKnowledgeAssistant --multicontainer-config-type compose --multicontainer-config-file docker-compose.prod.yml
```

3. **Configure environment variables:**
```bash
az webapp config appsettings set --resource-group myResourceGroup --name myKnowledgeAssistant --settings \
  AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
  AZURE_OPENAI_API_KEY="your-key" \
  AZURE_SEARCH_SERVICE_ENDPOINT="https://your-search.search.windows.net" \
  AZURE_SEARCH_ADMIN_KEY="your-key" \
  # ... other settings
```

4. **Enable continuous deployment (optional):**
```bash
az webapp deployment container config --enable-cd true --name myKnowledgeAssistant --resource-group myResourceGroup
```

## 🔧 Configuration

### Environment Variables

See `backend/.env.example` for all required environment variables.

### Azure Resources Required

1. **Azure OpenAI Service**
   - GPT-4 deployment
   - Text embedding deployment

2. **Azure Cognitive Search**
   - Search service with vector search capability

3. **Azure Storage Account**
   - Blob container for images
   - SAS token for access

4. **Azure Document Intelligence** (for document ingestion)

## 🏥 Health Checks

The application provides several health check endpoints:

- `/api/health` - Basic health check
- `/api/health/detailed` - Detailed health with dependency checks
- `/api/health/ready` - Kubernetes readiness probe
- `/api/health/live` - Kubernetes liveness probe

## 🔒 Security

- **CORS** configuration for API access control
- **Rate limiting** to prevent abuse
- **Security headers** in nginx configuration
- **Non-root user** in Docker containers
- **Environment-based** configuration

## 📊 Monitoring

For production monitoring, consider integrating:

1. **Azure Application Insights**
2. **Azure Monitor**
3. **Custom logging with Azure Log Analytics**

## 🧪 Testing

```bash
cd backend
pytest
pytest --cov=app  # With coverage
```

## 📚 API Documentation

### Chat Endpoints

- `POST /api/chat` - Process chat message
- `POST /api/chat/stream` - Stream chat response

### Health Endpoints

- `GET /api/health` - Basic health check
- `GET /api/health/detailed` - Detailed health status

### Image Proxy

- `GET /api/image/proxy?path={encoded_path}` - Proxy private blob images

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.