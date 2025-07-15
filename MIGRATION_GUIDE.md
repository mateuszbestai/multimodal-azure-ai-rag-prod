# Migration Guide: From Single File to Modular Flask Application

This guide helps you migrate from the original single-file `app.py` to the new modular Blueprint-based structure.

## 📋 Pre-Migration Checklist

1. **Backup your current application**
   ```bash
   cp -r . ../knowledge-assistant-backup
   ```

2. **Save your environment variables**
   ```bash
   cp .env ../backup.env
   ```

3. **Note any custom modifications** you've made to the original code

## 🚀 Migration Steps

### Step 1: Create New Directory Structure

```bash
# Create backend structure
mkdir -p backend/app/{api,config,core,models,utils,tests}
mkdir -p backend/app/api/{chat,health,images}
mkdir -p docker
mkdir -p scripts

# Move existing files
mv app.py scripts/app_original.py  # Keep as reference
mv ingest.py scripts/
mv requirements.txt backend/
```

### Step 2: Copy New Files

Copy all the new files from the artifacts above into their respective locations:

1. **Backend Application Files:**
   - `backend/app/__init__.py`
   - `backend/app/config/*.py`
   - `backend/app/api/*/*.py`
   - `backend/app/core/*.py`
   - `backend/app/models/*.py`
   - `backend/app/utils/*.py`
   - `backend/tests/*.py`
   - `backend/run.py`
   - `backend/wsgi.py`

2. **Docker Files:**
   - `docker/backend.Dockerfile`
   - `docker/frontend.Dockerfile`
   - `docker/nginx.conf`

3. **Configuration Files:**
   - `.dockerignore`
   - `.gitignore`
   - `docker-compose*.yml`
   - `backend/.env.example`

### Step 3: Update Environment Variables

```bash
cd backend
cp .env.example .env
# Copy your values from the backup
nano .env
```

### Step 4: Update Frontend API Configuration

Edit `frontend/src/App.tsx` if needed to ensure the API base URL is correct:

```typescript
const API_CONFIG = {
  baseUrl: import.meta.env.VITE_API_URL || 'http://localhost:5001',
  // ... rest of config
};
```

### Step 5: Install Dependencies

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend (if needed)
cd ../frontend
npm install
```

### Step 6: Test the Migration

1. **Run development servers:**
   ```bash
   # Terminal 1
   cd backend
   python run.py

   # Terminal 2
   cd frontend
   npm run dev
   ```

2. **Test key functionality:**
   - Health check: `curl http://localhost:5001/api/health`
   - Chat functionality
   - Image proxy
   - Source references

### Step 7: Docker Testing

```bash
# Build and test with Docker
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Access at http://localhost:3000
```

## 🔄 Key Changes

### API Endpoints
All endpoints now have the `/api` prefix and are organized by blueprint:
- `/api/chat` - Chat endpoints
- `/api/health` - Health check endpoints
- `/api/image/proxy` - Image proxy endpoint

### Configuration
- Environment-specific configuration classes
- Centralized configuration management
- Better secret handling

### Error Handling
- Custom exception classes
- Centralized error handlers
- Better error responses

### Code Organization
- Blueprints for modular endpoints
- Separate business logic in services
- Reusable utilities
- Type hints and schemas

## 🐛 Troubleshooting

### Import Errors
If you get import errors, ensure you're running from the correct directory:
```bash
cd backend
python run.py  # Not python app/run.py
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Azure Connection Issues
- Verify all environment variables are set correctly
- Check Azure service endpoints are accessible
- Ensure API keys are valid

### Port Conflicts
If port 5001 is in use:
```bash
PORT=5002 python run.py
```

## 🎯 Post-Migration

1. **Update CI/CD pipelines** to use the new structure
2. **Update documentation** with new API endpoints
3. **Plan for production deployment** using Docker
4. **Set up monitoring** with the new health endpoints

## 📚 Additional Resources

- [Flask Blueprints Documentation](https://flask.palletsprojects.com/en/3.0.x/blueprints/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Azure Web App Deployment](https://docs.microsoft.com/en-us/azure/app-service/)

## 💡 Best Practices

1. **Keep the old code** as reference until fully migrated
2. **Test thoroughly** before deploying to production
3. **Document any custom changes** you make
4. **Use environment variables** for all configuration
5. **Follow the new structure** for any new features