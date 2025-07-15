"""
Example API tests.
"""
import pytest
import json
from unittest.mock import patch, MagicMock

from app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app('testing')
    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture
def mock_llama_components():
    """Mock LlamaIndex components."""
    with patch('app.config.llama_config.initialize_llama_components'):
        yield


def test_health_check(client):
    """Test basic health check endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'timestamp' in data


def test_liveness_check(client):
    """Test liveness check endpoint."""
    response = client.get('/api/health/live')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['alive'] is True


def test_chat_endpoint_invalid_request(client):
    """Test chat endpoint with invalid request."""
    response = client.post('/api/chat', json={})
    assert response.status_code == 400
    
    response = client.post('/api/chat', json={'message': ''})
    assert response.status_code == 400


@patch('app.api.chat.services.ChatService.process_chat')
def test_chat_endpoint_success(mock_process, client):
    """Test successful chat request."""
    mock_process.return_value = {
        'response': 'Test response',
        'sources': {
            'pages': [1, 2],
            'images': []
        },
        'sourcePreviews': []
    }
    
    response = client.post('/api/chat', json={'message': 'Test question'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['response'] == 'Test response'
    assert 'sources' in data


@patch('app.api.chat.services.ChatService.process_chat')
def test_chat_endpoint_error_handling(mock_process, client):
    """Test chat endpoint error handling."""
    mock_process.side_effect = Exception('Test error')
    
    response = client.post('/api/chat', json={'message': 'Test question'})
    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data


def test_image_proxy_missing_path(client):
    """Test image proxy with missing path."""
    response = client.get('/api/image/proxy')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


@patch('requests.get')
def test_image_proxy_success(mock_get, client):
    """Test successful image proxy."""
    # Mock the requests.get response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {'Content-Type': 'image/jpeg'}
    mock_response.iter_content = lambda chunk_size: [b'fake-image-data']
    mock_get.return_value = mock_response
    
    response = client.get('/api/image/proxy?path=test.jpg')
    assert response.status_code == 200
    assert response.content_type == 'image/jpeg'


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.get('/api/health')
    assert 'Access-Control-Allow-Origin' in response.headers