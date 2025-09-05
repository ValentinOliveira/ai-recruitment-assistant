# Moses Omondi's AI Recruitment Assistant - Deployment Guide

## 🚀 Quick Start

### Local Development
1. **Start the API server:**
   ```bash
   cd api
   python main.py
   ```

2. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Open the web example:**
   ```bash
   open web/example.html
   ```

### Docker Deployment
1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Access the API:**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 4060 or better)
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ for models and data

### Software Requirements
- **Docker & Docker Compose** (recommended)
- **Python 3.9+** (for local development)
- **NVIDIA Docker Runtime** (for GPU support)
- **CUDA 12.1+** (if running locally)

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key configurations:
- `API_PORT`: API server port (default: 8000)
- `CORS_ORIGINS`: Allowed origins for web integration
- `ENVIRONMENT`: development/staging/production
- `CUDA_VISIBLE_DEVICES`: GPU device selection

### Model Path
Ensure the trained model is available at:
```
models/moses-recruitment-assistant/
├── adapter_config.json
├── adapter_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

## 🐳 Docker Deployment

### Basic Deployment
```bash
# Build and start the service
docker-compose up --build -d

# Check logs
docker-compose logs -f moses-ai-assistant

# Stop the service
docker-compose down
```

### Production Deployment
```bash
# Start with production profile
docker-compose --profile production up -d

# Start with monitoring
docker-compose --profile monitoring up -d

# Full production stack
docker-compose --profile production --profile monitoring --profile cache up -d
```

### GPU Support
Ensure NVIDIA Docker runtime is installed:
```bash
# Test GPU access
docker run --gpus all nvidia/cuda:12.1-runtime-ubuntu20.04 nvidia-smi
```

## ☁️ Cloud Deployment

### AWS Deployment
1. **EC2 with GPU** (g4dn.xlarge or better)
2. **ECS with GPU support**
3. **SageMaker Endpoint** (for managed inference)

```bash
# Example EC2 user data script
#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker

# Install NVIDIA Docker
curl -s -L https://nvidia.github.io/nvidia-docker/centos7/nvidia-docker.repo | tee /etc/yum.repos.d/nvidia-docker.repo
yum install -y nvidia-docker2
systemctl restart docker

# Clone and start the application
git clone https://github.com/Moses-Omondi/ai-recruitment-assistant.git
cd ai-recruitment-assistant
docker-compose up -d
```

### Google Cloud Platform
1. **Compute Engine with GPU**
2. **Cloud Run** (CPU-only version)
3. **GKE with GPU nodes**

### Microsoft Azure
1. **VM with GPU**
2. **Container Instances**
3. **AKS with GPU nodes**

## 🌐 Website Integration

### Simple Integration
Add to your website:

```html
<!-- Include the client library -->
<script src="https://cdn.jsdelivr.net/gh/moses-omondi/ai-recruitment-assistant@main/web/moses-ai-client.js"></script>

<!-- Add chat container -->
<div id="moses-ai-chat"></div>

<!-- Initialize the chat widget -->
<script>
new MosesAIChatWidget({
    container: 'moses-ai-chat',
    apiUrl: 'https://your-api-endpoint.com',
    theme: 'professional'
});
</script>
```

### Custom Implementation
```javascript
// Advanced usage with custom UI
const client = new MosesAIClient({
    apiUrl: 'https://your-api-endpoint.com',
    debug: false,
    onMessageComplete: (response) => {
        // Handle the AI response
        displayMessage(response, 'assistant');
    }
});

// Send a message
const response = await client.chat("Tell me about Moses's DevSecOps experience");
```

## 🔒 Security Configuration

### Production Security
```bash
# Enable API key authentication
API_KEY_REQUIRED=true
API_KEY=your-very-secure-api-key

# Configure CORS for your domains
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Enable rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
```

### HTTPS Setup
Use a reverse proxy (nginx) with SSL certificates:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 Monitoring & Logging

### Health Checks
The API provides health check endpoints:
- `GET /health` - Basic health status
- `GET /` - API information

### Monitoring Stack
Enable monitoring with:
```bash
docker-compose --profile monitoring up -d
```

Access:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Log Management
Logs are available via Docker:
```bash
# View live logs
docker-compose logs -f moses-ai-assistant

# Export logs
docker-compose logs moses-ai-assistant > moses-ai.log
```

## 🚨 Troubleshooting

### Common Issues

**Model Loading Fails**
```bash
# Check model files exist
ls -la models/moses-recruitment-assistant/

# Check GPU memory
nvidia-smi

# Check container logs
docker-compose logs moses-ai-assistant
```

**API Not Responding**
```bash
# Check if service is running
docker-compose ps

# Test health endpoint
curl http://localhost:8000/health

# Check port binding
netstat -tlnp | grep 8000
```

**Out of Memory**
```bash
# Reduce batch size or model precision
# Monitor GPU memory usage
watch -n 1 nvidia-smi

# Check container memory limits
docker stats moses-ai-assistant
```

### Performance Optimization

**GPU Optimization**
- Use appropriate GPU memory allocation
- Enable model quantization (4-bit)
- Optimize batch processing

**API Optimization**
- Enable caching with Redis
- Use connection pooling
- Implement request queuing

**Container Optimization**
- Use multi-stage builds
- Optimize image layers
- Enable health checks

## 📝 API Documentation

### Endpoints
- `POST /chat` - Chat with Moses's AI assistant
- `GET /capabilities` - Get expertise areas
- `GET /sample-questions` - Get sample questions
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

### Example Requests
```bash
# Chat request
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Tell me about Moses'\''s DevSecOps experience",
       "max_tokens": 400,
       "temperature": 0.7
     }'

# Get capabilities
curl http://localhost:8000/capabilities

# Health check
curl http://localhost:8000/health
```

## 🎯 Production Checklist

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] API keys configured
- [ ] CORS origins restricted
- [ ] Rate limiting enabled
- [ ] Health checks configured
- [ ] Logging configured
- [ ] Monitoring enabled
- [ ] Backup strategy implemented
- [ ] Load testing completed
- [ ] Security review completed

## 📞 Support

For deployment issues or questions:
- **GitHub Issues**: [Report issues](https://github.com/Moses-Omondi/ai-recruitment-assistant/issues)
- **Documentation**: Check this file and inline API docs
- **Model Issues**: Verify model files and GPU requirements

---

**Moses Omondi's AI Recruitment Assistant** - Ready for production deployment! 🚀
