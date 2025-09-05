# Remote Access Guide - MacBook to Windows AI Server 🍎 ➡️ 🖥️

Complete guide for accessing your AI Recruitment Assistant from your MacBook at home.

## 🏠 Network Setup Overview

```
Your Home Network:
┌─────────────────┐    WiFi    ┌──────────────────┐
│   MacBook       │ ◄─────────► │ Windows PC       │
│   (Client)      │             │ (AI Server)      │
│ 192.168.1.X     │             │ 192.168.1.Y      │
└─────────────────┘             └──────────────────┘
                    │
                    ▼
              Router/WiFi
            (192.168.1.1)
```

## 🚀 Quick Start (5 Minutes)

### Step 1: Start the Server (Windows PC)
1. **Double-click** `start_server.bat` in the project folder
2. **Note the IP address** shown (e.g., `192.168.1.100`)
3. **Keep the window open** - server is running

### Step 2: Connect from MacBook
1. **Copy** `macbook_client.py` to your MacBook (via USB, email, or AirDrop)
2. **Install Python requests** (if needed): `pip3 install requests`
3. **Run the client**: `python3 macbook_client.py --server 192.168.1.100:8000`

That's it! 🎉

## 📋 Detailed Setup Instructions

### Windows PC (Server Side)

#### Option 1: Quick Start (Recommended)
```cmd
# Just double-click this file:
start_server.bat
```

#### Option 2: Manual Start
```cmd
# Navigate to project folder
cd C:\Users\wesle_b0bdufu\engineer-blog\ai-recruitment-assistant

# Start server
py src\deployment\api_server.py --host 0.0.0.0 --port 8000
```

#### Server will show:
```
========================================
 AI Recruitment Assistant Server
========================================

Server will be available at:
 - Local:  http://localhost:8000
 - Remote: http://192.168.1.100:8000  ← Use this IP on MacBook
 - Docs:   http://localhost:8000/docs

For MacBook access, use: http://192.168.1.100:8000
```

### MacBook (Client Side)

#### Option 1: Using Python Client (Recommended)
```bash
# Download the client script first, then:
python3 macbook_client.py --server 192.168.1.100:8000

# Interactive mode (default)
python3 macbook_client.py -s 192.168.1.100:8000 -m interactive

# Quick test mode
python3 macbook_client.py -s 192.168.1.100:8000 -m test

# Health check only  
python3 macbook_client.py -s 192.168.1.100:8000 -m health
```

#### Option 2: Using cURL (For testing)
```bash
# Health check
curl http://192.168.1.100:8000/health

# Generate response
curl -X POST http://192.168.1.100:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Schedule an interview with a candidate.",
    "input": "Candidate: John Doe\nPosition: Software Engineer"
  }'
```

#### Option 3: Using Web Browser
```
Open: http://192.168.1.100:8000/docs
```
This gives you an interactive web interface! 🌐

## 🔍 Finding Your Windows PC's IP Address

### Method 1: From the startup script
The `start_server.bat` automatically shows the IP address.

### Method 2: Command line
```cmd
ipconfig | findstr IPv4
```

### Method 3: Settings
1. Windows Settings → Network & Internet
2. Properties → IPv4 address

## 🛡️ Firewall Configuration

### Windows Firewall (If connection fails)

**Quick Fix:**
1. **Windows key + R** → `wf.msc`
2. **New Rule** → Port → TCP → 8000
3. **Allow the connection**

**Or disable temporarily for testing:**
```cmd
# Temporarily disable (run as admin)
netsh advfirewall set allprofiles state off

# Remember to re-enable later:
netsh advfirewall set allprofiles state on
```

### Alternative: Use Windows Defender UI
1. **Settings** → Update & Security → Windows Security
2. **Firewall & network protection**
3. **Allow an app** → Add `python.exe`

## 💻 MacBook Client Features

### Interactive Mode
```bash
python3 macbook_client.py --server 192.168.1.100:8000

# Example session:
📝 What do you need help with? Schedule an interview
📄 Additional context: Candidate: Sarah, Position: Engineer, Time: Tomorrow 2pm
🤖 Generating response...
⏱️  Generated in 2.3 seconds

✅ Generated Response:
────────────────────────────────────────
Dear Sarah,

Thank you for your interest in the Engineer position...
```

### Built-in Examples
Type `help` in interactive mode to see examples:
- Interview Scheduling
- Application Status Updates  
- Job Offers
- Job Descriptions
- And more...

## 🌐 Web Interface Access

Visit `http://192.168.1.100:8000/docs` in any browser for:
- **Interactive API documentation**
- **Try it out** buttons for all endpoints
- **Real-time testing** interface
- **Response examples**

Perfect for non-technical users! 🎯

## 🔧 API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Service information |
| `/health` | GET | Server health check |
| `/model-info` | GET | Model details |
| `/generate` | POST | Generate responses |
| `/batch-generate` | POST | Multiple requests |
| `/docs` | GET | Interactive documentation |

## 🚨 Troubleshooting

### "Cannot connect to server"
1. ✅ **Check server is running** (Windows PC)
2. ✅ **Verify IP address** (try `ipconfig` on Windows)
3. ✅ **Same WiFi network** (both devices)
4. ✅ **Firewall settings** (see section above)

### "Server health check failed"
1. ✅ **Model still loading** (wait a few minutes)
2. ✅ **Memory issues** (close other applications)
3. ✅ **Try restarting server**

### "Generation failed"
1. ✅ **Check instruction format** (see examples)
2. ✅ **Reduce max_length** if memory issues
3. ✅ **Try simpler prompts**

### Test Connection
```bash
# Simple connection test
curl -I http://192.168.1.100:8000/health

# Expected: HTTP/1.1 200 OK
```

## 🎯 Usage Examples

### 1. Interview Scheduling
```python
# On MacBook:
python3 macbook_client.py -s 192.168.1.100:8000

📝 What do you need help with? 
Schedule an interview with a Software Engineer candidate

📄 Additional context: 
Candidate: Alex Johnson
Position: Senior Software Engineer  
Available: Mon-Wed afternoons
Interview type: Technical + Behavioral
```

### 2. Job Offer
```python
📝 What do you need help with?
Send a job offer to a successful candidate

📄 Additional context:
Candidate: Maria Garcia
Position: Product Manager
Salary: $130k
Start date: March 15th
```

### 3. Application Status
```python
📝 What do you need help with?
Respond to a candidate asking about application status

📄 Additional context:
Applied 3 weeks ago for Data Scientist role
No previous communication
```

## 📱 Mobile Access

The API works on mobile browsers too!
- **iPhone Safari**: `http://192.168.1.100:8000/docs`
- **Android Chrome**: Same URL
- **Responsive interface** adapts to small screens

## 🔒 Security Notes

### Local Network Only
- Server only accessible on your home network
- Not exposed to the internet
- Safe for personal use

### For Public Access (Advanced)
If you want internet access, you'd need:
1. **Port forwarding** on router (security risk)
2. **VPN setup** (recommended)
3. **Cloud deployment** (most secure)

## ⚡ Performance Tips

### For Faster Responses
1. **Close unnecessary applications** on Windows PC
2. **Use shorter prompts** when possible
3. **Lower temperature** (0.3-0.5) for more focused responses
4. **Batch multiple requests** when possible

### For Better Quality
1. **Be specific** in instructions
2. **Provide context** in the input field
3. **Use examples** from the help menu
4. **Iterate and refine** prompts

## 🚀 What's Next?

Once you have remote access working:

1. **Train the full model** for better responses:
   ```cmd
   py src\training\train_recruitment_model.py
   ```

2. **Try different configurations**:
   ```cmd
   py src\training\train_recruitment_model.py --config configs/rtx4060_config.yaml
   ```

3. **Integrate with your workflow**:
   - Bookmark the web interface
   - Save common prompts
   - Create custom client scripts

---

## 🆘 Need Help?

**Connection Issues:**
- Check Windows PC IP: `ipconfig`
- Test from Windows first: `http://localhost:8000/health`
- Verify firewall: Temporarily disable to test

**Model Issues:**
- Try demo training: `py train_demo.py`
- Check GPU: `py check_gpu.py`
- Restart server if needed

**Want the full Llama model?**
- Run: `py src\training\train_recruitment_model.py`
- Takes 2-4 hours but much better quality!

**Happy recruiting with AI! 🤖✨**
