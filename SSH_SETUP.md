# SSH Access Setup - Monitor Training from MacBook 🖥️ ➡️ 🍎

Simple setup to SSH into your Windows PC from MacBook to monitor training progress.

## Quick SSH Setup (5 minutes)

### Step 1: Enable SSH on Windows
```powershell
# Run as Administrator in PowerShell:
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'
```

### Step 2: Configure Firewall
```powershell
# Run as Administrator:
New-NetFirewallRule -Name sshd -DisplayName 'SSH Server' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

### Step 3: Find Your IP Address
```cmd
ipconfig | findstr IPv4
# Note the IP address (e.g., 192.168.1.100)
```

### Step 4: Connect from MacBook
```bash
# From your MacBook:
ssh wesle_b0bdufu@192.168.1.100

# Navigate to project:
cd "C:\Users\wesle_b0bdufu\engineer-blog\ai-recruitment-assistant"

# Monitor training:
Get-Content logs\training.log -Wait
```

## Training Commands

### Start Full Training (RTX 4060 Optimized)
```cmd
py src\training\train_recruitment_model.py --config configs/rtx4060_config.yaml
```

### Monitor Progress
```cmd
# Watch training logs:
Get-Content logs\training.log -Wait

# Check GPU usage:
nvidia-smi -l 1

# View training status:
py check_gpu.py
```

## Alternative: Use VS Code Remote SSH
1. Install "Remote - SSH" extension in VS Code
2. Connect to `wesle_b0bdufu@192.168.1.100`
3. Open project folder
4. Terminal → Monitor training progress

That's it! Much simpler. 🎯
