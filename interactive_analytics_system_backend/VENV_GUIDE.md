# Virtual Environment Guide

## Understanding Virtual Environments

**System Python** (what you have now):
- When you run `python3` or `pip` directly → installs to your **system**
- All projects share the same packages
- Can cause conflicts between projects

**Virtual Environment** (recommended):
- Isolated Python environment for this project
- Packages installed in `venv/` folder
- Each project has its own dependencies

## Quick Setup

### Option 1: Use the setup script
```bash
./setup_venv.sh
```

### Option 2: Manual setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate it (you MUST do this before installing/running)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## How to Tell if You're in a Virtual Environment

**When activated**, your terminal prompt will show `(venv)`:
```bash
(venv) user@computer:~/project$
```

**Check manually:**
```bash
echo $VIRTUAL_ENV
# If it shows a path, you're in a venv
# If it's empty, you're using system Python
```

## Daily Usage

**Always activate the venv before working:**
```bash
source venv/bin/activate
```

**Then run your commands:**
```bash
python3 app.py          # Uses venv Python
pip install something    # Installs to venv
streamlit run app.py     # Uses venv packages
```

**Or use the start script** (it activates venv automatically):
```bash
./start.sh
```

**To deactivate:**
```bash
deactivate
```

## Troubleshooting

**"Module not found" errors:**
- Make sure venv is activated: `source venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

**Packages installed but not found:**
- You installed to system Python, not venv
- Activate venv and reinstall

**Which Python am I using?**
```bash
which python3    # Shows path - should include 'venv' if activated
which pip        # Should also include 'venv'
```
