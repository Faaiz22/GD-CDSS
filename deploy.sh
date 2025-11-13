#!/bin/bash

# ============================================================================
# Gene-Drug CDSS v2 - Automated Deployment Script
# ============================================================================
# Usage: ./deploy.sh [git-lfs|external|build]
# ============================================================================

set -e  # Exit on any error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

echo -e "${GREEN}"
echo "=========================================="
echo "  Gene-Drug CDSS v2 Deployment Script"
echo "=========================================="
echo -e "${NC}"

# Check if streamlit_app.py exists at root
if [ ! -f "streamlit_app.py" ]; then
    echo -e "${RED}ERROR: streamlit_app.py not found at project root!${NC}"
    echo "This file is REQUIRED for Streamlit Cloud deployment."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p .streamlit
mkdir -p artifacts
mkdir -p data/raw

# Check for required files
REQUIRED_FILES=(
    "requirements.txt"
    "config/config.yaml"
    ".gitignore"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}ERROR: Required file missing: $file${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Found: $file${NC}"
done

# Handle deployment strategy
STRATEGY=${1:-"git-lfs"}

case $STRATEGY in
    "git-lfs")
        echo -e "${YELLOW}Using Git LFS for large files...${NC}"
        
        # Check if Git LFS is installed
        if ! command -v git-lfs &> /dev/null; then
            echo -e "${RED}ERROR: Git LFS not installed!${NC}"
            echo "Install: https://git-lfs.github.com/"
            exit 1
        fi
        
        # Initialize Git LFS
        git lfs install
        
        # Track large files
        git lfs track "artifacts/*.npy"
        git lfs track "artifacts/*.pt"
        git lfs track "artifacts/*.pth"
        git lfs track "artifacts/*.parquet"
        git lfs track "artifacts/*.pkl"
        
        # Add .gitattributes
        git add .gitattributes
        
        echo -e "${GREEN}✓ Git LFS configured${NC}"
        ;;
        
    "external")
        echo -e "${YELLOW}Using external storage (you must configure this manually)${NC}"
        echo "Edit src/utils/streamlit_helpers.py to add download logic"
        ;;
        
    "build")
        echo -e "${YELLOW}Building artifacts locally...${NC}"
        
        # Check if raw data exists
        if [ ! -f "data/raw/genes.tsv" ]; then
            echo -e "${RED}ERROR: Raw data not found in data/raw/${NC}"
            echo "Please add: genes.tsv, drugs.tsv, relationships.tsv, phytochemicals_new1.csv"
            exit 1
        fi
        
        # Install dependencies
        echo "Installing dependencies..."
        pip install -r requirements.txt
        
        # Run artifact generation
        echo "Generating artifacts (this may take 10-30 minutes)..."
        python scripts/01_build_artifacts.py
        python scripts/02_train_association_model.py
        python scripts/03_train_cvae.py
        
        echo -e "${GREEN}✓ Artifacts built${NC}"
        ;;
        
    *)
        echo -e "${RED}Invalid strategy: $STRATEGY${NC}"
        echo "Usage: ./deploy.sh [git-lfs|external|build]"
        exit 1
        ;;
esac

# Check artifact sizes
echo -e "${YELLOW}Checking artifact sizes...${NC}"

total_size=0
for file in artifacts/*; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        size_mb=$(du -m "$file" | cut -f1)
        total_size=$((total_size + size_mb))
        
        echo "  $file: $size"
        
        # Warn if file > 100MB and not using Git LFS
        if [ $size_mb -gt 100 ] && [ "$STRATEGY" != "git-lfs" ]; then
            echo -e "${RED}  WARNING: $file exceeds GitHub's 100MB limit!${NC}"
            echo "  Consider using Git LFS or external storage."
        fi
    fi
done

echo -e "${GREEN}Total artifacts size: ${total_size}MB${NC}"

# Configure secrets template
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo -e "${YELLOW}Creating secrets template...${NC}"
    cat > .streamlit/secrets.toml << 'EOF'
# IMPORTANT: Add your actual values before deploying

[ncbi]
email = "your.email@institution.edu"
EOF
    echo -e "${GREEN}✓ Created .streamlit/secrets.toml${NC}"
    echo -e "${RED}  ACTION REQUIRED: Edit this file with your actual NCBI email!${NC}"
fi

# Git setup
echo -e "${YELLOW}Preparing Git commit...${NC}"

# Stage all files
git add .

# Show status
echo -e "${YELLOW}Files to be committed:${NC}"
git status --short

# Commit
read -p "Commit message: " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Prepare for Streamlit Cloud deployment"
fi

git commit -m "$commit_msg" || true  # Don't fail if nothing to commit

# Check remote
if git remote | grep -q 'origin'; then
    echo -e "${GREEN}✓ Git remote configured${NC}"
    
    read -p "Push to GitHub now? (y/n): " push_now
    if [ "$push_now" = "y" ]; then
        git push origin main || git push origin master
        echo -e "${GREEN}✓ Pushed to GitHub${NC}"
    fi
else
    echo -e "${YELLOW}No Git remote configured.${NC}"
    echo "Add remote: git remote add origin https://github.com/yourusername/repo.git"
fi

# Final checklist
echo -e "${GREEN}"
echo "=========================================="
echo "  Deployment Preparation Complete!"
echo "=========================================="
echo -e "${NC}"

echo "Next steps:"
echo "1. Go to https://share.streamlit.io"
echo "2. Click 'New app'"
echo "3. Select your GitHub repository"
echo "4. Main file path: streamlit_app.py"
echo "5. Add secrets (copy from .streamlit/secrets.toml)"
echo "6. Click 'Deploy'"
echo ""
echo -e "${YELLOW}IMPORTANT: Edit .streamlit/secrets.toml with your NCBI email before deploying!${NC}"
echo ""
echo "Troubleshooting: See README_DEPLOYMENT.md"
