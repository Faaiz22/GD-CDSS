#!/bin/bash

# ============================================================================
# Quick Deploy Script for Gene-Drug CDSS v2
# Usage: ./deploy_quick.sh
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}"
echo "=========================================="
echo "  Gene-Drug CDSS v2 - Quick Setup"
echo "=========================================="
echo -e "${NC}"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python not found!${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"

# Check for Colab CSV
echo -e "\n${BLUE}Checking for data files...${NC}"

COLAB_CSV_FOUND=false
for csv_file in \
    "data/raw/Unified_Gene-Drug_Association_Protein_Features.csv" \
    "data/raw/Unified_Gene-Drug_Association_3D_Features.csv" \
    "data/raw/Unified_Gene-Drug_Association_with_Sequences.csv"; do
    
    if [ -f "$csv_file" ]; then
        echo -e "${GREEN}✓ Found: $csv_file${NC}"
        COLAB_CSV_FOUND=true
        break
    fi
done

# Check for TSV files
TSV_FOUND=false
if [ -f "data/raw/genes.tsv" ] && [ -f "data/raw/drugs.tsv" ]; then
    echo -e "${GREEN}✓ Found: PharmGKB TSV files${NC}"
    TSV_FOUND=true
fi

# Determine setup path
if [ "$COLAB_CSV_FOUND" = true ]; then
    echo -e "\n${GREEN}✨ Detected Colab CSV - Using fast setup!${NC}"
    SETUP_MODE="colab"
elif [ "$TSV_FOUND" = true ]; then
    echo -e "\n${YELLOW}⚠️  Using TSV files - Slower setup (15-30 min)${NC}"
    SETUP_MODE="tsv"
else
    echo -e "\n${RED}❌ No data files found!${NC}"
    echo ""
    echo "Please add ONE of the following to data/raw/:"
    echo "  1. Unified_Gene-Drug_Association_*.csv (from Colab)"
    echo "  2. genes.tsv + drugs.tsv + relationships.tsv (from PharmGKB)"
    echo ""
    exit 1
fi

# Install dependencies
echo -e "\n${BLUE}Installing dependencies...${NC}"
pip install -r requirements.txt -q

# Create directories
mkdir -p artifacts
mkdir -p .streamlit

# Generate artifacts
echo -e "\n${BLUE}Generating artifacts...${NC}"

if [ "$SETUP_MODE" = "colab" ]; then
    python scripts/00_generate_from_colab.py
else
    python scripts/01_build_artifacts.py
fi

# Check if artifacts were created
CORE_ARTIFACTS=(
    "artifacts/drug_library.npy"
    "artifacts/protein_library.npy"
    "artifacts/id_maps.json"
    "artifacts/association_dataset.pt"
)

ARTIFACTS_OK=true
for artifact in "${CORE_ARTIFACTS[@]}"; do
    if [ ! -f "$artifact" ]; then
        echo -e "${RED}❌ Missing: $artifact${NC}"
        ARTIFACTS_OK=false
    fi
done

if [ "$ARTIFACTS_OK" = false ]; then
    echo -e "\n${RED}❌ Artifact generation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Core artifacts generated${NC}"

# Train models
echo -e "\n${BLUE}Training association model (5-10 min)...${NC}"
python scripts/02_train_association_model.py

echo -e "\n${BLUE}Training C-VAE model (3-5 min)...${NC}"
python scripts/03_train_cvae.py

# Check models
if [ ! -f "artifacts/model.pt" ]; then
    echo -e "${RED}❌ Association model training failed!${NC}"
    exit 1
fi

if [ ! -f "artifacts/cvae_model.pt" ]; then
    echo -e "${RED}❌ C-VAE training failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Models trained${NC}"

# Create secrets template if needed
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo -e "\n${BLUE}Creating secrets template...${NC}"
    cat > .streamlit/secrets.toml << 'EOF'
[ncbi]
email = "your.email@institution.edu"
api_key = ""  # Optional: get from https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/
EOF
    echo -e "${YELLOW}⚠️  Please edit .streamlit/secrets.toml with your NCBI email!${NC}"
fi

# Summary
echo -e "\n${GREEN}"
echo "=========================================="
echo "  ✅ Setup Complete!"
echo "=========================================="
echo -e "${NC}"

echo "📦 Generated Artifacts:"
ls -lh artifacts/ | grep -E "\.(npy|pt|json|pkl|parquet)$" | awk '{print "   " $9 " (" $5 ")"}'

echo -e "\n🚀 Next Steps:"
echo "   1. Edit .streamlit/secrets.toml with your NCBI email"
echo "   2. Run: streamlit run streamlit_app.py"
echo "   3. Open browser: http://localhost:8501"

echo -e "\n📊 Test the system:"
echo "   - Go to 'Prediction & XAI' page"
echo "   - SMILES: CCO"
echo "   - Gene: TP53"
echo "   - Click 'Run Prediction'"

echo -e "\n${GREEN}✨ All done!${NC}"
