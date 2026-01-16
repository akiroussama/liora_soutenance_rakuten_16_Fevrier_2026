#!/bin/bash
# =============================================================================
# Script de lancement de l'application Streamlit Rakuten
# =============================================================================

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🛒 Rakuten Product Classifier - Application Streamlit${NC}"
echo "================================================================"

# Aller au répertoire du projet
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}📁 Répertoire du projet: $PROJECT_ROOT${NC}"

# Vérifier si l'environnement virtuel existe
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Environnement virtuel trouvé${NC}"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${GREEN}✓ Environnement virtuel trouvé (.venv)${NC}"
    source .venv/bin/activate
else
    echo -e "${YELLOW}⚠ Pas d'environnement virtuel détecté${NC}"
    echo -e "${YELLOW}  Création d'un environnement virtuel...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}✓ Environnement virtuel créé${NC}"
fi

# Vérifier les dépendances
echo -e "${YELLOW}📦 Vérification des dépendances...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dépendances installées${NC}"
else
    echo -e "${RED}✗ Erreur lors de l'installation des dépendances${NC}"
    exit 1
fi

# Lancer l'application
echo ""
echo -e "${GREEN}🚀 Lancement de l'application...${NC}"
echo "================================================================"
echo -e "${YELLOW}📌 Ouvrez votre navigateur à l'adresse affichée ci-dessous${NC}"
echo ""

cd src/streamlit
streamlit run app.py --server.port 8501 --server.headless true
