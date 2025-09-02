#!/bin/bash

# Database Agent App Setup Script
# This script helps you set up the project with either uv or pip

set -e

echo "🚀 Database Agent App Setup"
echo "=========================="

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.copy .env
    echo "⚠️  IMPORTANT: Edit .env and add your OpenAI API key!"
    echo ""
fi

# Detect package manager preference
echo "Choose your Python package manager:"
echo "1) uv (recommended - faster)"
echo "2) pip (traditional)"
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo "🔧 Setting up with uv..."
        
        # Check if uv is installed
        if ! command -v uv &> /dev/null; then
            echo "❌ uv is not installed. Please install it first:"
            echo "   Visit: https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
        fi
        
        echo "📦 Creating virtual environment with uv..."
        uv venv
        
        echo "📥 Installing dependencies..."
        source .venv/bin/activate
        uv pip install -r requirements.txt
        
        ACTIVATE_CMD="source .venv/bin/activate"
        ;;
    2)
        echo "🔧 Setting up with pip..."
        
        echo "📦 Creating virtual environment..."
        python -m venv venv
        
        echo "📥 Installing dependencies..."
        source venv/bin/activate
        pip install -r requirements.txt
        
        ACTIVATE_CMD="source venv/bin/activate"
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. Start the database: docker-compose up -d"
echo "3. Activate environment: $ACTIVATE_CMD"
echo "4. Start the app: python app.py"
echo "5. Open http://localhost:8050 in your browser"
echo ""
echo "🎉 Happy querying!"