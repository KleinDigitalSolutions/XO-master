#!/bin/bash

# Klein Digital Solutions - Modal A10G Deployment Script

echo "🚀 Klein Digital Solutions - Music AI Separator"
echo "📡 Deploying to Modal with A10G GPU..."
echo ""

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    pip install modal
    echo "✅ Modal CLI installed"
fi

# Check if user is logged in
echo "🔐 Checking Modal authentication..."
if ! modal token current &> /dev/null; then
    echo "⚠️  Not logged in to Modal. Please run:"
    echo "   modal token new"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Modal authentication verified"
echo ""

# Deploy the application
echo "🚀 Deploying Music AI Separator to Modal A10G..."
modal deploy modal_app.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Deployment successful!"
    echo ""
    echo "📡 Your Modal endpoints:"
    echo "   • Main API: https://[your-app]--music-separator-api.modal.run"
    echo "   • Health Check: https://[your-app]--health-check.modal.run" 
    echo "   • Models Info: https://[your-app]--models-info.modal.run"
    echo ""
    echo "💡 Next steps:"
    echo "   1. Update API_BASE in static/index.html with your Modal URL"
    echo "   2. Test the deployment with a sample audio file"
    echo "   3. Set up custom domain for kleindigitalsolutions.de"
    echo "   4. Integrate Stripe for payments"
    echo ""
    echo "💰 A10G GPU Costs:"
    echo "   • ~$1.10/hour (pay-per-second)"
    echo "   • ~$0.014-$0.037 per track"
    echo "   • 99.4%+ profit margins"
    echo ""
    echo "🎵 Ready for professional music AI service!"
else
    echo ""
    echo "❌ Deployment failed. Please check the error messages above."
    echo "💡 Common issues:"
    echo "   • Check Modal authentication: modal token current"
    echo "   • Verify modal_app.py syntax"
    echo "   • Ensure all dependencies are correct"
fi