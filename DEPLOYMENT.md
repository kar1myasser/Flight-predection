# 🚀 Deployment Guide - Flight Price Predictor

## Prerequisites

Before deploying, ensure you have:
- ✅ GitHub account
- ✅ Streamlit Cloud account (free at streamlit.io/cloud)
- ✅ Your trained models saved as `best_model.pkl` and `scaler.pkl`

## Step 1: Prepare Your Files

Ensure your project has these files:
```
DEPI/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── best_model.pkl        # Your trained model (IMPORTANT!)
├── scaler.pkl            # Your StandardScaler (IMPORTANT!)
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── DEPLOYMENT.md         # This file
└── README.md             # Project documentation
```

## Step 2: Save Your Best Model and Scaler

Run this in your Jupyter notebook (`final project[1].ipynb`) to save the model:

```python
import pickle

# Save your best model (Random Forest - the best performer)
with open('best_model.pkl', 'wb') as f:
    pickle.dump(RF_model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")
```

**Important:** Run this after training all your models to save the Random Forest model.

## Step 3: Test Locally First

```bash
# Navigate to DEPI folder
cd "c:\Users\kar1m\Desktop\Workspace\DEPI"

# Activate your virtual environment
& C:\Users\kar1m\Desktop\Workspace\my_env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Visit: http://localhost:8501

Test all features before deploying!

## Step 4: Push to GitHub

```bash
# Initialize git in DEPI folder (if not already done)
cd "c:\Users\kar1m\Desktop\Workspace\DEPI"
git init

# Add all files
git add .

# Commit
git commit -m "Add Streamlit deployment files for flight price predictor"

# Create repository on GitHub and push
git remote add origin https://github.com/YOUR_USERNAME/flight-price-predictor.git
git branch -M main
git push -u origin main
```

## Step 5: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your GitHub repository
5. Set:
   - **Branch:** main
   - **Main file path:** app.py
   - **Python version:** 3.11
6. Click **"Deploy!"**

Your app will be live in 2-3 minutes! 🎉

## Step 6: Test Your Deployment

Once deployed, test all features:
- ✅ Model loads correctly
- ✅ All input fields work
- ✅ Predictions are generated
- ✅ UI displays properly
- ✅ Price conversions are accurate

## Alternative Deployment Options

### Option 1: Hugging Face Spaces

1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space (select Streamlit)
3. Upload all files from DEPI folder
4. Auto-deploys automatically

**Pros:** Free, easy, integrated with ML community

### Option 2: Render

1. Create account at [render.com](https://render.com)
2. New Web Service
3. Connect GitHub repo
4. Set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Deploy

**Pros:** More control, custom domains, better performance

### Option 3: Heroku

```bash
# Install Heroku CLI
# Create Procfile
echo "web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0" > Procfile

# Create setup.sh
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > setup.sh

# Deploy
heroku create your-flight-predictor
git push heroku main
```

**Pros:** Professional, scalable, database support

## Troubleshooting

### ❌ Model Not Found Error

**Solution:**
```python
# In your notebook, ensure you're in the right directory
import os
os.chdir('c:/Users/kar1m/Desktop/Workspace/DEPI')

# Then save the models
import pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(RF_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

### ❌ Dependencies Error

**Solution:**
```bash
# Update requirements.txt with exact versions
pip freeze > requirements.txt

# Or use compatible versions
pip install streamlit pandas numpy scikit-learn
```

### ❌ Memory Issues on Streamlit Cloud

**Solution:**
- Streamlit Cloud free tier: 1GB RAM
- Optimize model size
- Use model compression if needed

### ❌ Encoding Issues

**Solution:**
Ensure the encoding in `app.py` matches your training:
```python
# Check your training encoding
print(le.classes_)  # In your notebook

# Update mappings in app.py accordingly
```

## Important Notes

### 🔥 Model File Size

- **Streamlit Cloud free tier:** 1GB RAM, 1GB storage
- **If model > 100MB:** Use Git LFS

```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add best_model.pkl scaler.pkl
git commit -m "Add model files with LFS"
git push
```

### 🔒 Security Best Practices

- ✅ Don't commit API keys or secrets
- ✅ Use environment variables for sensitive data
- ✅ Validate all user inputs
- ✅ Set reasonable input limits

### 📊 Monitoring

- Check app logs in Streamlit Cloud dashboard
- Monitor response times
- Track user interactions (optional analytics)
- Set up error notifications

## Environment Variables (Optional)

If you need to store secrets:

1. In Streamlit Cloud: Settings → Secrets
2. Add secrets in TOML format:
```toml
[secrets]
api_key = "your_key_here"
```

3. Access in code:
```python
import streamlit as st
api_key = st.secrets["api_key"]
```

## Custom Domain (Optional)

1. Go to Streamlit Cloud settings
2. Click "Custom domain"
3. Add your domain (e.g., flightprices.yourdomain.com)
4. Update DNS records as instructed
5. SSL automatically configured

## Performance Optimization

### Caching
```python
@st.cache_resource  # Already implemented for model loading
def load_model():
    # Model loading code
    pass
```

### Model Optimization
- Use model compression techniques
- Consider using ONNX for faster inference
- Reduce model complexity if needed

## Updating Your App

To update your deployed app:

```bash
# Make changes to your code
git add .
git commit -m "Update: description of changes"
git push

# Streamlit Cloud auto-redeploys!
```

## Support & Resources

- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum:** [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues:** Your repo issues page
- **Streamlit Gallery:** [streamlit.io/gallery](https://streamlit.io/gallery)

## Cost Breakdown

### Free Tier (Recommended for this project)
- **Streamlit Cloud:** Free, unlimited public apps
- **GitHub:** Free for public repos
- **Hugging Face:** Free Spaces

### Paid Options (For production)
- **Streamlit Cloud Teams:** $250/month (private apps, more resources)
- **Render:** $7/month (starter plan)
- **Heroku:** $7/month (hobby tier)

---

## Quick Start Checklist

- [ ] Save `best_model.pkl` and `scaler.pkl` from notebook
- [ ] Test locally: `streamlit run app.py`
- [ ] Create GitHub repository
- [ ] Push code to GitHub
- [ ] Sign up for Streamlit Cloud
- [ ] Deploy app
- [ ] Test deployed app
- [ ] Share the URL!

---

## Your App URL Format

**Streamlit Cloud:**
```
https://YOUR_USERNAME-flight-price-predictor-RANDOM.streamlit.app
```

**Example:**
```
https://kar1m-flight-price-predictor-x7k9m2.streamlit.app
```

---

## 🎉 Congratulations!

Your ML model is now deployed and accessible worldwide!

Share your app URL with friends, add it to your portfolio, and showcase your data science skills!

---

**Need Help?**

If you encounter any issues:
1. Check the troubleshooting section above
2. Review Streamlit Cloud logs
3. Verify all files are committed to GitHub
4. Ensure model files are in the same directory as app.py

**Happy Deploying! 🚀**
