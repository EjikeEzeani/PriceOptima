# PriceOptima 2.0 Deployment Guide

## 🚀 Quick Deployment Steps

### 1. Deploy Backend to Railway (Recommended)

1. **Go to [Railway.app](https://railway.app)**
2. **Sign up/Login with GitHub**
3. **Create New Project → Deploy from GitHub**
4. **Select your PriceOptima repository**
5. **Configure the service:**
   - **Root Directory**: Leave empty (deploy from root)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api_backend:app --host 0.0.0.0 --port $PORT`
6. **Add Environment Variables:**
   - `PRICEOPTIMA_DEBUG=0`
7. **Deploy!** Railway will give you a URL like `https://your-app-name.railway.app`

### 2. Update Frontend Configuration

1. **Copy your Railway backend URL**
2. **In Vercel Dashboard:**
   - Go to your project settings
   - Add Environment Variable:
     - **Name**: `NEXT_PUBLIC_API_URL`
     - **Value**: `https://your-app-name.railway.app`
3. **Redeploy your Vercel app**

### 3. Alternative: Deploy Backend to Render

1. **Go to [Render.com](https://render.com)**
2. **Create New Web Service**
3. **Connect GitHub repository**
4. **Configure:**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api_backend:app --host 0.0.0.0 --port $PORT`
5. **Deploy and get URL**

## 🔧 Environment Variables

### Backend (Railway/Render)
```
PRICEOPTIMA_DEBUG=0
```

### Frontend (Vercel)
```
NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
```

## 📁 Files Added for Deployment

- `Procfile` - For Heroku/Railway deployment
- `requirements.txt` - Python dependencies
- Updated `next.config.mjs` - Environment-aware rewrites
- Updated `lib/api.ts` - Dynamic API URL configuration

## 🐛 Troubleshooting

### 404 Errors on Vercel
- ✅ Backend deployed and accessible
- ✅ `NEXT_PUBLIC_API_URL` set correctly in Vercel
- ✅ Backend URL is HTTPS (not HTTP)

### CORS Errors
- ✅ Backend has CORS enabled for your Vercel domain
- ✅ Frontend using correct backend URL

### Build Errors
- ✅ All dependencies in `requirements.txt`
- ✅ Python version compatible (3.8+)

## 🎯 Testing

1. **Test Backend**: Visit `https://your-backend-url.railway.app/health`
2. **Test Frontend**: Visit your Vercel URL
3. **Test Upload**: Try uploading a CSV file

## 📞 Support

If you encounter issues:
1. Check Railway/Render logs
2. Check Vercel function logs
3. Verify environment variables
4. Test backend URL directly
