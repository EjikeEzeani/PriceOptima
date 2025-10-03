# 🚀 Railway Deployment Guide

## Quick Deploy Steps

### 1. Go to Railway
- Visit: https://railway.app
- Click "Start a New Project"
- Sign up with GitHub

### 2. Deploy from GitHub
- Click "Deploy from GitHub repo"
- Select "PriceOptima" repository
- Railway will auto-detect Python and deploy

### 3. Get Your Backend URL
- After deployment, Railway will give you a URL like:
  - `https://priceoptima-production.up.railway.app`
- Copy this URL - you'll need it for Vercel!

### 4. Configure Vercel
- Go to your Vercel dashboard
- Select your PriceOptima project
- Go to Settings → Environment Variables
- Add:
  - **Name**: `NEXT_PUBLIC_API_URL`
  - **Value**: `https://your-railway-url.railway.app`
- Redeploy your Vercel app

## ✅ That's it! Your 404 issue will be fixed!

## 🔧 Troubleshooting

### If Railway deployment fails:
1. Check the logs in Railway dashboard
2. Make sure all dependencies are in `requirements.txt`
3. Verify the start command is correct

### If Vercel still shows 404:
1. Verify the `NEXT_PUBLIC_API_URL` is set correctly
2. Make sure the Railway URL is accessible
3. Redeploy Vercel after setting the environment variable

## 🎯 Test Your Deployment

1. **Test Backend**: Visit `https://your-railway-url.railway.app/health`
2. **Test Frontend**: Your Vercel app should now work!

## 📞 Need Help?

If you encounter any issues:
1. Check Railway logs
2. Check Vercel function logs
3. Verify environment variables are set
4. Test the backend URL directly in browser
