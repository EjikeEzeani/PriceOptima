'use client'

import { useState, useEffect } from 'react'

export default function Home() {
  const [backendStatus, setBackendStatus] = useState('Checking...')
  const [apiUrl, setApiUrl] = useState('Not configured')
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    // Test backend connection - use environment variable or fallback
    const testBackend = async () => {
      try {
        const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8002'
        const response = await fetch(`${backendUrl}/health`)
        if (response.ok) {
          const data = await response.json()
          setBackendStatus('Connected')
          setApiUrl(backendUrl)
          setIsConnected(true)
        } else {
          setBackendStatus('Error')
          setIsConnected(false)
        }
      } catch (error) {
        setBackendStatus('Disconnected')
        setIsConnected(false)
      }
    }

    testBackend()
  }, [])

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      const formData = new FormData()
      formData.append('file', file)
      const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8002'

      const response = await fetch(`${backendUrl}/upload`, {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const result = await response.json()
        alert(`File uploaded successfully! Records: ${result.totalRows}`)
      } else {
        const error = await response.json()
        alert(`Upload failed: ${error.detail}`)
      }
    } catch (error) {
      alert(`Upload error: ${error}`)
    }
  }

  return (
    <div style={{ 
      minHeight: '100vh', 
      background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
      fontFamily: 'Arial, sans-serif'
    }}>
      {/* Header */}
      <header style={{
        background: 'white',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        borderBottom: '1px solid #e5e7eb'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 1rem' }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            padding: '1.5rem 0' 
          }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{
                width: '40px',
                height: '40px',
                background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginRight: '1rem'
              }}>
                <span style={{ color: 'white', fontWeight: 'bold', fontSize: '1.25rem' }}>P</span>
              </div>
              <div>
                <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#111827', margin: 0 }}>
                  PriceOptima
                </h1>
                <p style={{ fontSize: '0.875rem', color: '#6b7280', margin: 0 }}>
                  Dynamic Pricing Analytics
                </p>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{
                padding: '0.25rem 0.75rem',
                borderRadius: '9999px',
                fontSize: '0.875rem',
                fontWeight: '500',
                backgroundColor: isConnected ? '#dcfce7' : '#fef2f2',
                color: isConnected ? '#166534' : '#dc2626'
              }}>
                {backendStatus}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '3rem 1rem' }}>
        {/* Hero Section */}
        <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
          <h2 style={{ 
            fontSize: '2.25rem', 
            fontWeight: 'bold', 
            color: '#111827', 
            marginBottom: '1rem' 
          }}>
            Optimize Your Pricing Strategy
          </h2>
          <p style={{ 
            fontSize: '1.25rem', 
            color: '#4b5563', 
            maxWidth: '48rem', 
            margin: '0 auto' 
          }}>
            Leverage AI-powered analytics to maximize revenue, reduce waste, and enhance customer satisfaction through dynamic pricing optimization.
          </p>
        </div>

        {/* Upload Section */}
        <div style={{
          background: 'white',
          borderRadius: '1rem',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
          padding: '2rem',
          marginBottom: '3rem'
        }}>
          <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
            <h3 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#111827', marginBottom: '0.5rem' }}>
              Upload Your Data
            </h3>
            <p style={{ color: '#4b5563' }}>
              Upload a CSV file with product, price, quantity, revenue, and category columns
            </p>
          </div>

          <div style={{ maxWidth: '28rem', margin: '0 auto' }}>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              style={{
                width: '100%',
                padding: '0.75rem',
                border: '2px dashed #d1d5db',
                borderRadius: '0.5rem',
                background: '#f9fafb',
                cursor: 'pointer',
                fontSize: '0.875rem'
              }}
            />
          </div>
        </div>

        {/* Features Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '1.5rem',
          marginBottom: '3rem'
        }}>
          {[
            { icon: '📊', title: 'Data Analysis', desc: 'Comprehensive EDA with statistical insights and trend analysis' },
            { icon: '🤖', title: 'AI Models', desc: 'Machine learning models for predictive pricing and optimization' },
            { icon: '⚡', title: 'RL Simulation', desc: 'Reinforcement learning for dynamic pricing strategies' },
            { icon: '📄', title: 'Export Reports', desc: 'Generate comprehensive reports in multiple formats' },
            { icon: '📈', title: 'Real-time Analytics', desc: 'Live monitoring and optimization recommendations' },
            { icon: '🎯', title: 'Performance Metrics', desc: 'Track revenue growth and optimization effectiveness' }
          ].map((feature, index) => (
            <div key={index} style={{
              background: 'white',
              borderRadius: '0.75rem',
              boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
              padding: '1.5rem',
              transition: 'all 0.3s ease'
            }}>
              <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>{feature.icon}</div>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '600', color: '#111827', marginBottom: '0.5rem' }}>
                {feature.title}
              </h3>
              <p style={{ color: '#4b5563' }}>{feature.desc}</p>
            </div>
          ))}
        </div>

        {/* Status Section */}
        <div style={{
          background: 'white',
          borderRadius: '1rem',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
          padding: '2rem'
        }}>
          <h3 style={{ 
            fontSize: '1.5rem', 
            fontWeight: 'bold', 
            color: '#111827', 
            marginBottom: '1.5rem', 
            textAlign: 'center' 
          }}>
            System Status
          </h3>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '1.5rem'
          }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                padding: '0.5rem 1rem',
                borderRadius: '9999px',
                fontSize: '0.875rem',
                fontWeight: '500',
                backgroundColor: isConnected ? '#dcfce7' : '#fef2f2',
                color: isConnected ? '#166534' : '#dc2626',
                marginBottom: '0.5rem'
              }}>
                <div style={{
                  width: '0.5rem',
                  height: '0.5rem',
                  borderRadius: '50%',
                  backgroundColor: isConnected ? '#22c55e' : '#ef4444',
                  marginRight: '0.5rem'
                }}></div>
                Backend: {backendStatus}
              </div>
              <p style={{ fontSize: '0.875rem', color: '#6b7280' }}>{apiUrl}</p>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                padding: '0.5rem 1rem',
                borderRadius: '9999px',
                fontSize: '0.875rem',
                fontWeight: '500',
                backgroundColor: '#dbeafe',
                color: '#1e40af',
                marginBottom: '0.5rem'
              }}>
                <div style={{
                  width: '0.5rem',
                  height: '0.5rem',
                  borderRadius: '50%',
                  backgroundColor: '#3b82f6',
                  marginRight: '0.5rem'
                }}></div>
                Frontend: Ready
              </div>
              <p style={{ fontSize: '0.875rem', color: '#6b7280' }}>Vercel Deployment</p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={{
        background: '#111827',
        color: 'white',
        padding: '3rem 0',
        marginTop: '4rem'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 1rem', textAlign: 'center' }}>
          <h3 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem' }}>PriceOptima</h3>
          <p style={{ color: '#9ca3af', marginBottom: '1.5rem' }}>
            AI-Powered Dynamic Pricing Analytics Platform
          </p>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <span style={{ fontSize: '0.875rem', color: '#9ca3af' }}>
              © 2024 PriceOptima. All rights reserved.
            </span>
          </div>
        </div>
      </footer>
    </div>
  )
}