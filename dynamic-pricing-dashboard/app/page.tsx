import React from 'react';

export default function Home() {
  return (
    <div style={{ padding: '2rem', fontFamily: 'Arial, sans-serif' }}>
      <h1>PriceOptima Dashboard</h1>
      <p>Dynamic Pricing Analytics Application</p>
      <div style={{ marginTop: '2rem' }}>
        <h2>Upload Your Data</h2>
        <input 
          type="file" 
          accept=".csv" 
          style={{ padding: '0.5rem', margin: '1rem 0' }}
        />
        <br />
        <button 
          style={{ 
            padding: '0.75rem 1.5rem', 
            backgroundColor: '#0070f3', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Process Data
        </button>
      </div>
      <div style={{ marginTop: '2rem' }}>
        <h3>Backend API</h3>
        <p>Connect to your Render backend for data processing</p>
        <p>API URL: <code>{process.env.NEXT_PUBLIC_API_URL || 'Not configured'}</code></p>
      </div>
    </div>
  );
}