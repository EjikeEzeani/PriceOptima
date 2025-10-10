export default function Home() {
  return (
    <div style={{ padding: '2rem', fontFamily: 'Arial, sans-serif' }}>
      <h1>PriceOptima Dashboard</h1>
      <p>Dynamic Pricing Analytics Application</p>
      
      <div style={{ marginTop: '2rem', padding: '1rem', border: '1px solid #ddd', borderRadius: '8px' }}>
        <h2>Upload Your Data</h2>
        <input 
          type="file" 
          accept=".csv" 
          style={{ padding: '0.5rem', margin: '1rem 0', width: '100%' }}
        />
        <br />
        <button 
          style={{ 
            padding: '0.75rem 1.5rem', 
            backgroundColor: '#0070f3', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px',
            cursor: 'pointer',
            marginTop: '1rem'
          }}
        >
          Process Data
        </button>
      </div>
      
      <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
        <h3>Backend Status</h3>
        <p>API URL: {process.env.NEXT_PUBLIC_API_URL || 'Not configured'}</p>
        <p>Status: <span style={{ color: 'green' }}>Ready</span></p>
      </div>
    </div>
  )
}
