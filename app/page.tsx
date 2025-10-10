export default function Home() {
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>PriceOptima Dashboard</h1>
      <p>Dynamic Pricing Analytics Application</p>
      <div style={{ marginTop: '20px' }}>
        <h2>Upload Your Data</h2>
        <input 
          type="file" 
          accept=".csv" 
          style={{ padding: '8px', margin: '10px 0' }}
        />
        <br />
        <button 
          style={{ 
            padding: '10px 20px', 
            backgroundColor: '#0070f3', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px',
            cursor: 'pointer',
            marginTop: '10px'
          }}
          onClick={() => alert('File upload functionality will connect to Render backend')}
        >
          Process Data
        </button>
      </div>
      <div style={{ marginTop: '30px' }}>
        <h3>Backend API Status</h3>
        <p>API URL: {process.env.NEXT_PUBLIC_API_URL || 'Not configured'}</p>
        <p>Status: <span style={{ color: 'green' }}>Ready</span></p>
        <p>Note: This is a static frontend. Backend processing happens on Render.</p>
      </div>
    </div>
  )
}