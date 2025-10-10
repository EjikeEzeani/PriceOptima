export default function Home() {
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h1>PriceOptima Dashboard</h1>
      <p>Dynamic Pricing Analytics</p>
      <div>
        <h2>Upload Data</h2>
        <input type="file" accept=".csv" />
        <button style={{ marginLeft: '10px', padding: '5px 10px' }}>Process</button>
      </div>
      <p>API: {process.env.NEXT_PUBLIC_API_URL || 'Not configured'}</p>
    </div>
  )
}
