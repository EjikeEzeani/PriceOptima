"use client";

import { useState } from "react";

export default function FileUpload() {
  const [status, setStatus] = useState<string | null>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.type !== "text/csv") {
      setStatus("❌ Only CSV files are allowed.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/upload-dataset", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        setStatus(`❌ Upload failed: ${err.detail}`);
        return;
      }

      const data = await res.json();
      setStatus(
        `✅ Uploaded ${data.filename}. Rows: ${data.rows}, Columns: ${data.columns.join(", ")}`
      );
    } catch (err) {
      setStatus("❌ Error uploading dataset. Check server connection.");
    }
  };

  return (
    <div className="p-4 border rounded-lg shadow-md w-full max-w-md mx-auto">
      <h2 className="text-lg font-semibold mb-2">Upload Dataset (CSV)</h2>
      <input
        type="file"
        accept=".csv"
        onChange={handleFileUpload}
        className="block w-full border p-2 rounded"
      />
      {status && <p className="mt-2 text-sm">{status}</p>}
    </div>
  );
}
