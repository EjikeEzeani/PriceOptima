"use client";

import { useState } from "react";
import { uploadData, type UploadResponse } from "@/lib/api";

export default function FileUpload() {
  const [status, setStatus] = useState<string | null>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.type !== "text/csv") {
      setStatus("❌ Only CSV files are allowed.");
      return;
    }

    try {
      const data: UploadResponse = await uploadData(file);
      const columnNames = data.headers ?? (data.preview?.length ? Object.keys(data.preview[0]) : []);
      const rowsCount = data.totalRows ?? data.rows?.length ?? data.preview?.length ?? 0;

      setStatus(`✅ Uploaded. Rows: ${rowsCount}, Columns: ${columnNames.join(", ")}`);
    } catch (err: any) {
      setStatus(`❌ Error uploading dataset: ${err?.message || 'Check server connection.'}`);
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
