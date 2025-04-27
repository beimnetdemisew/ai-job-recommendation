import React, { useState } from "react";
import axios from "axios";
import JobMatches from "./JobMatches";

const ResumeUpload = () => {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [fileHash, setFileHash] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setStatus("");
    setFileHash("");
  };

  const handleUpload = async () => {
    if (!file) {
      setStatus("❌ Please select a file first.");
      return;
    }

    // Only allow PDFs and DOCX
    const validTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ];
    if (!validTypes.includes(file.type)) {
      setStatus("❌ Invalid file type. Upload PDF or DOCX only.");
      return;
    }

    const formData = new FormData();
    formData.append("resume", file);

    try {
      setIsUploading(true);
      setStatus("⏳ Uploading…");

      // **POST** to your upload_resume endpoint
      const { data } = await axios.post(
        "http://localhost:5000/process_resume",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      if (data.file_hash) {
        setStatus("✅ Uploaded! Got file hash.");
        setFileHash(data.file_hash);
      } else {
        setStatus("❌ No file_hash returned by server.");
      }
    } catch (err) {
      console.error(err);
      setStatus(
        "❌ Upload failed: " + (err.response?.data?.error || err.message)
      );
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>📤 Upload Resume</h2>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={isUploading}>
        {isUploading ? "Uploading…" : "Upload"}
      </button>
      <p>{status}</p>

      {fileHash && (
        <div style={{ marginTop: 20 }}>
          <p>
            🔑 <strong>File Hash:</strong> {fileHash}
          </p>
          <JobMatches fileHash={fileHash} />
        </div>
      )}
    </div>
  );
};

export default ResumeUpload;
