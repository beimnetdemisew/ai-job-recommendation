import React from "react";
import ResumeUpload from "./components/ResumeUpload"; // Fixed import statement

function App() {
  return (
    <div>
      <ResumeUpload /> {/* Removed .jsx extension from component usage */}
    </div>
  );
}

export default App;
