import React, { useEffect, useState } from "react";
import axios from "axios";

const JobMatching = ({ fileHash }) => {
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchJobMatches = async () => {
      if (!fileHash) {
        setError("No file hash available");
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const response = await axios.get(
          `http://localhost:5000/match_jobs/${fileHash}`
        );

        if (response.data && response.data.matches) {
          setMatches(response.data.matches);
        } else {
          setError("No matches found");
        }
      } catch (err) {
        setError("Failed to fetch job matches");
        console.error("Job match fetch error:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchJobMatches();
  }, [fileHash]);

  if (loading) {
    return <p>Loading job matches...</p>;
  }

  if (error) {
    return <p>{error}</p>;
  }

  return (
    <div>
      {matches.length > 0 ? (
        <div>
          <h3 className="text-lg font-semibold mb-4">Job Matches</h3>
          <ul className="list-disc ml-6">
            {matches.map((job, index) => (
              <li key={index} className="mb-2">
                <strong>{job.title}</strong> — Score: {job.match_score}% (
                {job.relevance})
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <p>No job matches found.</p>
      )}
    </div>
  );
};

export default JobMatching;
