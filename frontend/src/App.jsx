import React, { useState } from 'react';
import './App.css';

const EXPECTED_HEADERS = [
  'hotel_id',
  'hotel_name',
  'hotel_address',
  'country_iso_code',
  'latitude',
  'longitude'
];

function App() {
  const [file, setFile] = useState(null);
  const [csvHeaders, setCsvHeaders] = useState([]);
  const [csvData, setCsvData] = useState([]);
  const [headerMapping, setHeaderMapping] = useState({});
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'text/csv') {
      setFile(selectedFile);
      setError(null);
      setResult(null);
      uploadAndParseCSV(selectedFile);
    } else {
      setError('Please select a valid CSV file');
    }
  };

  const uploadAndParseCSV = async (csvFile) => {
    try {
      const formData = new FormData();
      formData.append('csv', csvFile);

      const response = await fetch('http://localhost:3001/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload CSV');
      }

      const data = await response.json();
      setCsvHeaders(data.headers);
      setCsvData(data.sampleRecord ? [data.sampleRecord] : []);
      
      // Initialize mapping with empty selections
      const initialMapping = {};
      EXPECTED_HEADERS.forEach(header => {
        initialMapping[header] = '';
      });
      setHeaderMapping(initialMapping);
    } catch (err) {
      setError(err.message || 'Error uploading CSV');
      console.error('Upload error:', err);
    }
  };

  const handleMappingChange = (expectedHeader, csvHeader) => {
    setHeaderMapping(prev => ({
      ...prev,
      [expectedHeader]: csvHeader
    }));
  };

  const handleProcess = async () => {
    // Validate all mappings are set
    const missingMappings = EXPECTED_HEADERS.filter(
      header => !headerMapping[header]
    );

    if (missingMappings.length > 0) {
      setError(`Please map all required headers: ${missingMappings.join(', ')}`);
      return;
    }

    setProcessing(true);
    setError(null);

    try {
      // Send CSV file and header mapping to backend
      const formData = new FormData();
      formData.append('csv', file);
      formData.append('headerMapping', JSON.stringify(headerMapping));

      const response = await fetch('http://localhost:3001/api/process', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process data');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Error processing data');
      console.error('Process error:', err);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="app">
      <div className="container">
        <h1>Hotel Mapping Experiment</h1>
        
        <div className="upload-section">
          <h2>Upload CSV File</h2>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="file-input"
          />
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {csvHeaders.length > 0 && (
          <div className="mapping-section">
            <h2>Map CSV Headers</h2>
            <p className="instruction">
              Match each expected header with a column from your CSV file:
            </p>
            
            <div className="mapping-table">
              <div className="mapping-header">
                <div className="mapping-col">Expected Header</div>
                <div className="mapping-col">CSV Column</div>
              </div>
              
              {EXPECTED_HEADERS.map(expectedHeader => (
                <div key={expectedHeader} className="mapping-row">
                  <div className="mapping-col">
                    <strong>{expectedHeader}</strong>
                  </div>
                  <div className="mapping-col">
                    <select
                      value={headerMapping[expectedHeader] || ''}
                      onChange={(e) => handleMappingChange(expectedHeader, e.target.value)}
                      className="mapping-select"
                    >
                      <option value="">-- Select CSV Column --</option>
                      {csvHeaders.map(csvHeader => (
                        <option key={csvHeader} value={csvHeader}>
                          {csvHeader}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              ))}
            </div>

            <button
              onClick={handleProcess}
              disabled={processing}
              className="process-button"
            >
              {processing ? 'Processing...' : 'Process Hotels'}
            </button>
          </div>
        )}

        {result && (
          <div className="result-section">
            <h2>Processing Result</h2>
            <p>Successfully processed <strong>{result.count}</strong> hotels.</p>
            {result.hotels && result.hotels.length > 0 && (
              <div className="sample-data">
                <h3>Sample Hotel Data:</h3>
                <pre>{JSON.stringify(result.hotels[0], null, 2)}</pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

