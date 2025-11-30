import React, { useState, useEffect, useRef } from 'react';
import Papa from 'papaparse';
import './App.css';

const EXPECTED_HEADERS = [
  'hotel_id',
  'hotel_name',
  'hotel_address',
  'country_iso_code',
  'latitude',
  'longitude'
];

// API base URL from environment variable (defaults to localhost for local testing)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001';

function App() {
  const [file, setFile] = useState(null);
  const [csvHeaders, setCsvHeaders] = useState([]);
  const [csvData, setCsvData] = useState([]);
  const [headerMapping, setHeaderMapping] = useState({});
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [timeRemaining, setTimeRemaining] = useState(null);
  const [autoMapped, setAutoMapped] = useState(false);
  const pollingIntervalRef = useRef(null);

  const validateHeaders = (uploadedHeaders) => {
    // Check that CSV has headers
    if (!uploadedHeaders || uploadedHeaders.length === 0) {
      return {
        valid: false,
        message: 'CSV file does not contain headers'
      };
    }
    
    // Check that all expected headers can potentially be mapped
    // (at least one header exists to map to)
    if (uploadedHeaders.length < EXPECTED_HEADERS.length) {
      return {
        valid: false,
        message: `CSV has ${uploadedHeaders.length} headers but ${EXPECTED_HEADERS.length} are required. Found: ${uploadedHeaders.join(', ')}`
      };
    }
    
    return { valid: true };
  };

  const parseCSVLocally = (csvFile) => {
    return new Promise((resolve, reject) => {
      Papa.parse(csvFile, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          if (results.errors.length > 0) {
            reject(new Error(`CSV parsing error: ${results.errors[0].message}`));
            return;
          }
          
          if (results.data.length === 0) {
            reject(new Error('CSV file is empty'));
            return;
          }
          
          const headers = results.meta.fields || [];
          const validation = validateHeaders(headers);
          
          if (!validation.valid) {
            reject(new Error(validation.message));
            return;
          }
          
          resolve({
            headers,
            data: results.data
          });
        },
        error: (error) => {
          reject(new Error(`Failed to parse CSV: ${error.message}`));
        }
      });
    });
  };

  // Normalize header for better matching
  const normalizeHeader = (header) => {
    if (!header) return '';
    
    let normalized = header.trim();
    
    // Handle camelCase and PascalCase by inserting spaces before capitals
    normalized = normalized.replace(/([a-z])([A-Z])/g, '$1 $2');
    
    // Now convert to lowercase
    normalized = normalized.toLowerCase();
    
    // Replace common separators and special chars with spaces
    normalized = normalized.replace(/[_\-\s\.]+/g, ' ');
    
    // Remove extra spaces
    normalized = normalized.replace(/\s+/g, ' ').trim();
    
    // Remove common prefixes/suffixes
    normalized = normalized.replace(/^(the|a|an)\s+/i, '');
    
    // Normalize common abbreviations and variations
    normalized = normalized
      .replace(/\bhotel\b/g, 'hotel')
      .replace(/\biso\b/g, 'iso')
      .replace(/\bcode\b/g, 'code')
      .replace(/\bcountry\b/g, 'country')
      .replace(/\baddress\b/g, 'address')
      .replace(/\bname\b/g, 'name')
      .replace(/\bid\b/g, 'id')
      .replace(/\bidentifier\b/g, 'id')
      .replace(/\blat\b|\blatitude\b/g, 'latitude')
      .replace(/\blng\b|\blon\b|\blongitude\b/g, 'longitude')
      .replace(/\bcoord\b|\bcoordinate\b/g, '')
      .replace(/\bgeo\b|\bgeographic\b/g, '');
    
    // Remove spaces for comparison
    normalized = normalized.replace(/\s/g, '');
    
    return normalized;
  };

  // Calculate similarity score between two normalized headers
  const calculateSimilarity = (normalizedExpected, normalizedCsv) => {
    if (normalizedExpected === normalizedCsv) return 1.0;
    
    // Exact substring match
    if (normalizedCsv.includes(normalizedExpected) || normalizedExpected.includes(normalizedCsv)) {
      const longer = Math.max(normalizedExpected.length, normalizedCsv.length);
      const shorter = Math.min(normalizedExpected.length, normalizedCsv.length);
      return shorter / longer;
    }
    
    // Check for key word matches
    const expectedWords = normalizedExpected.match(/.{2,}/g) || [];
    const csvWords = normalizedCsv.match(/.{2,}/g) || [];
    
    let matches = 0;
    expectedWords.forEach(word => {
      if (csvWords.some(csvWord => csvWord.includes(word) || word.includes(csvWord))) {
        matches++;
      }
    });
    
    if (matches === 0) return 0;
    return matches / Math.max(expectedWords.length, csvWords.length);
  };

  // Perform auto-mapping given CSV headers
  const performAutoMatch = (headersToMatch) => {
    const autoMapping = {};
    EXPECTED_HEADERS.forEach(expectedHeader => {
      const normalizedExpected = normalizeHeader(expectedHeader);
      let bestMatch = null;
      let bestScore = 0;
      
      headersToMatch.forEach(csvHeader => {
        const normalizedCsv = normalizeHeader(csvHeader);
        const score = calculateSimilarity(normalizedExpected, normalizedCsv);
        
        if (score > bestScore && score >= 0.5) { // Minimum threshold
          bestScore = score;
          bestMatch = csvHeader;
        }
      });
      
      if (bestMatch) {
        autoMapping[expectedHeader] = bestMatch;
      }
    });
    return autoMapping;
  };

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) {
      return;
    }
    
    // Check file type
    if (selectedFile.type !== 'text/csv' && !selectedFile.name.endsWith('.csv')) {
      setError('Please select a valid CSV file');
      setFile(null);
      setCsvHeaders([]);
      setCsvData([]);
      setHeaderMapping({});
      return;
    }
    
    setFile(selectedFile);
    setError(null);
    setResult(null);
    setJobId(null);
    setJobStatus(null);
    setAutoMapped(false);
    // Clear any existing polling
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    
    try {
      // Parse and validate CSV locally first
      const parsed = await parseCSVLocally(selectedFile);
      
      // Headers validated, proceed with mapping UI
      setCsvHeaders(parsed.headers);
      setCsvData(parsed.data.slice(0, 1)); // Store first row as sample
      
      // Automatically perform auto-mapping with normalized headers
      const autoMapping = performAutoMatch(parsed.headers);
      setHeaderMapping(autoMapping);
      setAutoMapped(true);
    } catch (err) {
      setError(err.message || 'Error parsing CSV file');
      setFile(null);
      setCsvHeaders([]);
      setCsvData([]);
      setHeaderMapping({});
      setAutoMapped(false);
      console.error('CSV validation error:', err);
    }
  };

  const uploadAndParseCSV = async (csvFile) => {
    try {
      const formData = new FormData();
      formData.append('csv', csvFile);

      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload CSV');
      }

      const data = await response.json();
      setCsvHeaders(data.headers);
      setCsvData(data.sampleRecord ? [data.sampleRecord] : []);
      
      // Automatically perform auto-mapping with normalized headers
      const autoMapping = performAutoMatch(data.headers);
      setHeaderMapping(autoMapping);
      setAutoMapped(true);
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

  const getAutoMatchSuggestion = (expectedHeader) => {
    const normalizedExpected = normalizeHeader(expectedHeader);
    
    // Find best match with similarity score
    let bestMatch = null;
    let bestScore = 0;
    
    csvHeaders.forEach(csvHeader => {
      const normalizedCsv = normalizeHeader(csvHeader);
      const score = calculateSimilarity(normalizedExpected, normalizedCsv);
      
      if (score > bestScore && score >= 0.5) { // Minimum threshold
        bestScore = score;
        bestMatch = csvHeader;
      }
    });
    
    return bestMatch;
  };

  const handleAutoMatch = () => {
    const autoMapping = performAutoMatch(csvHeaders);
    setHeaderMapping(autoMapping);
    setAutoMapped(true);
  };

  const getMappedHeaders = () => {
    return Object.values(headerMapping).filter(h => h !== '');
  };

  const isHeaderMapped = (csvHeader) => {
    return getMappedHeaders().includes(csvHeader);
  };

  const getMappingStatus = () => {
    const mapped = EXPECTED_HEADERS.filter(h => headerMapping[h]).length;
    return { mapped, total: EXPECTED_HEADERS.length };
  };

  const getProcessingButtonText = () => {
    if (!processing) return 'Process Hotels';
    
    if (jobStatus?.status === 'processing') {
      const processed = jobStatus.processed || 0;
      const total = jobStatus.total_hotels || 0;
      const percent = jobStatus.progress_percentage || 0;
      if (timeRemaining) {
        return `Processing ${processed}/${total} hotels (${percent}%) - ETA: ${timeRemaining}`;
      }
      return `Processing ${processed}/${total} hotels (${percent}%)`;
    }
    
    if (jobStatus?.status === 'pending') {
      if (timeRemaining) {
        return `Starting... (ETA: ${timeRemaining})`;
      }
      if (jobStatus.total_hotels) {
        return `Starting... (${jobStatus.total_hotels} hotels)`;
      }
      return 'Starting...';
    }
    
    return 'Processing...';
  };

  const formatTimeRemaining = (estimatedCompletion) => {
    if (!estimatedCompletion) return 'Calculating...';
    
    try {
      const now = new Date();
      const completion = new Date(estimatedCompletion);
      
      // Check if date is valid
      if (isNaN(completion.getTime())) {
        console.warn('Invalid estimated_completion date:', estimatedCompletion);
        return 'Calculating...';
      }
      
      const diff = completion - now;
      
      // If the estimated completion is in the past or very close (less than 1 second), show "Almost done"
      if (diff <= 1000) {
        return 'Almost done...';
      }
      
      const totalSeconds = Math.floor(diff / 1000);
      const minutes = Math.floor(totalSeconds / 60);
      const seconds = totalSeconds % 60;
      
      if (minutes > 0) {
        return `${minutes}m ${seconds}s`;
      }
      return `${seconds}s`;
    } catch (err) {
      console.error('Error formatting time remaining:', err, estimatedCompletion);
      return 'Calculating...';
    }
  };

  const formatSeconds = (seconds) => {
    if (seconds === undefined || seconds === null) return 'N/A';
    
    if (seconds < 1) {
      return `${(seconds * 1000).toFixed(0)}ms`;
    }
    
    if (seconds < 60) {
      return `${seconds.toFixed(2)}s`;
    }
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    if (minutes < 60) {
      if (remainingSeconds < 1) {
        return `${minutes}m`;
      }
      return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
    }
    
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    
    if (remainingMinutes === 0) {
      return `${hours}h`;
    }
    return `${hours}h ${remainingMinutes}m`;
  };

  const pollJobStatus = async (jobId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/job/${jobId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch job status');
      }
      const status = await response.json();
      console.log('Polling job status:', status);
      
      // Log processing time metrics if available
      if (status.status === 'completed') {
        console.log('Completed job status keys:', Object.keys(status));
        console.log('total_processing_time_seconds:', status.total_processing_time_seconds);
        console.log('time_per_hotel_seconds:', status.time_per_hotel_seconds);
        
        if (status.total_processing_time_seconds !== undefined && status.total_processing_time_seconds !== null) {
          console.log('Processing time metrics found:', {
            total_processing_time_seconds: status.total_processing_time_seconds,
            time_per_hotel_seconds: status.time_per_hotel_seconds
          });
        } else {
          console.warn('Processing time metrics NOT found in completed job status');
        }
      }
      
      setJobStatus(status);
      
      // Update live ETA
      if (status.estimated_completion) {
        const remaining = formatTimeRemaining(status.estimated_completion);
        setTimeRemaining(remaining);
      } else {
        console.warn('No estimated_completion in status response:', status);
      }
      
      // Stop polling if job is completed or failed
      if (status.status === 'completed' || status.status === 'failed') {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
        setProcessing(false);
        setTimeRemaining(null);
      }
    } catch (err) {
      console.error('Error polling job status:', err);
      setError(err.message || 'Error checking job status');
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      setProcessing(false);
      setTimeRemaining(null);
    }
  };

  // Update ETA countdown every second
  useEffect(() => {
    if (!jobStatus?.estimated_completion || jobStatus.status === 'completed' || jobStatus.status === 'failed') {
      return;
    }

    const updateETA = () => {
      const remaining = formatTimeRemaining(jobStatus.estimated_completion);
      setTimeRemaining(remaining);
    };

    // Update immediately
    updateETA();

    // Update every second for live countdown
    const etaInterval = setInterval(updateETA, 1000);

    return () => clearInterval(etaInterval);
  }, [jobStatus?.estimated_completion, jobStatus?.status]);

  // Poll job status when jobId is set (fallback if not started in handleProcess)
  useEffect(() => {
    if (jobId && !pollingIntervalRef.current) {
      // Poll immediately
      pollJobStatus(jobId);
      
      // Then poll every second
      pollingIntervalRef.current = setInterval(() => {
        pollJobStatus(jobId);
      }, 1000);
    }
    
    // Cleanup on unmount or when jobId changes
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [jobId]);

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
    setResult(null);
    setJobStatus(null);
    setJobId(null);
    setTimeRemaining(null);

    try {
      // Send CSV file and header mapping to backend
      const formData = new FormData();
      formData.append('csv', file);
      formData.append('headerMapping', JSON.stringify(headerMapping));

      const response = await fetch(`${API_BASE_URL}/api/process`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to create processing job');
      }

      const data = await response.json();
      console.log('Job created response:', data);
      const newJobId = data.job_id;
      setJobId(newJobId);
      setJobStatus({
        status: data.status,
        total_hotels: data.total_hotels,
        processed: 0,
        estimated_completion: data.estimated_completion
      });
      // Set initial ETA
      if (data.estimated_completion) {
        const remaining = formatTimeRemaining(data.estimated_completion);
        console.log('Calculated time remaining:', remaining);
        setTimeRemaining(remaining);
      } else {
        console.warn('No estimated_completion in response:', data);
      }
      
      // Start polling immediately, don't wait for useEffect
      if (newJobId && !pollingIntervalRef.current) {
        pollJobStatus(newJobId);
        pollingIntervalRef.current = setInterval(() => {
          pollJobStatus(newJobId);
        }, 1000);
      }
    } catch (err) {
      setError(err.message || 'Error processing data');
      console.error('Process error:', err);
      setProcessing(false);
    }
  };

  const handleDownload = () => {
    if (jobStatus?.download_url) {
        window.open(`${API_BASE_URL}${jobStatus.download_url}`, '_blank');
    }
  };

  return (
    <div className="app">
      <div className="container">
        <h1>Hotel Mapping Experiment</h1>
        
        {!processing && !jobStatus && (
          <>
            {csvHeaders.length === 0 && (
              <div className="csv-structure-section">
                <h2>Expected CSV Structure</h2>
                <p className="instruction">
                  Your CSV file should contain the following columns (column names can vary, but you'll map them):
                </p>
                <div className="csv-structure-table">
                  <div className="csv-structure-header-row">
                    <div className="csv-structure-col-header">Column Name</div>
                    <div className="csv-structure-col-header">Description</div>
                    <div className="csv-structure-col-header">Required</div>
                  </div>
                  {EXPECTED_HEADERS.map(header => {
                    const descriptions = {
                      'hotel_id': 'Unique identifier for the hotel',
                      'hotel_name': 'Name of the hotel',
                      'hotel_address': 'Street address of the hotel',
                      'country_iso_code': 'ISO country code (e.g., US, CA, GB)',
                      'latitude': 'Latitude coordinate (decimal)',
                      'longitude': 'Longitude coordinate (decimal)'
                    };
                    return (
                      <div key={header} className="csv-structure-row">
                        <div className="csv-structure-col">
                          <code>{header}</code>
                        </div>
                        <div className="csv-structure-col">
                          {descriptions[header] || 'N/A'}
                        </div>
                        <div className="csv-structure-col">
                          <span className="required-badge">Required</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            
            {csvHeaders.length === 0 && (
              <div className="upload-section">
                <h2>Upload CSV File</h2>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  className="file-input"
                />
              </div>
            )}
          </>
        )}

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {csvHeaders.length > 0 && !processing && !jobStatus && (
          <div className="mapping-section">
            {autoMapped && (
              <div className="auto-mapped-notice">
                ✓ Headers have been automatically mapped. Please review and adjust if needed.
              </div>
            )}
            <div className="mapping-header-section">
              <div>
                <h2>Map CSV Headers</h2>
                <p className="instruction">
                  Match each expected header with a column from your CSV file:
                </p>
              </div>
              <div className="mapping-actions">
                <div className="mapping-status">
                  {getMappingStatus().mapped} / {getMappingStatus().total} mapped
                </div>
              </div>
            </div>

            <div className="csv-headers-preview">
              <strong>CSV Headers:</strong> {csvHeaders.join(', ')}
            </div>
            
            <div className="mapping-table">
              <div className="mapping-header-row">
                <div className="mapping-col-header">Expected Header</div>
                <div className="mapping-col-header">CSV Column</div>
                <div className="mapping-col-header">Status</div>
              </div>
              
              {EXPECTED_HEADERS.map(expectedHeader => {
                const isMapped = !!headerMapping[expectedHeader];
                const autoSuggestion = getAutoMatchSuggestion(expectedHeader);
                return (
                  <div 
                    key={expectedHeader} 
                    className={`mapping-row ${isMapped ? 'mapped' : 'unmapped'}`}
                  >
                    <div className="mapping-col">
                      <strong>{expectedHeader}</strong>
                      {autoSuggestion && !isMapped && (
                        <span className="suggestion-hint" title={`Suggested: ${autoSuggestion}`}>
                          💡
                        </span>
                      )}
                    </div>
                    <div className="mapping-col">
                      <select
                        value={headerMapping[expectedHeader] || ''}
                        onChange={(e) => handleMappingChange(expectedHeader, e.target.value)}
                        className={`mapping-select ${isMapped ? 'mapped-select' : ''}`}
                      >
                        <option value="">-- Select CSV Column --</option>
                        {csvHeaders.map(csvHeader => {
                          const isUsed = isHeaderMapped(csvHeader) && csvHeader !== headerMapping[expectedHeader];
                          return (
                            <option 
                              key={csvHeader} 
                              value={csvHeader}
                              disabled={isUsed}
                              style={isUsed ? { color: '#999', fontStyle: 'italic' } : {}}
                            >
                              {csvHeader} {isUsed && '(already mapped)'}
                            </option>
                          );
                        })}
                      </select>
                    </div>
                    <div className="mapping-col status-col">
                      {isMapped ? (
                        <span className="status-badge mapped-badge">✓ Mapped</span>
                      ) : (
                        <span className="status-badge unmapped-badge">⚠ Required</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {csvData.length > 0 && csvData[0] && (
              <div className="sample-preview">
                <h3>Sample Data Preview</h3>
                <div className="preview-table">
                  <div className="preview-row">
                    {csvHeaders.slice(0, 5).map(header => (
                      <div key={header} className="preview-cell">
                        <strong>{header}</strong>
                        <div className="preview-value">
                          {csvData[0][header] || '(empty)'}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div className="process-section">
              <button
                onClick={handleProcess}
                disabled={processing || getMappingStatus().mapped !== getMappingStatus().total}
                className="process-button"
              >
                {getProcessingButtonText()}
              </button>
              {getMappingStatus().mapped !== getMappingStatus().total && (
                <p className="process-hint">
                  Please map all {getMappingStatus().total} required headers before processing.
                </p>
              )}
            </div>
          </div>
        )}

        {(processing || jobStatus) && (
          <div className="result-section">
            <h2>Processing Status</h2>
            
            {(!jobStatus || jobStatus.status === 'pending') && (
              <div className="job-status">
                <p><strong>✅ Job created. Starting processing...</strong></p>
                {jobStatus ? (
                  <>
                    <p><strong>📊 Total hotels to process:</strong> {jobStatus.total_hotels || 'Calculating...'}</p>
                    <p><strong>⏱️ Estimated time remaining:</strong> {timeRemaining || (jobStatus.estimated_completion ? formatTimeRemaining(jobStatus.estimated_completion) : 'Calculating...')}</p>
                    {jobStatus.estimated_completion && (
                      <p><strong>📅 Estimated completion:</strong> {new Date(jobStatus.estimated_completion).toLocaleString()}</p>
                    )}
                  </>
                ) : (
                  <p>Waiting for job details...</p>
                )}
              </div>
            )}
            
            {jobStatus?.status === 'processing' && (
              <div className="job-status">
                <p><strong>🔄 Processing hotels...</strong></p>
                <p><strong>📊 Total hotels:</strong> {jobStatus.total_hotels}</p>
                <div className="progress-container">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${jobStatus.progress_percentage || 0}%` }}
                    ></div>
                  </div>
                  <p className="progress-text">
                    <strong>{jobStatus.processed || 0} / {jobStatus.total_hotels}</strong> hotels processed 
                    <strong> ({jobStatus.progress_percentage || 0}%)</strong>
                  </p>
                </div>
                <div className="eta-info">
                  <p>
                    <strong>⏱️ Time remaining:</strong> {timeRemaining || 'Calculating...'}
                  </p>
                  {jobStatus.estimated_completion && (
                    <p>
                      <strong>📅 Estimated completion:</strong> {new Date(jobStatus.estimated_completion).toLocaleString()}
                    </p>
                  )}
                </div>
              </div>
            )}
            
            {jobStatus?.status === 'completed' && (
              <div className="job-status completed">
                <p className="success-message">✓ Processing completed successfully!</p>
                <p><strong>Total processed:</strong> {jobStatus.total_hotels} hotels</p>
                
                {/* Processing Time Information */}
                {(jobStatus.total_processing_time_seconds !== undefined && jobStatus.total_processing_time_seconds !== null) ? (
                  <div className="processing-time-info">
                    <p><strong>⏱️ Total processing time:</strong> {formatSeconds(jobStatus.total_processing_time_seconds)}</p>
                    {(jobStatus.time_per_hotel_seconds !== undefined && jobStatus.time_per_hotel_seconds !== null) && (
                      <p><strong>⚡ Average time per hotel:</strong> {formatSeconds(jobStatus.time_per_hotel_seconds)}</p>
                    )}
                  </div>
                ) : (
                  <div className="processing-time-info">
                    <p><em>Processing time information not available</em></p>
                  </div>
                )}
                
                {/* Coverage Information */}
                {jobStatus.matched_count !== undefined && jobStatus.matched_count !== null && (
                  <div className="coverage-info">
                    <p><strong>📈 Coverage:</strong> {jobStatus.matched_count} out of {jobStatus.total_hotels} hotels successfully mapped</p>
                    <p><strong>Coverage rate:</strong> {jobStatus.coverage_percentage || 0}%</p>
                  </div>
                )}
                
                <button 
                  onClick={handleDownload}
                  className="download-button"
                >
                  📥 Download Processed CSV
                </button>
              </div>
            )}
            
            {jobStatus?.status === 'failed' && (
              <div className="job-status failed">
                <p className="error-message">✗ Processing failed</p>
                {jobStatus.error && (
                  <p><strong>Error:</strong> {jobStatus.error}</p>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;


