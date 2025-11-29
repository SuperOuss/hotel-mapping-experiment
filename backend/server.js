import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { parse } from 'csv-parse/sync';
import fs from 'fs';

const app = express();
const PORT = 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Configure multer for file uploads
const upload = multer({ dest: 'uploads/' });

// Ensure uploads directory exists
if (!fs.existsSync('uploads')) {
  fs.mkdirSync('uploads');
}

// Expected headers
const EXPECTED_HEADERS = [
  'hotel_id',
  'hotel_name',
  'hotel_address',
  'country_iso_code',
  'latitude',
  'longitude'
];

// Upload endpoint
app.post('/api/upload', upload.single('csv'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const filePath = req.file.path;
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    
    // Parse CSV
    const records = parse(fileContent, {
      columns: true,
      skip_empty_lines: true,
      trim: true
    });

    // Get headers from uploaded file
    const uploadedHeaders = Object.keys(records[0] || {});

    // Clean up uploaded file
    fs.unlinkSync(filePath);

    res.json({
      message: 'CSV parsed successfully',
      headers: uploadedHeaders,
      recordCount: records.length,
      sampleRecord: records[0] || null
    });
  } catch (error) {
    console.error('Error processing CSV:', error);
    res.status(500).json({ error: 'Error processing CSV file' });
  }
});

// Process mapped data endpoint
app.post('/api/process', upload.single('csv'), (req, res) => {
  try {
    const { headerMapping } = JSON.parse(req.body.headerMapping || '{}');
    
    if (!headerMapping || !req.file) {
      return res.status(400).json({ error: 'Missing headerMapping or CSV file' });
    }

    const filePath = req.file.path;
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    
    // Parse CSV
    const records = parse(fileContent, {
      columns: true,
      skip_empty_lines: true,
      trim: true
    });

    // Transform data based on header mapping
    const processedHotels = records.map(record => {
      const hotel = {};
      Object.keys(headerMapping).forEach(expectedHeader => {
        const mappedHeader = headerMapping[expectedHeader];
        hotel[expectedHeader] = record[mappedHeader] || null;
      });
      return hotel;
    });

    // Clean up uploaded file
    fs.unlinkSync(filePath);

    // For now, just return the processed data
    // Mapping logic will be added later
    res.json({
      message: 'Data processed successfully',
      hotels: processedHotels,
      count: processedHotels.length
    });
  } catch (error) {
    console.error('Error processing data:', error);
    res.status(500).json({ error: 'Error processing hotel data' });
  }
});

app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`);
});

