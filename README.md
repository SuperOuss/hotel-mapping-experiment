# Hotel Mapping Experiment

A simple application for uploading CSV files containing hotel data and mapping them to a standardized format.

## Project Structure

- `frontend/` - React application with Vite
- `backend/` - Express.js server

## Setup

### Backend

```bash
cd backend
npm install
npm start
```

The backend server will run on `http://localhost:3001`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will run on `http://localhost:3000`

## Usage

1. Start both the backend and frontend servers
2. Open `http://localhost:3000` in your browser
3. Upload a CSV file with hotel data
4. Map the CSV columns to the expected headers:
   - hotel_id
   - hotel_name
   - hotel_address
   - country_iso_code
   - latitude
   - longitude
5. Click "Process Hotels" to send the data to the backend

## CSV Format

The application expects CSV files with headers that can be mapped to:
- `hotel_id`
- `hotel_name`
- `hotel_address`
- `country_iso_code`
- `latitude`
- `longitude`

