# Hotel Mapping Experiment

A simple application for uploading CSV files containing hotel data and mapping them to a standardized format.

## Project Structure

- `frontend/` - React application with Vite
- `backend/` - Express.js server

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
python server.py
```

Or using uvicorn directly:
```bash
uvicorn server:app --reload --port 3001
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

## GitHub Setup

To push this repository to GitHub:

1. Create a new repository on GitHub (e.g., `hotel-mapping-experiment`)
2. Add the remote and push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/hotel-mapping-experiment.git
git branch -M main
git push -u origin main
```

