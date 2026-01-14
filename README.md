# ApexFlow API Backend

Premium forex signals API powered by intelligent market analysis.

## Quick Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Environment Variables Required:

| Variable | Description | Example |
|----------|-------------|---------|
| `MONGO_URL` | MongoDB Atlas connection string | `mongodb+srv://user:pass@cluster.mongodb.net/apexflow` |
| `DB_NAME` | Database name | `apexflow` |
| `DATA_MODE` | `demo` or `live` | `demo` |
| `SIGNAL_MODE` | `conservative`, `balanced`, or `aggressive` | `balanced` |

### Build & Start Commands:

```bash
# Build
pip install -r requirements-render.txt

# Start
uvicorn server:app --host 0.0.0.0 --port $PORT
```

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/signals/active` - Get active signals
- `GET /api/signals/history` - Get signal history
- `POST /api/demo/seed` - Reset demo data
- `POST /api/signals/{id}/resolve` - Resolve a signal
- `GET /api/analytics/equity-curve` - Get equity curve data

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from .env.example)
cp .env.example .env

# Run server
uvicorn server:app --reload --port 8001
```
