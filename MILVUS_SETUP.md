# Milvus Setup Guide

This guide will help you resolve the Milvus connection error and get your application running.

## Problem
Your application is failing to start because it cannot connect to Milvus vector database at `localhost:19530`.

## Solutions

### Option 1: Run Milvus with Docker (Recommended)

1. **Install Docker and Docker Compose** (if not already installed):
   - Docker: https://docs.docker.com/get-docker/
   - Docker Compose: https://docs.docker.com/compose/install/

2. **Start Milvus using the provided Docker Compose file**:
   ```bash
   # Start Milvus (this will download and start all required containers)
   docker-compose up -d
   
   # Check if containers are running
   docker-compose ps
   ```

3. **Wait for Milvus to be ready** (usually takes 1-2 minutes):
   ```bash
   # You can use the setup script to automate this
   python setup_milvus.py
   ```

4. **Start your application**:
   ```bash
   uvicorn app.main:app --reload --port 5000
   ```

5. **To stop Milvus later**:
   ```bash
   docker-compose down
   ```

### Option 2: Use the Automated Setup Script

Run the setup script that will check prerequisites and start Milvus:

```bash
python setup_milvus.py
```

This script will:
- Check if Docker and Docker Compose are installed
- Start Milvus containers
- Wait for Milvus to be ready
- Verify the connection

### Option 3: Application Already Fixed (Graceful Degradation)

The application code has been modified to handle Milvus unavailability gracefully. Even if Milvus is not running, your application will now start successfully with the following behavior:

- **Application starts normally** even without Milvus
- **Vector search features are disabled** when Milvus is unavailable
- **Warning messages** are logged instead of crashing
- **Other features** continue to work normally

## Verification

After setting up Milvus, you can verify it's working:

1. **Check Milvus health**:
   ```bash
   curl http://localhost:9091/healthz
   ```

2. **Check if your application can connect**:
   ```bash
   python -c "from pymilvus import connections; connections.connect('default', 'localhost', '19530'); print('Milvus connection successful!')"
   ```

3. **Start your application**:
   ```bash
   uvicorn app.main:app --reload
   ```

## Milvus Management

### Useful Docker Commands

```bash
# View logs
docker-compose logs milvus

# Restart Milvus
docker-compose restart milvus

# Stop all services
docker-compose down

# Remove all data (careful!)
docker-compose down -v
```

### Accessing Milvus

- **Milvus Server**: `localhost:19530`
- **Milvus Health Check**: `http://localhost:9091/healthz`
- **MinIO Console** (object storage): `http://localhost:9001` (admin/minioadmin)

## Troubleshooting

### Port Already in Use
If you get port conflicts, you can modify the `docker-compose.yml` file to use different ports:

```yaml
ports:
  - "19531:19530"  # Change external port to 19531
```

Then update your application settings:
```bash
export MILVUS_PORT=19531
```

### Memory Issues
Milvus requires at least 4GB of RAM. If you have limited memory, you can:

1. Reduce the number of Milvus shards in the configuration
2. Use a lighter Milvus image
3. Increase Docker's memory allocation

### Connection Still Fails
If Milvus is running but connection still fails:

1. Check if Windows Firewall is blocking the connection
2. Verify Docker is running in the correct network mode
3. Try connecting from within the Docker network

## Environment Variables

You can customize Milvus connection settings using environment variables:

```bash
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
export MILVUS_COLLECTION_NAME=document_chunks
```

## Next Steps

Once Milvus is running:
1. Your application should start without errors
2. You can upload documents and they will be stored in the vector database
3. Vector search functionality will be available
4. RAG (Retrieval-Augmented Generation) features will work properly
