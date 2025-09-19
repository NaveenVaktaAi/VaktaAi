#!/usr/bin/env python3
"""
Setup script for Milvus vector database
This script helps you set up Milvus for the vakta-ai-backend application
"""

import subprocess
import sys
import time
import requests
from pymilvus import connections, utility
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Docker found: {result.stdout.strip()}")
            return True
        else:
            logger.error("Docker is not installed or not in PATH")
            return False
    except FileNotFoundError:
        logger.error("Docker is not installed")
        return False

def check_docker_compose():
    """Check if Docker Compose is installed"""
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Docker Compose found: {result.stdout.strip()}")
            return True
        else:
            # Try docker compose (newer version)
            result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Docker Compose found: {result.stdout.strip()}")
                return True
            else:
                logger.error("Docker Compose is not installed")
                return False
    except FileNotFoundError:
        logger.error("Docker Compose is not installed")
        return False

def start_milvus():
    """Start Milvus using Docker Compose"""
    try:
        logger.info("Starting Milvus with Docker Compose...")
        
        # Try docker-compose first, then docker compose
        try:
            result = subprocess.run(['docker-compose', 'up', '-d'], capture_output=True, text=True)
        except FileNotFoundError:
            result = subprocess.run(['docker', 'compose', 'up', '-d'], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Milvus containers started successfully")
            return True
        else:
            logger.error(f"Failed to start Milvus: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error starting Milvus: {e}")
        return False

def wait_for_milvus(host="localhost", port=19530, timeout=120):
    """Wait for Milvus to be ready"""
    logger.info(f"Waiting for Milvus to be ready at {host}:{port}...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Try to connect to Milvus
            connections.connect(
                alias="test",
                host=host,
                port=port
            )
            
            # Test if we can list collections
            collections = utility.list_collections(using="test")
            connections.disconnect("test")
            
            logger.info("Milvus is ready!")
            return True
            
        except Exception as e:
            logger.debug(f"Milvus not ready yet: {e}")
            time.sleep(5)
    
    logger.error(f"Milvus did not become ready within {timeout} seconds")
    return False

def check_milvus_health():
    """Check Milvus health endpoint"""
    try:
        response = requests.get("http://localhost:9091/healthz", timeout=5)
        if response.status_code == 200:
            logger.info("Milvus health check passed")
            return True
        else:
            logger.warning(f"Milvus health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"Could not check Milvus health: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("=== Milvus Setup Script ===")
    
    # Check prerequisites
    if not check_docker():
        logger.error("Please install Docker first: https://docs.docker.com/get-docker/")
        sys.exit(1)
    
    if not check_docker_compose():
        logger.error("Please install Docker Compose first: https://docs.docker.com/compose/install/")
        sys.exit(1)
    
    # Check if docker-compose.yml exists
    try:
        with open('docker-compose.yml', 'r') as f:
            logger.info("Found docker-compose.yml")
    except FileNotFoundError:
        logger.error("docker-compose.yml not found. Please make sure it exists in the current directory.")
        sys.exit(1)
    
    # Start Milvus
    if not start_milvus():
        logger.error("Failed to start Milvus")
        sys.exit(1)
    
    # Wait for Milvus to be ready
    if not wait_for_milvus():
        logger.error("Milvus did not start properly")
        sys.exit(1)
    
    # Check health
    check_milvus_health()
    
    logger.info("=== Setup Complete ===")
    logger.info("Milvus is now running and ready!")
    logger.info("You can now start your application with: uvicorn app.main:app --reload")
    logger.info("To stop Milvus later, run: docker-compose down")

if __name__ == "__main__":
    main()
