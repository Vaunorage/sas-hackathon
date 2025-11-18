"""
Test script for the Flask API
Demonstrates how to use the API endpoints
"""

import requests
import time
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:5000"
TEST_DATA_PATH = Path(__file__).parent / "data_in"

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_health_endpoints():
    """Test health and status endpoints"""
    print_section("Testing Health Endpoints")
    
    # Test welcome endpoint
    print("\n1. GET / - Welcome message")
    response = requests.get(f"{API_BASE_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    
    # Test ping endpoint
    print("\n2. GET /ping - Health check")
    response = requests.get(f"{API_BASE_URL}/ping")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    
    # Test ready endpoint
    print("\n3. GET /ready - Readiness probe")
    response = requests.get(f"{API_BASE_URL}/ready")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")

def test_create_job():
    """Test job creation with file uploads"""
    print_section("Testing Job Creation")
    
    # Check if test data exists
    if not TEST_DATA_PATH.exists():
        print(f"   WARNING: Test data path not found: {TEST_DATA_PATH}")
        print(f"   Creating mock job without files for testing...")
        
        # Create job without files (will fail but tests the endpoint)
        response = requests.post(
            f"{API_BASE_URL}/jobs",
            data={
                'nb_an_projection': 10,
                'nb_scenarios': 5,
                'max_accounts': 100
            }
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return None
    
    # Prepare files for upload
    print(f"   Loading test data from: {TEST_DATA_PATH}")
    required_files = [
        'POPULATION.csv',
        'MORTALITE.csv',
        'RENDEMENTS.csv',
        'DEPOTS_FUTURS.csv',
        'FRAIS_ADMIN.csv',
        'MIN_FERR.csv',
        'TX_LAPSE_PART.csv',
        'TX_LAPSE_TOT.csv',
        'ACQUISITION.csv',
        'COUSSINS_ESCAP.csv'
    ]
    
    files = []
    for filename in required_files:
        filepath = TEST_DATA_PATH / filename
        if filepath.exists():
            files.append(('files', (filename, open(filepath, 'rb'), 'text/csv')))
            print(f"   ✓ Found: {filename}")
        else:
            print(f"   ✗ Missing: {filename}")
    
    if not files:
        print("   ERROR: No CSV files found")
        return None
    
    # Create job
    print(f"\n   Creating job with {len(files)} files...")
    response = requests.post(
        f"{API_BASE_URL}/jobs",
        files=files,
        data={
            'nb_an_projection': 10,  # Small values for testing
            'nb_scenarios': 5,
            'max_accounts': 100
        }
    )
    
    # Close file handles
    for _, file_tuple in files:
        file_tuple[1].close()
    
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 201:
        job_data = response.json()
        print(f"   Response: {json.dumps(job_data, indent=2)}")
        return job_data['job_id']
    else:
        print(f"   Error: {response.text}")
        return None

def test_list_jobs():
    """Test listing all jobs"""
    print_section("Testing Job Listing")
    
    print("\n   GET /jobs - List all jobs")
    response = requests.get(f"{API_BASE_URL}/jobs")
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Total jobs: {data['count']}")
        
        if data['jobs']:
            print(f"\n   Most recent job:")
            job = data['jobs'][0]
            print(f"     Job ID: {job['job_id']}")
            print(f"     Status: {job['status']}")
            print(f"     Created: {job['created_at']}")
    else:
        print(f"   Error: {response.text}")

def test_get_job_details(job_id):
    """Test getting job details"""
    print_section(f"Testing Job Details - {job_id}")
    
    print(f"\n   GET /jobs/{job_id}")
    response = requests.get(f"{API_BASE_URL}/jobs/{job_id}")
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        job = response.json()
        print(f"   Job ID: {job['job_id']}")
        print(f"   Status: {job['status']}")
        print(f"   Created: {job['created_at']}")
        print(f"   Started: {job.get('started_at', 'N/A')}")
        print(f"   Completed: {job.get('completed_at', 'N/A')}")
        print(f"   Files uploaded: {len(job['uploaded_files'])}")
        print(f"   Result files: {len(job['result_files'])}")
        return job
    else:
        print(f"   Error: {response.text}")
        return None

def test_poll_job_status(job_id, max_wait=300, poll_interval=5):
    """Poll job status until completion"""
    print_section(f"Polling Job Status - {job_id}")
    
    print(f"\n   Polling every {poll_interval} seconds (max {max_wait}s)...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}")
        
        if response.status_code == 200:
            job = response.json()
            status = job['status']
            elapsed = int(time.time() - start_time)
            
            print(f"   [{elapsed:3d}s] Status: {status}")
            
            if status == 'completed':
                print(f"   ✓ Job completed successfully!")
                return True
            elif status == 'failed':
                print(f"   ✗ Job failed!")
                print(f"   Error: {job.get('error_message', 'Unknown error')}")
                return False
            
            time.sleep(poll_interval)
        else:
            print(f"   Error checking status: {response.text}")
            return False
    
    print(f"   Timeout after {max_wait} seconds")
    return False

def test_get_results(job_id):
    """Test getting job results"""
    print_section(f"Testing Result Retrieval - {job_id}")
    
    result_types = ['summary', 'detailed', 'internal']
    
    for result_type in result_types:
        print(f"\n   GET /jobs/{job_id}/results?type={result_type}")
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/results?type={result_type}")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Result type: {data['result_type']}")
            print(f"   Record count: {data['count']}")
            
            if data['data']:
                print(f"   First record keys: {list(data['data'][0].keys())}")
        else:
            print(f"   Error: {response.text}")

def test_list_files(job_id):
    """Test listing job files"""
    print_section(f"Testing File Listing - {job_id}")
    
    print(f"\n   GET /jobs/{job_id}/files")
    response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/files")
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Uploaded files: {len(data['uploaded_files'])}")
        print(f"   Result files: {len(data['result_files'])}")
        
        if data['uploaded_files']:
            print(f"\n   Uploaded files:")
            for file in data['uploaded_files']:
                print(f"     - {file['name']} ({file['size']} bytes)")
        
        if data['result_files']:
            print(f"\n   Result files:")
            for file in data['result_files']:
                print(f"     - {file['name']} ({file['size']} bytes)")
    else:
        print(f"   Error: {response.text}")

def test_download_file(job_id, filename):
    """Test downloading a file"""
    print_section(f"Testing File Download - {filename}")
    
    print(f"\n   GET /jobs/{job_id}/files/{filename}")
    response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/files/{filename}")
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        print(f"   File size: {len(response.content)} bytes")
        print(f"   Content-Type: {response.headers.get('Content-Type')}")
        
        # Preview first few lines if it's text
        try:
            lines = response.text.split('\n')[:5]
            print(f"\n   Preview (first 5 lines):")
            for line in lines:
                print(f"     {line[:100]}")
        except:
            print(f"   (Binary content)")
    else:
        print(f"   Error: {response.text}")

def run_all_tests():
    """Run all API tests"""
    print("\n" + "#" * 60)
    print("#  Flask API Test Suite")
    print("#" * 60)
    
    # Test health endpoints
    test_health_endpoints()
    
    # Test job creation
    job_id = test_create_job()
    
    # Test listing jobs
    test_list_jobs()
    
    if job_id:
        # Test job details
        test_get_job_details(job_id)
        
        # Poll for completion (with short timeout for testing)
        completed = test_poll_job_status(job_id, max_wait=60, poll_interval=5)
        
        if completed:
            # Test results
            test_get_results(job_id)
            
            # Test file listing
            test_list_files(job_id)
            
            # Test file download
            test_download_file(job_id, "VP_FLUX_TOTAL_GPU.csv")
    
    print_section("Test Suite Complete")

if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API")
        print(f"Make sure the Flask app is running at {API_BASE_URL}")
        print("\nTo start the Flask app, run:")
        print("  python flask_app.py")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
