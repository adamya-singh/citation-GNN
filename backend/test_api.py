import requests
import json
import time
import sys

# API base URL
BASE_URL = "http://localhost:8000"

def test_predict_link():
    """Test the /predict_link endpoint"""
    print("\nTesting /predict_link endpoint...")
    
    # Test with valid node indices
    print("\n1. Testing with valid node indices (0, 1)...")
    try:
        response = requests.post(
            f"{BASE_URL}/predict_link",
            json={"node1": 0, "node2": 1}
        )
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return False
    
    # Test with invalid node indices
    print("\n2. Testing with invalid node indices (999999, 999999)...")
    try:
        response = requests.post(
            f"{BASE_URL}/predict_link",
            json={"node1": 999999, "node2": 999999}
        )
        print(f"Status code: {response.status_code}")
        print(f"Error message: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return False
    
    return True

def test_graph():
    """Test the /graph endpoint"""
    print("\nTesting /graph endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/graph")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Number of nodes: {len(result['nodes'])}")
            print(f"Number of edges: {len(result['edges'])}")
            print("\nSample node:")
            print(json.dumps(list(result['nodes'].values())[0], indent=2))
            
            # Only print sample edge if edges exist
            if result['edges']:
                print("\nSample edge:")
                print(json.dumps(list(result['edges'].values())[0], indent=2))
            else:
                print("\nNo edges found in the graph subset")
        else:
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Starting API tests...")
    
    # Wait for the server to start
    print("Waiting for server to start...")
    time.sleep(2)
    
    # Run tests
    predict_link_success = test_predict_link()
    graph_success = test_graph()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Predict Link Test: {'✓' if predict_link_success else '✗'}")
    print(f"Graph Test: {'✓' if graph_success else '✗'}")
    
    # Exit with appropriate status code
    sys.exit(0 if (predict_link_success and graph_success) else 1)

if __name__ == "__main__":
    main() 