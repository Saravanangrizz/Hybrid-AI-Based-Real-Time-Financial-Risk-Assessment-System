import requests
import json
import time
import random
from typing import Dict, Any, List

# --- Configuration ---
# Your FastAPI server should run on this host and port:
API_BASE_URL = "http://127.0.0.1:8000" 
SCORING_ENDPOINT = f"{API_BASE_URL}/score_transaction"
# Must match the hardcoded secret in fraud_scoring_api.py for testing
API_AUTHORIZATION_TOKEN = "Bearer SECURE_API_KEY_12345"

# --- Simulated User Profiles Data used to generate realistic features for API call ---
USER_PROFILES_SIM = {
    "U123": {"geo": "New York", "device": "Mobile-A", "merchant": "groceries"},
    "U456": {"geo": "London", "device": "Desktop-B", "merchant": "e-commerce"},
    "U789": {"geo": "Tokyo", "device": "Tablet-C", "merchant": "utilities"},
    "U999": {"geo": "Paris", "device": "Mobile-D", "merchant": "entertainment"},
}
USER_IDS = list(USER_PROFILES_SIM.keys())
transaction_counter = 1000

def random_choice(arr):
    return random.choice(arr)

def random_float(min_val, max_val):
    return random.uniform(min_val, max_val)

def generate_transaction_data() -> Dict[str, Any]:
    """Generates mock transaction data simulating input from a payment system."""
    global transaction_counter
    transaction_counter += 1
    
    user_id = random_choice(USER_IDS)
    profile = USER_PROFILES_SIM[user_id]
    
    # Simulate a 35% chance of high-risk characteristics
    is_fraudulent = random.random() < 0.35 
    
    # Amounts: high if fraudulent and random.random() > 0.6, else low
    if is_fraudulent and random.random() > 0.6:
        amount = random_float(4000, 15000)
    else:
        amount = random_float(10, 800)
    
    # Generate features that will trigger rules/profiler/ML
    velocity_spike = is_fraudulent and random.random() > 0.4 
    geo_mismatch = is_fraudulent and random.random() > 0.3 
    
    location = random_choice(["Moscow", "Shanghai", "Dublin"]) if (geo_mismatch and random.random() > 0.5) else profile["geo"]
    device_id = random_choice(["New-E", "New-F"]) if random.random() < 0.15 else profile["device"]
    time_hour = random_choice([1, 2, 23]) if random.random() < 0.2 else random_choice([9, 10, 11, 15, 20])
    merchant_type = random_choice(["crypto", "vpn", "dating"]) if random.random() < 0.25 else profile["merchant"]

    return {
        "transaction_id": f"TXN_{transaction_counter}",
        "user_id": user_id,
        "amount": round(amount, 2),
        "velocity_spike": velocity_spike,
        "location": location,
        "device_id": device_id,
        "time_hour": time_hour,
        "merchant_type": merchant_type
    }

def send_test_request(txn_data: Dict[str, Any]):
    """Sends the transaction data to the local FastAPI endpoint."""
    
    print("-" * 50)
    print(f"Sending TXN {txn_data['transaction_id']} (User: {txn_data['user_id']}, Amount: ${txn_data['amount']})")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': API_AUTHORIZATION_TOKEN
    }
    
    try:
        start_time = time.perf_counter()
        response = requests.post(SCORING_ENDPOINT, headers=headers, json=txn_data, timeout=5)
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000

        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            score = result.get('final_risk_score', 0)
            decision = result.get('decision', 'UNKNOWN')
            
            print(f"DECISION: {decision} | Score: {score:.2f}% | Latency: {latency:.2f} ms")
            print("Audit Breakdown:")
            for key, value in result.get('audit_breakdown', {}).items():
                if isinstance(value, list) and value:
                    print(f"  - {key}: {'; '.join(value)}")
                elif not isinstance(value, (dict, list)):
                    print(f"  - {key}: {value}")
        else:
            print(f"API Error Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("\n!!! ERROR: Connection Refused. Please ensure the FastAPI server is running !!!")
        print(f"    Run: uvicorn fraud_scoring_api:app --reload --port 8000")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def run_local_simulation(num_requests=5):
    """Runs a series of simulated transactions through the local API."""
    print("Starting Local API Simulation...")
    print(f"Target Endpoint: {SCORING_ENDPOINT}")
    
    for i in range(num_requests):
        txn = generate_transaction_data()
        send_test_request(txn)
        time.sleep(1) # Wait 1 second between requests

if __name__ == "__main__":
    run_local_simulation(num_requests=10)