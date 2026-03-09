import time
import random
import joblib
import pandas as pd
import numpy as np
import json 
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Header, Body
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware 
from fastapi import Depends

# --- Configuration & Model Paths ---
RISK_THRESHOLD = 60
RULE_ENGINE_MAX_SCORE = 50

# Configuration for Dynamic Weight Adjustment
DYNAMIC_WEIGHT_CONFIG = {
    "adjustment_interval": 10,
    "min_weight": 0.30,
    "max_weight": 0.60,        
    "step_size": 0.02,         
    "TOTAL_ML_WEIGHT": 0.90    
}

# --- Risk Weights (Default values if no persistence file is found) ---
DEFAULT_RISK_WEIGHTS = { 
    "supervised_weight": 0.45,  
    "unsupervised_weight": 0.45,
    "behavior_weight": 0.1,    
}

# --- FIXED PATHS (Using '../' to reference parallel 'ml-development' folder) ---
SUPERVISED_MODEL_PATH = joblib.load("models/random_forest_classifier.joblib")
UNSUPERVISED_MODEL_PATH = joblib.load("models/isolation_forest_anomaly_detector.joblib")
SCALER_SUPERVISED_PATH = joblib.load("models/scaler_supervised.joblib")
SCALER_UNSUPERVISED_PATH = joblib.load("models/scaler_unsupervised.joblib") 

# Weights persistence file (local to the api/ directory)
WEIGHTS_PERSISTENCE_PATH = 'ml_weights_state.json' 

HEURISTICS = {
    "high_value": {
        "desc": "Rule: Transaction amount exceeds High-Value Threshold.",
        "threshold": 4500,
        "points": 20
    },
    "velocity_check": {
        "desc": "Rule: Real-time velocity spike detected.",
        "points": 15
    },
    "geo_mismatch": {
        "desc": "Rule: Current Geo does not match typical user location.",
        "points": 15
    },
}
# --- 2. DATA MODELS (Input Schema for API) ---

class TransactionInput(BaseModel):
    """Schema for the incoming transaction request."""
    transaction_id: str = Field(..., description="Unique ID for the transaction.")
    user_id: str = Field(..., description="The ID of the user performing the transaction.")
    amount: float = Field(..., description="Transaction monetary value.")
    
    # Pre-calculated Real-Time Features (from Feature Store)
    velocity_spike: bool = Field(False, description="True if transaction velocity exceeds norm.")
    
    # Transaction Context Features (for Rules & Behavior Profiling)
    location: str = Field(..., description="Transaction city/location.")
    device_id: str = Field(..., description="Unique device fingerprint.")
    time_hour: int = Field(..., description="Hour of the day (0-23).")
    merchant_type: str = Field(..., description="Category of the merchant.")


# --- 3. SIMULATED USER DATA STORE (Digital Persona) ---
USER_PROFILES = {
    "U123": {
        "typical_time": (9, 17), "typical_merchant": "groceries, utilities", "typical_geo": "New York", "typical_device": "Mobile-A"
    },
    "U456": {
        "typical_time": (20, 23), "typical_merchant": "e-commerce, entertainment", "typical_geo": "London", "typical_device": "Desktop-B"
    },
    "U789": {
        "typical_time": (12, 16), "typical_merchant": "utilities, bills", "typical_geo": "Tokyo", "typical_device": "Tablet-C"
    },
    "U999": {
        "typical_time": (18, 22), "typical_merchant": "restaurants, travel", "typical_geo": "Paris", "typical_device": "Mobile-D"
    },
    "DEFAULT": { 
        "typical_time": (0, 23), "typical_merchant": "any", "typical_geo": "unknown", "typical_device": "unknown"
    }
}

# --- 4. ENGINE CLASSES (Logic) ---

class RuleEngine:
    """Implements fast, transparent, and auditable rule-based heuristics."""
    def __init__(self, heuristics):
        self.rules = heuristics
        self.max_rule_score = sum(r["points"] for r in self.rules.values())

    def evaluate(self, txn: TransactionInput) -> Dict[str, Any]:
        rule_score = 0
        triggered_rules = []

        if txn.amount > self.rules["high_value"]["threshold"]:
            rule_score += self.rules["high_value"]["points"]
            triggered_rules.append(self.rules["high_value"]["desc"])

        if txn.velocity_spike:
            rule_score += self.rules["velocity_check"]["points"]
            triggered_rules.append(self.rules["velocity_check"]["desc"])

        profile = USER_PROFILES.get(txn.user_id, USER_PROFILES["DEFAULT"])
        geo_mismatch_flag = (txn.location != profile["typical_geo"])
        if geo_mismatch_flag and profile["typical_geo"] != "unknown":
            rule_score += self.rules["geo_mismatch"]["points"]
            triggered_rules.append(self.rules["geo_mismatch"]["desc"])

        return {
            "rule_score": rule_score,
            "triggered_rules": triggered_rules
        }

class UserProfiler:
    """Calculates the deviation of a transaction from a user's digital persona."""
    def __init__(self, profiles):
        self.profiles = profiles

    def calculate_deviation_score(self, txn: TransactionInput) -> Dict[str, Any]:
        user_id = txn.user_id
        profile = self.profiles.get(user_id, self.profiles["DEFAULT"])
        
        if profile is self.profiles["DEFAULT"]:
             return {"deviation_score": 0.5, "deviations": ["New user profile: Default score applied."]}

        deviation_points = 0
        max_points = 4
        deviations = []

        start_hour, end_hour = profile["typical_time"]
        current_hour = txn.time_hour
        is_outside_time = False
        if start_hour <= end_hour:
             if not (start_hour <= current_hour <= end_hour): is_outside_time = True
        else:
             if not (current_hour >= start_hour or current_hour <= end_hour): is_outside_time = True
                 
        if is_outside_time:
            deviation_points += 1
            deviations.append(f"Time deviation: {current_hour}:00 is outside usual window.")

        if txn.merchant_type not in profile["typical_merchant"]:
            deviation_points += 1
            deviations.append(f"Merchant deviation: '{txn.merchant_type}' is atypical.")

        if txn.location != profile["typical_geo"]:
            deviation_points += 1
            deviations.append(f"Geo deviation: Location '{txn.location}' is atypical.")
            
        if txn.device_id != profile["typical_device"]:
             deviation_points += 1
             deviations.append(f"Device deviation: New device ID '{txn.device_id}' detected.")

        deviation_score = deviation_points / max_points
        return {"deviation_score": deviation_score, "deviations": deviations}

class ModelInferenceEngine:
    """
    Loads trained models, performs real-time scoring, and tracks performance for dynamic weights.
    """
    def __init__(self, dynamic_config, default_weights):
        # 0. Load Local State (Weights and Counters)
        self.risk_weights, self.transaction_count, self.supervised_fraud_hits, self.anomaly_fraud_hits = \
            self._load_local_state(default_weights)
        
        # 1. Load ML Models
        self.supervised_model = self._safe_load(SUPERVISED_MODEL_PATH)
        self.unsupervised_model = self._safe_load(UNSUPERVISED_MODEL_PATH)
        self.scaler_supervised = self._safe_load(SCALER_SUPERVISED_PATH)
        self.scaler_unsupervised = self._safe_load(SCALER_UNSUPERVISED_PATH)
        
        self.dynamic_config = dynamic_config
        
        if not all([self.supervised_model, self.unsupervised_model, self.scaler_supervised]):
             # If models fail to load, switch to simulation mode
             print("WARNING: One or more models/scalers failed to load. Falling back to simulation.")
             self.is_ready = False
        else:
             print("SUCCESS: All Kaggle-trained models and scalers loaded successfully.")
             self.is_ready = True

    def _load_local_state(self, default_weights):
        """Loads weights and state from JSON file or uses defaults."""
        try:
            with open(WEIGHTS_PERSISTENCE_PATH, 'r') as f:
                state = json.load(f)
                print(f"INFO: Loaded persistent weights state from {WEIGHTS_PERSISTENCE_PATH}")
                # Ensure structure is valid before returning
                return state['risk_weights'], state['transaction_count'], state['supervised_fraud_hits'], state['anomaly_fraud_hits']
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            print(f"INFO: Weights persistence file not found or invalid. Using default weights.")
            return default_weights, 0, 0, 0

    def _save_local_state(self):
        """Saves current weights and state to JSON file."""
        state = {
            "risk_weights": self.risk_weights,
            "transaction_count": self.transaction_count,
            "supervised_fraud_hits": self.supervised_fraud_hits,
            "anomaly_fraud_hits": self.anomaly_fraud_hits
        }
        try:
            with open(WEIGHTS_PERSISTENCE_PATH, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print(f"ERROR: Failed to save weights state: {e}")

    def _safe_load(self, path):
        try:
            return joblib.load(path)
        except Exception as e:
            # We explicitly print the error here to help debugging, but return None to continue
            print(f"Error loading model from {path}: {e}")
            return None

    def _generate_mock_kaggle_vector(self, txn: TransactionInput, profile: Dict[str, Any]) -> pd.DataFrame:
        """
        Creates a mock 30-feature vector (Time, V1-V28, Amount) for model inference.
        """
        if self.scaler_supervised is None:
            # Cannot proceed without scaler; return empty frame
            return pd.DataFrame() 

        is_high_risk = txn.velocity_spike or (txn.amount > HEURISTICS["high_value"]["threshold"]) or (txn.location != profile.get("typical_geo"))
        
        features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        data = {f: [random.uniform(-1, 1)] for f in features} 
        
        data['Time'] = [txn.time_hour] 
        data['Amount'] = [txn.amount]
        
        if is_high_risk:
            for f in features[1:29]:
                 data[f] = [random.uniform(-5, 5)]
        else:
            for f in features[1:29]:
                 data[f] = [random.uniform(-0.1, 0.1)]

        df = pd.DataFrame(data)
        # Apply the scaler transformation to the two features used in the Kaggle dataset
        df[['Time', 'Amount']] = self.scaler_supervised.transform(df[['Time', 'Amount']])
        
        return df
    
    def adjust_weights(self):
        """Dynamic Weight Adjustment: Modifies ML weights based on recent performance."""
        
        if self.transaction_count < self.dynamic_config["adjustment_interval"]:
            return

        print(f"\n[AI Weight Adjustment Running (After {self.transaction_count} txns)...]")
        
        current_sup_w = self.risk_weights["supervised_weight"]
        current_un_w = self.risk_weights["unsupervised_weight"]
        step = self.dynamic_config["step_size"]
        min_w = self.dynamic_config["min_weight"]
        max_w = self.dynamic_config["max_weight"]
        
        adjustment_made = False

        if self.supervised_fraud_hits > self.anomaly_fraud_hits:
            new_sup_w = min(current_sup_w + step, max_w)
            new_un_w = self.dynamic_config["TOTAL_ML_WEIGHT"] - new_sup_w
            new_un_w = max(new_un_w, min_w)
            
            self.risk_weights["supervised_weight"] = new_sup_w
            self.risk_weights["unsupervised_weight"] = new_un_w
            print(f"  > Supervised leads ({self.supervised_fraud_hits} hits). New Weights: Sup={new_sup_w:.2f}, Unsup={new_un_w:.2f}")
            adjustment_made = True

        elif self.anomaly_fraud_hits > self.supervised_fraud_hits:
            new_un_w = min(current_un_w + step, max_w)
            new_sup_w = self.dynamic_config["TOTAL_ML_WEIGHT"] - new_un_w
            new_sup_w = max(new_sup_w, min_w)
            
            self.risk_weights["unsupervised_weight"] = new_un_w
            self.risk_weights["supervised_weight"] = new_sup_w
            print(f"  > Anomaly leads ({self.anomaly_fraud_hits} hits). New Weights: Sup={new_sup_w:.2f}, Unsup={new_un_w:.2f}")
            adjustment_made = True
            
        # Reset counters and persist state
        if adjustment_made:
            self.supervised_fraud_hits = 0
            self.anomaly_fraud_hits = 0
            self.transaction_count = 0
            self._save_local_state()


    def get_ml_scores(self, txn: TransactionInput) -> Dict[str, float]:
        """Performs real inference or falls back to simulation if models failed to load."""
        
        self.transaction_count += 1
        self.adjust_weights()

        if not self.is_ready:
             return self.get_ml_scores_simulation(txn)

        # --- Real Model Inference ---
        profile = USER_PROFILES.get(txn.user_id, USER_PROFILES["DEFAULT"])
        feature_vector = self._generate_mock_kaggle_vector(txn, profile)
        
        if feature_vector.empty:
             return self.get_ml_scores_simulation(txn) # Fallback if feature vector generation failed
        
        # 1. Supervised Model Score (Probability of Fraud)
        supervised_score = self.supervised_model.predict_proba(feature_vector)[:, 1][0]
        
        # 2. Unsupervised Model Score (Anomaly Deviation)
        raw_anomaly_score = self.unsupervised_model.decision_function(feature_vector)[0]
        normalized_anomaly_score = max(0, min(1, 1 - (raw_anomaly_score / 0.3))) 
        
        # Track hits for Dynamic Weight Adjustment
        if supervised_score > 0.8:
            self.supervised_fraud_hits += 1
        if normalized_anomaly_score > 0.8:
            self.anomaly_fraud_hits += 1

        return {
            "ml_supervised_score": supervised_score,
            "ml_unsupervised_score": normalized_anomaly_score,
        }
        
    def get_ml_scores_simulation(self, txn: TransactionInput) -> Dict[str, float]:
        """Fallback simulation logic."""
        is_high_risk = txn.velocity_spike or (txn.amount > 3000) or (txn.location != USER_PROFILES.get(txn.user_id, {}).get("typical_geo"))
        sup_score = random.uniform(0.75, 0.95) if is_high_risk else random.uniform(0.01, 0.15)
        un_score = random.uniform(0.85, 0.99) if is_high_risk else random.uniform(0.01, 0.1)
        return {"ml_supervised_score": sup_score, "ml_unsupervised_score": un_score}


# --- 5. FASTAPI APPLICATION SETUP ---
app = FastAPI(
    title="Hybrid AI Fraud Detection API",
    description="Real-Time Scoring service combining Rules, Behavior, and Hybrid ML.",
    version="1.0.0"
)

# NEW: CORS Middleware for local development access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows connection from any source (including local HTML file or Canvas)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines once on startup
rule_engine = RuleEngine(HEURISTICS)
profiler = UserProfiler(USER_PROFILES)
model_inference_engine = ModelInferenceEngine(DYNAMIC_WEIGHT_CONFIG, DEFAULT_RISK_WEIGHTS) 


# --- 6. SECURITY LAYER & 7. MAIN SCORING ENDPOINT ---
def authenticate_jwt(authorization: str = Header(..., description="Bearer token for service-to-service authentication.")):
    """Simulates JWT token validation."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    
    token = authorization.split(" ")[1]
    if token != "SECURE_API_KEY_12345": 
        raise HTTPException(status_code=403, detail="Invalid JWT token provided.")
    
    return {"service_name": "PaymentProcessor_v1"} 




@app.post("/score_transaction")
async def score_transaction(
    txn: TransactionInput,
    auth_info: Dict[str, Any] = Depends(authenticate_jwt)
):

    start_time = time.perf_counter()

    # 1. Rule-Based Heuristics
    rule_results = rule_engine.evaluate(txn)
    rule_score = rule_results["rule_score"]

    # 2. Behavior-Based Profiling
    behavior_results = profiler.calculate_deviation_score(txn)
    deviation_score = behavior_results["deviation_score"]

    # 3. Hybrid Machine Learning Models (Real Inference and Weight Adjustment)
    ml_scores = model_inference_engine.get_ml_scores(txn)
    
    # 4. Cloud-Based Risk Score Calculation (uses the CURRENT persistent weights)
    sup_w = model_inference_engine.risk_weights["supervised_weight"]
    un_w = model_inference_engine.risk_weights["unsupervised_weight"]
    beh_w = model_inference_engine.risk_weights["behavior_weight"]
    
    remaining_points = 100 - RULE_ENGINE_MAX_SCORE
    
    # Normalize combined ML score to 0-1 range
    combined_ml_score = (ml_scores["ml_supervised_score"] * sup_w + ml_scores["ml_unsupervised_score"] * un_w) / (sup_w + un_w)
    
    # Contribution from ML models
    ml_risk_contribution = combined_ml_score * remaining_points * (sup_w + un_w)
    
    # Contribution from Behavior Profiling
    behavior_risk_contribution = deviation_score * remaining_points * beh_w
    
    final_risk_score = min(
        rule_score + ml_risk_contribution + behavior_risk_contribution,
        100.0
    )

    # 5. Decision Making
    action = "APPROVED"
    if final_risk_score >= RISK_THRESHOLD:
        action = "BLOCKED"
    elif final_risk_score >= RISK_THRESHOLD * 0.7:
        action = "FLAGGED_FOR_REVIEW"
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    # 6. Construct Audit Log and Response
    return {
        "transaction_id": txn.transaction_id,
        "user_id": txn.user_id,
        "status": "Success",
        "decision": action,
        "final_risk_score": round(final_risk_score, 2),
        "processing_latency_ms": round(latency_ms, 3),
        "audit_breakdown": {
            "rule_score": rule_results["rule_score"],
            "triggered_rules": rule_results["triggered_rules"],
            "behavior_risk": round(behavior_risk_contribution, 2),
            "behavior_deviations": behavior_results["deviations"],
            "ml_risk": round(ml_risk_contribution, 2),
            "ml_supervised_score": round(ml_scores["ml_supervised_score"], 4),
            "ml_unsupervised_score": round(ml_scores["ml_unsupervised_score"], 4),
            "current_sup_weight": round(sup_w, 2),
            "current_un_weight": round(un_w, 2),
        }
    }