import os
import certifi
import requests
import json
import ssl
from requests.exceptions import RequestException

# --------- Config ----------
API_KEY = "AIzaSyAouDwhFUCk3WNY7xrLn-rUDoMTBef0UHY"  # replace with your key
MODEL = "models/text-bison-001"    # example GA model; change if you have a different model name
# REST endpoint (v1beta2 or v1 depending on availability). Use the correct version for your quota/region.
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta2/{MODEL}:generateText"

# Helpful headers
HEADERS = {
    "Content-Type": "application/json",
    # We use API key in query param below, not bearer header
}

# --------- Helper to call REST securely (certifi) then insecurely if needed ----------
def call_google_gen_rest(prompt_text, timeout=20, use_insecure_fallback=True):
    # request body per Generative Language REST API (simple prompt shape)
    body = {
        "prompt": {
            "text": prompt_text
        },
        # optional tuning params:
        "temperature": 0.2,
        "maxOutputTokens": 512
    }

    params = {"key": API_KEY}

    # Try secure call first using certifi CA bundle
    try:
        ca_bundle = certifi.where()
        resp = requests.post(BASE_URL, headers=HEADERS, params=params, json=body, timeout=timeout, verify=ca_bundle)
        resp.raise_for_status()
        return resp.json()
    except RequestException as e_secure:
        # If secure call fails and fallback allowed, try insecure (debug only)
        print("⚠️ Secure REST call failed:", e_secure)
        if not use_insecure_fallback:
            raise

        print("⚠️ Retrying with SSL verification disabled (INSECURE fallback).")
        try:
            resp2 = requests.post(BASE_URL, headers=HEADERS, params=params, json=body, timeout=timeout, verify=False)
            resp2.raise_for_status()
            return resp2.json()
        except RequestException as e_insecure:
            # both failed
            print("❌ Insecure fallback also failed:", e_insecure)
            # raise the original secure exception for debugging
            raise

# --------- Utility to extract text from response ----------
def extract_generated_text(resp_json):
    """
    For the v1beta2 / v1 REST API responses the shape may look like:
    {
      "candidates": [{"output": "generated text here"}],
      ...
    }
    or
    {
      "output": "..."
    }
    Different versions/quotas may vary — inspect `resp_json` if this doesn't match.
    """
    # Try a few common shapes:
    if not isinstance(resp_json, dict):
        return None
    # 1) v1beta2 might return 'candidates'
    if "candidates" in resp_json and isinstance(resp_json["candidates"], list) and len(resp_json["candidates"]) > 0:
        cand = resp_json["candidates"][0]
        # candidate might have 'output' or 'content'
        return cand.get("output") or cand.get("content") or json.dumps(cand)
    # 2) some responses put text in 'output'
    if "output" in resp_json and isinstance(resp_json["output"], str):
        return resp_json["output"]
    # 3) some variants use 'results' -> 'content' etc (less common)
    if "results" in resp_json and isinstance(resp_json["results"], list) and len(resp_json["results"]) > 0:
        first = resp_json["results"][0]
        if isinstance(first, dict) and "content" in first:
            return first["content"]
    # 4) fallback, return full json as string
    return json.dumps(resp_json, indent=2)

# --------- Demo run ----------
if __name__ == "__main__":
    test_prompt = "Hello! Please say 'API working' if you can respond."
    try:
        resp = call_google_gen_rest(test_prompt)
    except Exception as e:
        print("Final error calling Generative REST API:", e)
    else:
        print("Raw response JSON:\n", json.dumps(resp, indent=2)[:2000])
        text = extract_generated_text(resp)
        print("\n=== Generated text ===\n", text)
