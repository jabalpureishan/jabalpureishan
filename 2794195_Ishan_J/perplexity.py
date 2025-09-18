import requests
import warnings
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# ---- cheap insecure debug setup ----
warnings.simplefilter("ignore", InsecureRequestWarning)  # hide the warning
API_KEY = "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"  # replace with your key
URL = "https://api.perplexity.ai/chat/completions"

# ---- one-line-ish request, skipping verification (verify=False) ----
resp = requests.post(
    URL,
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "Hello Perplexity — is my key working?"}]
    },
    verify=False,   # <-- disables SSL cert verification (INSECURE)
    timeout=15
)

if resp.ok:
    try:
        print("Reply:", resp.json()["choices"][0]["message"]["content"])
    except Exception:
        print("Got response but could not parse JSON:", resp.text)
else:
    print("Request failed:", resp.status_code, resp.text)
