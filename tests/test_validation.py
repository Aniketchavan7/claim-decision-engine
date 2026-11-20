import json
import os
import pytest

def test_sample_claim_upload_structure():
    sample_file = os.path.join(os.path.dirname(__file__), "..", "sample_claim_upload.json")
    assert os.path.exists(sample_file), "sample_claim_upload.json must exist"
    
    with open(sample_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert isinstance(data, (dict, list)), "Claim payload must be a valid JSON object or list"
    if isinstance(data, dict):
        assert len(data.keys()) > 0, "Claim payload must not be empty"

def test_requirements_file_validity():
    req_file = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
    assert os.path.exists(req_file), "requirements.txt must exist"
    
    with open(req_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        
    assert len(lines) > 0, "requirements.txt should have dependencies"
