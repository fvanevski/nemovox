"""Worker runtime-identity checks, with no installed model or CUDA dependency."""
import hashlib
import io
import json
import sys
from types import SimpleNamespace

import pytest

from nvidia_transcribe import _whisper_challenge_worker as worker


@pytest.mark.parametrize("mode", ["cuda", "cpu", "unsupported", "wrong_compute"])
def test_worker_validates_actual_runtime_before_observations(tmp_path, monkeypatch, capsys, mode):
    import importlib.metadata
    for name in ("model.bin","config.json","tokenizer.json","preprocessor_config.json"):
        (tmp_path/name).write_bytes(b"cached fixture")
    wav=tmp_path/"crop.wav"
    wav.write_bytes(b"fixture audio")
    request={"checkpoint":str(tmp_path),"device":"cuda:0","compute_type":"int8_float16",
        "language":"fr","crops":[{"id":"crop","path":str(wav),"start":10.,"end":12.,
        "audio_sha256":hashlib.sha256(wav.read_bytes()).hexdigest()}]}
    monkeypatch.setattr(sys,"stdin",io.StringIO(json.dumps(request)))
    monkeypatch.setattr(importlib.metadata,"version",lambda name:"fixture")
    calls=[]
    monkeypatch.setitem(sys.modules,"ctranslate2",SimpleNamespace(
        get_cuda_device_count=lambda:1,
        get_supported_compute_types=lambda device,index: (
            {"float16"} if mode=="unsupported" else {"int8_float16"})))
    class Model:
        def __init__(self,path,**kwargs):
            assert kwargs=={"device":"cuda","device_index":0,
                            "compute_type":"int8_float16","local_files_only":True}
            calls.append("load")
            self.model=SimpleNamespace(device="cpu" if mode=="cpu" else "cuda",
                device_index=[0],compute_type="float16" if mode=="wrong_compute" else "int8_float16",
                model_is_loaded=True,is_multilingual=True)
        def transcribe(self,path,**kwargs):
            calls.append("transcribe")
            assert kwargs=={"language":"fr","beam_size":5,"word_timestamps":True,"vad_filter":False}
            return iter([SimpleNamespace(start=.2,end=1.8,text="Bonjour",words=[])]),SimpleNamespace(language="fr")
    monkeypatch.setitem(sys.modules,"faster_whisper",SimpleNamespace(WhisperModel=Model))
    if mode!="cuda":
        with pytest.raises(RuntimeError):
            worker.main()
        assert "transcribe" not in calls
        assert not capsys.readouterr().out
        return
    worker.main()
    result=json.loads(capsys.readouterr().out)
    assert calls==["load","transcribe"]
    assert result["device"]=="cuda" and result["device_index"]==[0]
    assert result["compute_type"]=="int8_float16" and result["model_is_loaded"]
    assert result["observations"][0]["rows"][0]["source_start"]==10.2
    assert result["observations"][0]["audio_sha256"]==request["crops"][0]["audio_sha256"]
