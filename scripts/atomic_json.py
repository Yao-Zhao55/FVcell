"""Atomic state writes with bounded retries for Windows sharing violations."""
import json,time
def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.new')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(50):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==49:raise
            time.sleep(min(0.1*(attempt+1),0.5))
