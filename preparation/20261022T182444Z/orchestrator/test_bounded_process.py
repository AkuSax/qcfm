"""Test a bounded process without model calls."""
import json
from pathlib import Path
from run_bounded import run

root = Path(__file__).resolve().parent
completed = run(root/'synthetic_process.py',['0'],root/'process_test_completed',10)
timed_out = run(root/'synthetic_process.py',['5'],root/'process_test_timeout',1)
assert completed == 0 and timed_out == 124
result = {'status':'PASS','completed_process_exit':completed,'timed_out_process_exit':timed_out,
          'pretrained_model_calls':0,'fits':0}
(root/'bounded_process_synthetic_test.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
