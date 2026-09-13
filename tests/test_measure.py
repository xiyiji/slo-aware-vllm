import json
import pytest
from slo_bench.measure import StreamMeasurement, summarize


def test_chunk_is_not_token():
    m = StreamMeasurement()
    m.consume(json.dumps({"choices": [{"text": "many tokens here"}]}), .1)
    m.consume(json.dumps({"choices": [{"text": " end", "finish_reason": "length"}]}), .4)
    m.consume(json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 8}}), .5)
    m.consume("[DONE]", .5)
    assert m.result(8)["completion_tokens"] == 8
    assert m.result(8)["tpot_s"] == pytest.approx(.3/7)


def test_role_event_does_not_start_ttft():
    m = StreamMeasurement()
    m.consume('{"choices":[{"delta":{"role":"assistant"}}]}', .01)
    assert m.first is None


@pytest.mark.parametrize("payload", ['[DONE]', '{"choices":[]}'])
def test_missing_usage_fails(payload):
    m = StreamMeasurement()
    m.consume(payload, 1)
    with pytest.raises(ValueError):
        m.result(10)


def test_goodput_counts_failures_and_both_slos():
    rows = [{"ok": True, "ttft_s": .2, "e2e_s": 2, "completion_tokens": 10},
            {"ok": True, "ttft_s": 2, "e2e_s": 3, "completion_tokens": 10},
            {"ok": False}]
    s = summarize(rows, 10, 1, 10)
    assert s["goodput_rps"] == .1
    assert s["output_tokens_s"] == 2
    assert s["errors"] == 1


def test_wrong_token_count_is_not_a_success():
    m = StreamMeasurement()
    m.consume('{"choices":[{"text":"hi","finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1}}', .1)
    m.consume('[DONE]', .2)
    with pytest.raises(ValueError, match='Expected'):
        m.result(128)
    assert m.result(1)['tpot_s'] is None


def test_server_error_is_not_content():
    m = StreamMeasurement()
    with pytest.raises(ValueError):
        m.consume('{"error":{"message":"out of memory"}}', 1)
