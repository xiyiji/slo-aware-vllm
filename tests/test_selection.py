import json
from slo_bench.finalize import choose


def test_selection_prefers_small_error_free_config_within_noise_band(tmp_path):
    paths = []
    for seq, goodput, errors in [(8,1.95,0), (16,2,0), (32,3,1)]:
        root = tmp_path / str(seq)
        run = root / "seed-17"
        run.mkdir(parents=True)
        (root / "environment.json").write_text(json.dumps({"command":["--max-num-seqs",str(seq),"--max-num-batched-tokens","4096"]}))
        path = run / "summary.json"
        path.write_text(json.dumps({"goodput_rps":goodput,"errors":errors}))
        paths.append(path)
    assert choose(paths)["seqs"] == 8
