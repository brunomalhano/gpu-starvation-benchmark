import ast

with open("Experimento_artigo_jitter.py") as f:
    source = f.read()
    ast.parse(source)

checks = [
    ("NUM_RODADAS = 10", "NUM_RODADAS = 10" in source),
    ("NUM_EPOCHS = 3", "NUM_EPOCHS = 3" in source),
    ("NUM_WORKERS = 0", "NUM_WORKERS = 0" in source),
    ("BATCH_SIZE = 32", "BATCH_SIZE = 32" in source),
    ("auto-detect device", 'torch.cuda.is_available()' in source),
    ("torch.cuda.synchronize", "torch.cuda.synchronize()" in source),
    ("socket.gethostname", "socket.gethostname()" in source),
    ("jitter_espera", "jitter_espera" in source),
    ("cv_espera", "cv_espera" in source),
    ("p50/p95/p99", "p50_espera" in source and "p95_espera" in source and "p99_espera" in source),
    ("latencias_espera_batch", "latencias_espera_batch" in source),
    ("latencias_compute_batch", "latencias_compute_batch" in source),
    ("batch_latencies CSV", "batch_latencies_" in source),
    ("jitter_results CSV", "jitter_results_" in source),
    ("no scipy", "scipy" not in source),
    ("utf-8 encoding", source.count("utf-8") >= 3),
    ("KeyboardInterrupt handler", "_silenciar_keyboard_interrupt" in source),
    ("DatasetImagensSoltas", "class DatasetImagensSoltas" in source),
    ("Sscenario A", "Sscenario A" in source),
    ("Sscenario B", "Sscenario B" in source),
    ("Sscenario C", "Sscenario C" in source),
    ("re-instantiate model", source.count("models.resnet18()") >= 3),
    ("host column", '"host"' in source),
    ("16 epoch CSV columns", source.count("num_batches") >= 2),
]

all_pass = True
for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  [{status}] {name}")

print(f"\nAll checks passed: {all_pass}")
