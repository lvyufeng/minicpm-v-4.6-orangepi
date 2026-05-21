# MiniCPM-V Engine (Phase 0)

Current scope:
- ACL context and stream wrapper
- Device tensor RAII
- safetensors header/index parser
- weight loading smoke test

Build:
```bash
bash /mnt/data/minicpm/engine/scripts/build.sh
```

Run smoke test:
```bash
source /mnt/data/minicpm/engine/scripts/set_env.sh
/mnt/data/minicpm/engine/build/test_weights_load
```
