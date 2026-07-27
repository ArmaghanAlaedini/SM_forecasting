# Strictly sequential Slurm submission update

Replace these two files in your clean local `sbatch/` folder:

```text
submit_modeling_pipeline.sh
submit_full_pipeline.sh
```

The revised dependency chain is linear. For the modeling workflow:

```text
07 -> 09 -> 10a -> 10b -> 10c -> 10d -> 10e -> 10h -> 11a -> 11b -> 11c
```

No ML/GI pair runs concurrently.

All jobs are still submitted to Slurm at once, but only the first is eligible to
run. Each later job remains in `PENDING (Dependency)` until the immediately
preceding job completes successfully.

After replacement:

```bash
chmod +x sbatch/submit_modeling_pipeline.sh
chmod +x sbatch/submit_full_pipeline.sh
```

Then transfer the clean folder:

```bash
bash sbatch/sync_to_nova.sh
bash sbatch/sync_to_nova.sh --apply
```

On Nova, submit the modeling-only workflow:

```bash
bash sbatch/submit_modeling_pipeline.sh
```
