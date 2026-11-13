#!/bin/bash
# Run each script on a few images to check that it executes.
# Usage, from the repository root: tests/smoke_test.sh data/imagenet/val
set -u
VAL=${1:-data/imagenet/val}
PY=${PYTHON:-python}
O=$(mktemp -d)
R=results
E=experiments
M=vit_base_patch16_224.augreg2_in21k_ft_in1k
TAX=$R/taxonomy_vitb_in21k.json
DIAG=$R/fixed_target_vitb.json
FLAT=$(find "$VAL" -mindepth 1 -maxdepth 1 -type d | sort | head -1)
fail=0

run() {
    name=$1
    shift
    if "$@" > "$O/$name.log" 2>&1; then
        echo "ok    $name"
    else
        echo "FAIL  $name"
        tail -4 "$O/$name.log" | cut -c1-160
        fail=$((fail + 1))
    fi
}

run unit_kernel $PY tests/test_kernel_substitution.py
run unit_taxonomy $PY tests/test_operator_taxonomy.py
run threshold $PY $E/threshold_ablation.py
run reproduce_stored $PY reproduce.py --val-dir "$VAL" --n-eval 64 --batch-size 32
run reproduce_fresh $PY reproduce.py --val-dir "$VAL" --fresh --n-cal 16 --n-eval 64 --batch-size 32
run reproduce_exact $PY reproduce.py --val-dir "$VAL" --exact-svd --n-eval 32 --batch-size 32 --eval-order random
run adaptive $PY $E/adaptive_rank_eval.py --model $M --taxonomy $TAX --svd-spectrum $R/svd_spectrum_vitb.json \
    --n-eval 64 --cls-cos-n 64 --batch-size 32 --imagenet-val-dir "$VAL" --out $O/adaptive.json
run broadcast $PY $E/broadcast_controls.py --model $M --taxonomy $TAX --diag $DIAG --n-eval-batches 1 \
    --eval-batch-size 32 --imagenet-val-dir "$VAL" --out $O/broadcast.json
run rank_k $PY $E/rank_k_controls.py --model $M --taxonomy $TAX --diag $DIAG --k-rank 10 --n-eval-batches 1 \
    --eval-batch-size 32 --imagenet-val-dir "$VAL" --out $O/rank.json
run kernel $PY $E/kernel_substitution_controls.py --model $M --taxonomy $TAX --categories delta_shift \
    --source imagenet-val --batches 1 --batch-size 32 --imagenet-val-dir "$VAL" --out $O/kernel.json
run image_mean $PY $E/attn_matrix_controls.py --model $M --taxonomy $TAX --diag $DIAG --n-eval-batches 1 \
    --eval-batch-size 32 --calib-batches 1 --imagenet-val-dir "$VAL" --out $O/matrix.json
run svd_spectrum $PY $E/svd_spectrum_diagnostic.py --model $M --taxonomy $TAX --batches 1 --batch-size 8 \
    --imagenet-val-dir "$VAL" --out $O/svd.json
run faithfulness $PY $E/faithfulness_probe.py --model $M --taxonomy $TAX --diag $DIAG --batch-size 16 --n-batches 1 \
    --imagenet-val-dir "$VAL" --out $O/faith.json
run robustness $PY $E/substitution_robustness.py --model $M --taxonomy $TAX --diag $DIAG --batch-size 16 --max-batches 1 \
    --corruptions clean,gauss_blur_2 --imagenet-val-dir "$VAL" --out $O/robust.json
run circuit $PY $E/circuit_composition.py --model $M --taxonomy $TAX --batch-size 16 --n-batches 1 --max-senders 1 \
    --imagenet-val-dir "$VAL" --out $O/circuit.json
run circuit_full $PY $E/circuit_composition_full.py --model $M --taxonomy $TAX --diag $DIAG --operator broadcast \
    --batch-size 16 --n-batches 1 --max-senders 1 --imagenet-val-dir "$VAL" --out $O/circuit_full.json
run behavioral $PY $E/behavioral_mapping.py --model $M --taxonomy $TAX --diag $DIAG --n-eval-batches 1 --eval-batch-size 32 \
    --imagenet-val-dir "$VAL" --out $O/behav.json
run analyze_behavioral $PY $E/analyze_behavioral.py --input $O/behav.json
run rank_curve $PY $E/rank_curve_near_global.py --model $M --taxonomy $TAX --ks 1 10 --n-batches 1 --batch-size 32 \
    --imagenet-val-dir "$VAL" --out $O/curve.json
run other_sweep $PY $E/other_rank_sweep.py --imagenet-val-dir "$VAL" --n-eval-batches-top1 1 --n-eval-batches-cos 1 \
    --eval-batch-size 16 --ks 30 --only-models $M --out $O/other.json
run fill_cells $PY $E/fill_cells.py --model $M --op shift --metric top1 --imagenet-val-dir "$VAL" --n-cal 8 --n-eval 32 \
    --out $O/fill.json
run tau_sweep $PY $E/tau_sweep.py --mode dinov2 --model deit_tiny_patch16_224 --imagenet-val-dir "$VAL" --n-cal 8 --n-eval 32 \
    --taus 0.9 --out $O/tau.json
run coco_stability $PY $E/coco_taxonomy_stability.py --imagenet-val-dir "$VAL" --coco-dir "$FLAT" --n-cal 8 --n-repeats 1 \
    --out $O/stab.json
run cross_attention $PY $E/cross_attention_eval.py --coco-dir "$FLAT" --n-images 2 --out $O/cross.json
run siglip_zero_shot $PY $E/zeroshot_siglip_eval.py --imagenet-val-dir "$VAL" --n-eval 32 --out $O/zero.json
run siglip_zero_shot_random $PY $E/zeroshot_siglip_eval.py --imagenet-val-dir "$VAL" --n-eval 32 --order random \
    --out $O/zero_random.json
run siglip_taxonomy env SOLA_PREPROCESS=siglip $PY $E/operator_taxonomy.py --model vit_base_patch16_siglip_224.webli \
    --batches 1 --batch-size 8 --out $O/tax_siglip.json
run siglip_adaptive env SOLA_PREPROCESS=siglip $PY $E/adaptive_rank_eval.py --model vit_base_patch16_siglip_224.webli \
    --taxonomy $R/taxonomy_siglip.json --svd-spectrum $R/svd_spectrum_siglip.json --no-top1 --cls-cos-n 32 \
    --batch-size 16 --imagenet-val-dir "$VAL" --out $O/adaptive_siglip.json
run preprocess_timm_cfg env SOLA_PREPROCESS=$M $PY $E/adaptive_rank_eval.py --model $M --taxonomy $TAX \
    --svd-spectrum $R/svd_spectrum_vitb.json --n-eval 32 --cls-cos-n 32 --batch-size 32 --imagenet-val-dir "$VAL" \
    --out $O/adaptive_cfg.json
run triton_tests $PY $E/test_triton_shift.py
run wallclock $PY $E/fused_wallclock.py --model $M --taxonomy $TAX --broadcast-json $R/broadcast_vitb.json --batch-size 8 \
    --n-warmup 2 --n-runs 2 --source synthetic --n-acc-batches 0 --out $O/wall.json
run energy $PY $E/energy_benchmark.py --model $M --taxonomy $TAX --broadcast-json $R/broadcast_vitb.json --batch-size 8 \
    --idle-s 1 --warmup-s 1 --run-s 2 --gpu-index 0 --dtype fp32 --out $O/energy.json
run taxonomy $PY $E/operator_taxonomy.py --model deit_tiny_patch16_224 --batches 1 --batch-size 8 --out $O/tax.json
run fixed_target $PY $E/fixed_target_diagnostic.py --model deit_tiny_patch16_224 --taxonomy $R/taxonomy_deit.json --batches 1 \
    --batch-size 8 --out $O/ft.json
run per_image_shift $PY $E/per_image_shift_diagnostic.py --model deit_tiny_patch16_224 --taxonomy $R/taxonomy_deit.json \
    --batches 1 --batch-size 8 --out $O/pis.json
run escape_mass $PY $E/escape_mass_sweep.py --models deit_tiny_patch16_224 --batch-size 8 --batches 1 --out-dir $O
run shift_controls $PY $E/shift_substitution_controls.py --model deit_tiny_patch16_224 --shift-heads 1,2 2,0 --batches 1 \
    --batch-size 16 --n-extract-batches 1 --n-random-seeds 1 --out $O/shiftc.json
run visualize $PY $E/visualize_heads.py --model deit_tiny_patch16_224 --heads 1,2 --n-images 1 --n-fingerprint-images 4 \
    --n-queries 2 --out $O/heads.png
run fig_taxonomy $PY $E/make_taxonomy_figure.py --inputs $R/taxonomy_deit.json $R/taxonomy_vitb_in21k.json --out $O/tax.pdf
run fig_comparison $PY $E/make_comparison_figure.py --sup-dir $R/emergence/deit_tiny --mae-dir $R/emergence/mae_tiny --out $O/cmp.pdf
run fig_circuit $PY $E/make_circuit_composition_figure.py --vitb $R/circuit_composition_vitb.json \
    --deit $R/circuit_composition_deit.json --out $O/circ.pdf
run fig_escape $PY $E/make_escape_mass_figure.py --in $R/escape_comparison.json --out $O/esc.pdf
run fig_map $PY $E/make_operator_map_figure.py --taxonomy $TAX --diag $DIAG --out $O/map.pdf
for script in train_deit_tiny train_mae_tiny analyze_checkpoint mae_other_rank_test llava_substitution_eval coco_seg_probe; do
    run "help_$script" $PY $E/$script.py --help
done
echo "$fail failures, logs in $O"
exit $fail
