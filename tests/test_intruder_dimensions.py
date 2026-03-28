"""Tests for intruder dimension analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from lora_finetune.intruder_dimensions import (
    DEFAULT_LLAMA_MODULE_REGEXES,
    AnalysisConfig,
    analyze_model,
    build_tuned_tensor,
    compute_intruders,
    detect_tuned_type,
    filter_weight_names,
    load_lora_delta_tensor,
    resolve_checkpoint_dir,
)


def test_compute_intruders_no_intruder_case():
    base = torch.diag(torch.tensor([5.0, 3.0, 1.0]))
    tuned = torch.diag(torch.tensor([5.1, 2.9, 1.0]))

    result = compute_intruders(base, tuned, epsilon=0.9, k=2, weight_name="test.weight")

    assert result.intruder_count == 0
    assert result.examined_k == 2
    assert result.intruder_indices == []


def test_compute_intruders_detects_intruder():
    base = torch.diag(torch.tensor([5.0, 4.0, 1.0]))
    intruder_vector = torch.tensor([1.0, 1.0, 1.0]) / torch.sqrt(torch.tensor(3.0))
    basis = torch.eye(3)
    basis[:, 0] = intruder_vector
    q, _ = torch.linalg.qr(basis)
    tuned = q @ torch.diag(torch.tensor([6.0, 4.0, 1.0]))

    result = compute_intruders(base, tuned, epsilon=0.7, k=1, weight_name="test.weight")

    assert result.intruder_count == 1
    assert result.intruder_indices == [0]
    assert result.max_abs_cosines[0] < 0.7


def test_compute_intruders_is_sign_invariant():
    base = torch.diag(torch.tensor([5.0, 3.0, 1.0]))
    tuned = -base

    result = compute_intruders(base, tuned, epsilon=0.99, k=3, weight_name="test.weight")

    assert result.intruder_count == 0
    assert result.intruder_indices == []


def test_load_lora_delta_tensor_reconstructs_delta(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()

    config = {"r": 2, "lora_alpha": 4}
    (checkpoint_dir / "adapter_config.json").write_text(json.dumps(config))

    lora_a = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
    lora_b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": lora_a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": lora_b,
        },
        str(checkpoint_dir / "adapter_model.safetensors"),
    )

    delta = load_lora_delta_tensor(checkpoint_dir, "model.layers.0.self_attn.q_proj.weight")

    expected = (lora_b @ lora_a) * 2.0
    torch.testing.assert_close(delta, expected)


def test_build_tuned_tensor_for_lora(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}))
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.tensor([[2.0, 0.0]]),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.tensor([[1.0], [3.0]]),
        },
        str(checkpoint_dir / "adapter_model.safetensors"),
    )

    base = torch.zeros(2, 2)
    tuned = build_tuned_tensor(
        base,
        checkpoint_dir,
        "lora",
        "model.layers.0.self_attn.q_proj.weight",
    )

    expected = torch.tensor([[2.0, 0.0], [6.0, 0.0]])
    torch.testing.assert_close(tuned, expected)


def test_detect_tuned_type(tmp_path: Path):
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    save_file({"weight": torch.ones(1)}, str(full_dir / "model.safetensors"))

    lora_dir = tmp_path / "lora"
    lora_dir.mkdir()
    save_file({"weight": torch.ones(1)}, str(lora_dir / "adapter_model.safetensors"))
    (lora_dir / "adapter_config.json").write_text("{}")

    assert detect_tuned_type(full_dir) == "full"
    assert detect_tuned_type(lora_dir) == "lora"


def test_resolve_checkpoint_dir_uses_latest_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run-abc"
    run_dir.mkdir()
    checkpoint_100 = run_dir / "checkpoint-100"
    checkpoint_100.mkdir()
    save_file({"weight": torch.ones(1)}, str(checkpoint_100 / "model.safetensors"))
    checkpoint_500 = run_dir / "checkpoint-500"
    checkpoint_500.mkdir()
    save_file({"weight": torch.ones(1)}, str(checkpoint_500 / "model.safetensors"))

    assert resolve_checkpoint_dir(run_dir) == checkpoint_500.resolve()


def test_filter_weight_names_keeps_llama_projection_weights():
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "lm_head.weight",
    ]

    filtered = filter_weight_names(names, DEFAULT_LLAMA_MODULE_REGEXES)

    assert filtered == [
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]


def test_analyze_model_on_tiny_full_checkpoint(tmp_path: Path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    weight_name = "model.layers.0.self_attn.q_proj.weight"
    base_tensor = torch.diag(torch.tensor([5.0, 3.0]))
    save_file({weight_name: base_tensor}, str(base_dir / "model.safetensors"))

    checkpoint_dir = tmp_path / "run-1" / "checkpoint-10"
    checkpoint_dir.mkdir(parents=True)
    tuned_tensor = torch.diag(torch.tensor([5.1, 2.9]))
    save_file({weight_name: tuned_tensor}, str(checkpoint_dir / "model.safetensors"))

    report = analyze_model(
        AnalysisConfig(
            base_model_path=base_dir,
            tuned_path=checkpoint_dir.parent,
            epsilon=0.9,
            k=2,
            device="cpu",
        )
    )

    assert report.tuned_type == "full"
    assert report.total_intruders == 0
    assert report.num_matrices == 1
    assert report.results[0].weight_name == weight_name


def test_analyze_model_emits_progress_events(tmp_path: Path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    weight_name = "model.layers.0.self_attn.q_proj.weight"
    save_file({weight_name: torch.diag(torch.tensor([2.0, 1.0]))}, str(base_dir / "model.safetensors"))

    checkpoint_dir = tmp_path / "run-1" / "checkpoint-10"
    checkpoint_dir.mkdir(parents=True)
    save_file(
        {weight_name: torch.diag(torch.tensor([2.1, 0.9]))},
        str(checkpoint_dir / "model.safetensors"),
    )

    phases = []

    def on_progress(event: dict) -> None:
        phases.append(event["phase"])

    analyze_model(
        AnalysisConfig(
            base_model_path=base_dir,
            tuned_path=checkpoint_dir.parent,
            epsilon=0.9,
            k=2,
            device="cpu",
        ),
        progress_callback=on_progress,
    )

    assert phases == [
        "start",
        "load_base",
        "build_tuned",
        "svd_base",
        "svd_tuned",
        "compare",
        "matrix_complete",
        "complete",
    ]
