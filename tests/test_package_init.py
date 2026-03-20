"""Tests for package import bootstrap behavior."""

import importlib

import lora_finetune
import lora_finetune._optional_unsloth as optional_unsloth


class TestPackageBootstrap:
    def test_package_import_does_not_bootstrap_unsloth(self, monkeypatch):
        events = []

        monkeypatch.setattr(
            optional_unsloth,
            "ensure_unsloth_imported",
            lambda: events.append("bootstrap"),
        )

        importlib.reload(lora_finetune)

        assert events == []
