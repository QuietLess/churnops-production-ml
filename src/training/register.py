"""Promote a registered model version to the `champion` alias and export it.

Usage:
  python -m src.training.register                 # promote latest trained version if it is not worse
  python -m src.training.register --version 3     # promote a specific version
  python -m src.training.register --force         # skip the challenger check
  python -m src.training.register --export-only   # just export the current champion

Light champion/challenger rule: the candidate is promoted only if its validation PR-AUC
is not lower than the current champion's. The promoted model is exported to
artifacts/champion_model/ so it can be baked into a deployment image when a remote
registry is not available (documented compromise; lineage stays in MLflow).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config import (
    ARTIFACTS_DIR,
    CHAMPION_EXPORT_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
)
from src.training.serialization import trusted_types_for

logger = logging.getLogger(__name__)


def current_champion(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    except MlflowException:
        return None


def latest_version(client: MlflowClient) -> str:
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise SystemExit(f"No versions registered for {REGISTERED_MODEL_NAME}. Run `make train`.")
    return str(max(int(v.version) for v in versions))


def export_champion(version: str, export_dir: Path = CHAMPION_EXPORT_DIR) -> None:
    """Materialise models:/<name>@champion as a self-contained MLflow model directory."""
    uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(uri)
    metadata = dict(mlflow.models.get_model_info(uri).metadata or {})
    metadata.update(
        {"model_name": REGISTERED_MODEL_NAME, "model_alias": MODEL_ALIAS, "model_version": version}
    )
    if export_dir.exists():
        shutil.rmtree(export_dir)
    mlflow.sklearn.save_model(
        model,
        str(export_dir),
        metadata=metadata,
        skops_trusted_types=trusted_types_for(model),
    )
    logger.info("Exported champion v%s to %s", version, export_dir)


def promote(
    client: MlflowClient,
    version: str | None = None,
    force: bool = False,
    export_dir: Path | None = CHAMPION_EXPORT_DIR,
) -> tuple[str, bool]:
    """Point the champion alias at `version` unless it is worse than the current champion.

    Returns (serving_version, promoted).
    """
    champion = current_champion(client)
    version = version or latest_version(client)
    candidate = client.get_model_version(REGISTERED_MODEL_NAME, version)
    cand_score = float(candidate.tags.get("valid_pr_auc", "nan"))

    if champion is not None and str(champion.version) != version and not force:
        champ_score = float(champion.tags.get("valid_pr_auc", "nan"))
        if cand_score < champ_score:
            logger.warning(
                "Not promoting v%s (valid PR-AUC %.4f) over champion v%s (%.4f). Use --force.",
                version,
                cand_score,
                champion.version,
                champ_score,
            )
            if export_dir is not None:
                export_champion(str(champion.version), export_dir)
            return str(champion.version), False

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS, version)
    logger.info("Alias %s -> %s v%s", MODEL_ALIAS, REGISTERED_MODEL_NAME, version)
    if export_dir is not None:
        export_champion(version, export_dir)
    return version, True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", help="model version to promote (default: latest)")
    parser.add_argument("--force", action="store_true", help="promote even if worse than champion")
    parser.add_argument("--export-only", action="store_true", help="only export current champion")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    if args.export_only:
        champion = current_champion(client)
        if champion is None:
            raise SystemExit("No champion alias set yet.")
        export_champion(str(champion.version))
        return

    version, promoted = promote(client, args.version, args.force)
    score = client.get_model_version(REGISTERED_MODEL_NAME, version).tags.get("valid_pr_auc")
    (ARTIFACTS_DIR / "champion.json").write_text(
        json.dumps(
            {
                "model_name": REGISTERED_MODEL_NAME,
                "alias": MODEL_ALIAS,
                "version": version,
                "valid_pr_auc": float(score) if score else None,
                "promoted_this_run": promoted,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
