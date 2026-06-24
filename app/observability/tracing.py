import logging

logger = logging.getLogger(__name__)


def setup_tracing() -> None:
    from app.utils.config import get_settings
    cfg = get_settings().observability.mlflow

    if not cfg.enabled:
        return

    try:
        import mlflow

        mlflow.set_tracking_uri(cfg.tracking_uri)
        mlflow.set_experiment(cfg.experiment)

        if cfg.autolog.langchain:
            mlflow.langchain.autolog()
        if cfg.autolog.openai:
            mlflow.openai.autolog()

        logger.info("MLflow tracing enabled at %s (experiment: %s)", cfg.tracking_uri, cfg.experiment)
    except ImportError:
        logger.warning("mlflow not installed — tracing disabled")
    except Exception as exc:
        logger.warning("MLflow setup failed: %s", exc)
