from src.logger import logging

from src.pipeline.train_pipeline import TrainingPipeline


STAGE_NAME = "Model Trainer stage"
try:
    logging.info(f"*******************")
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer = TrainingPipeline()
    model_trainer.train()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e
