import os
import itertools
import mlflow
import logging
import concurrent.futures
from pyspark.sql import SparkSession

# Local module imports
from utils.dataPipeline import DataPreparation
from utils.ALSModelPipeline import ALSModelPipeline

# ==========================================
# 0. LOGGING & MLFLOW SETUP
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://gcp-prd-ds-mlflow.prd.wynk.internal/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

experiment_name = "Popularity_Coverage_ALS"

try:
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    logger.info(f"Connected to MLflow. Experiment ID: {exp.experiment_id}")
    
    # KILL ANY ZOMBIE RUNS FROM PREVIOUS CRASHES
    while mlflow.active_run():
        mlflow.end_run()
        
except Exception as e:
    logger.error(f"MLflow Connection Failed: {e}")
    exp = None

# ==========================================
# 1. SPARK INIT & CONFIG
# ==========================================
spark = SparkSession.builder \
    .appName("classified_training") \
    .config("spark.sql.shuffle.partitions", "4000") \
    .config("spark.executor.memoryOverhead", "4g") \
    .config("spark.sql.files.ignoreCorruptFiles", "true") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .config("spark.sql.parquet.mergeSchema", "true") \
    .config("spark.executor.failuresValidityInterval", "1h") \
    .config("spark.scheduler.mode", "FAIR") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

config = {
    "base_date_str": "2026-04-12",
    "db_path": "gs://wynk-ml-workspace/projects/xstream_nlu/catalog-db/",
    "temp_path": "gs://wynk-ml-workspace/_temp/harshith/als",
    "test_days": 30,
    "train_days": 90,
    "playtime_history_path": "gs://wynk-ml-workspace/projects/rails_reranking/daily_user_watch_history_new",
    "click_history_path": "gs://wynk-ml-workspace/projects/rails_reranking/daily_user_click_watch_history_new",
    "distinct_user_content_threshold": 3
}

# ==========================================
# 2. DATA PREP & CHECKPOINTING
# ==========================================
logger.info("Starting Data Preparation...")
decayed_data_prep = DataPreparation(spark, config, decay=True)
decayed_data_prep.prepare_features_and_metadata()
# Applies your structural fix for popularity bias
decayed_data_prep.build_indices_and_ground_truth(penalize_popularity=True)
decayed_data_prep.save_data_checkpoint()

# Reload from checkpoint to guarantee clean state for models
logger.info("Loading Fresh Data Checkpoint...")
fresh_data_prep = DataPreparation(spark, config)
fresh_data_prep.load_data_checkpoint()

# ==========================================
# 3. TRAINING FUNCTION
# ==========================================
import traceback
from mlflow.tracking import MlflowClient

# ==========================================
# 3. THREAD-SAFE TRAINING FUNCTION
# ==========================================
def train_and_log_metrics(params_dict):
    current_params = params_dict["params"]
    parent_id = params_dict["parent_run_id"]
    experiment_id = params_dict["exp_id"]
    
    als_params = {
        "coldStartStrategy": "drop",
        "k": 15,
        "implicitPrefs": True,
        **current_params
    }

    # 1. Use MlflowClient for thread-safe logging
    client = MlflowClient()
    
    # Create the run manually and attach to parent
    run = client.create_run(
        experiment_id=experiment_id,
        tags={"mlflow.parentRunId": parent_id, "mlflow.runName": f"Rank_{als_params['rank']}_Alpha_{als_params['alpha']}"}
    )
    run_id = run.info.run_id

    try:
        print(f"\n[STARTING] Run {run_id[-6:]} | Params: {current_params}")
        
        # Log params safely
        for k, v in als_params.items():
            client.log_param(run_id, k, v)

        # Train Model
        als_pipeline = ALSModelPipeline(spark, config, data_prep=fresh_data_prep)
        Model_Metrics = als_pipeline.train_ALS(als_params, print_metrics=True)

        # Log Metrics
        client.log_metric(run_id, "coverage", Model_Metrics['coverage'])
        client.log_metric(run_id, "precision_at_k", Model_Metrics['precision_k'])
        client.log_metric(run_id, "recall_at_k", Model_Metrics['avg_recall'])
        client.log_metric(run_id, "map", Model_Metrics['map'])

        print(f"[SUCCESS] Run {run_id[-6:]} | Coverage: {Model_Metrics['coverage']:.4f} | MAP: {Model_Metrics['map']:.4f}")
        
        # Manually close as finished
        client.set_terminated(run_id, status="FINISHED")
        return Model_Metrics

    except Exception as e:
        # If it crashes, CATCH the error, print it, and mark as failed
        print(f"\n[CRASHED] Run {run_id[-6:]} | Params: {current_params}")
        print(f"Error Message: {str(e)}")
        traceback.print_exc()  # This prints the actual Spark error so you can debug!
        
        client.set_terminated(run_id, status="FAILED")
        return None

# ==========================================
# 4. EXECUTION
# ==========================================
hyper_params = {
    "rank": [100],
    "maxIter": [10, 15],
    "alpha": [2.0, 3.0, 5.0,10.0, 20.0],
    "regParam": [1.0]
}

keys = hyper_params.keys()
grid = [dict(zip(keys, v)) for v in itertools.product(*hyper_params.values())]

logger.info(f"Total hyperparameter combinations to test: {len(grid)}")

# Start the main parent run
with mlflow.start_run(run_name="Parallel_Grid_Search_Run") as parent_run:
    parent_run_id = parent_run.info.run_id
    
    # Package parameters, including the exp_id for the thread-safe client
    tasks = [
        {
            "params": combo, 
            "parent_run_id": parent_run_id,
            "exp_id": exp.experiment_id
        } 
        for combo in grid
    ]
    
    # I highly recommend lowering this to 2 or 3 to avoid Spark OOM errors
    max_workers = 4 
    print(f"Submitting {len(tasks)} jobs, running {max_workers} at a time...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(train_and_log_metrics, tasks))

logger.info("Parallel grid search complete.")