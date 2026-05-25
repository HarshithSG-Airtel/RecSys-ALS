import ast
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import datetime, timedelta
import itertools

from sklearn import logger

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics

from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k

import mlflow
import mlflow.spark

import mlflow

# 1. Set the URI
mlflow.set_tracking_uri("http://gcp-prd-ds-mlflow.prd.wynk.internal/") 

# 2. Robust Experiment Loading
experiment_name = "ALS_6m"

try:
    # Try to set it (this creates it if it doesn't exist in most versions)
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    
    if exp is None:
        # If it still isn't found, try creating it explicitly
        exp_id = mlflow.create_experiment(experiment_name)
        exp = mlflow.get_experiment(exp_id)
        
    logger.info(f"Successfully connected to MLflow. Experiment ID: {exp.experiment_id}")
    
except Exception as e:
    logger.error(f"MLflow Connection Failed: {e}")
    logger.warning("Continuing training WITHOUT MLflow logging...")
    exp = None

# Initialize Spark
spark = SparkSession.builder \
    .appName("als_tuning") \
    .config("spark.sql.shuffle.partitions", "4000") \
    .config("spark.executor.memoryOverhead", "4g") \
    .config("spark.sql.files.ignoreCorruptFiles", "true") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .config("spark.sql.parquet.mergeSchema", "true") \
    .getOrCreate()

class RecommendationDataPipeline:
    def __init__(self, spark, config):
        self.spark = spark
        self.config = config
        self.train_df = None
        self.test_df = None
        self.metadata_df = None
        self.indexed_train = None
        self.indexed_test = None
        
    def _get_valid_paths(self, base_path, start_days_ago, end_days_ago):
        base_date = datetime.strptime(self.config['date'], "%Y-%m-%d")
        paths = []
        for i in range(start_days_ago, end_days_ago):
            target_date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            paths.append(f"{base_path}/day={target_date}")
        return paths

    def _read_and_union_paths(self, paths):
        df = None
        for path in paths:
            try:
                temp_df = self.spark.read.parquet(path)
                df = temp_df if df is None else df.unionByName(temp_df)
            except Exception:
                pass # Skipping missing paths
        return df

    def load_raw_data(self):
        print("Loading raw interactions and metadata...")
        test_paths = self._get_valid_paths(self.config['watch_history_path'], 0, self.config['test_days'])
        train_paths = self._get_valid_paths(self.config['watch_history_path'], self.config['test_days'], self.config['test_days'] + self.config['train_days'])
        
        self.test_df = self._read_and_union_paths(test_paths)
        self.train_df = self._read_and_union_paths(train_paths)
        
        # Load Metadata
        tv_df = self.spark.read.parquet(f"{self.config['db_path']}{self.config['date']}/enriched_tv.parquet")
        movie_df = self.spark.read.parquet(f"{self.config['db_path']}{self.config['date']}/enriched_movie.parquet")
        
        def clean_meta(df):
            return (df.filter((F.col('XstreamContentIds') != F.array()) & (F.col("published") == True))
                    .withColumn("item_id_exploded", F.explode("XstreamContentIds")) # Step 1: Explode
                    .select(
                        F.col("item_id_exploded").cast("string").alias("item_id"), # Step 2: Cast
                        "title", 
                        F.col('OriginalLanguage').alias('original_language').cast("string"),
                        "Genres"
                    ))
        self.metadata_df = clean_meta(tv_df).unionByName(clean_meta(movie_df)).distinct()
        

    def process_and_index_data(self):
        print("Filtering users, aggregating playtime, and building indexes...")
        
        # 1. Overlapping Users & Aggregation
        common_users = self.train_df.select("userId").distinct().join(self.test_df.select("userId").distinct(), "userId")
        
        train_filtered = self.train_df.join(common_users, "userId")
        test_filtered = self.test_df.join(common_users, "userId")
        
        train_stats = train_filtered.groupBy("userId", "item_id") \
            .agg(F.sum("total_play_time_sec").alias("total_playtime_combined")) \
            .withColumn("distinct_content_count", F.count("item_id").over(Window.partitionBy("userId"))) \
            .filter(F.col("total_playtime_combined").isNotNull() & ~F.isnan("total_playtime_combined"))
            
        als_input_base = train_stats.filter("distinct_content_count >= 2")
        
        # Re-filter test users based on the >= 2 rule
        valid_users = als_input_base.select("userId").distinct()
        test_filtered = test_filtered.join(valid_users, "userId")
        
        # 2. Build Lookup Tables
        distinct_users = valid_users.rdd.zipWithIndex().toDF(["user_struct", "userIndex"]).select(F.col("user_struct.*"), F.col("userIndex").cast("int"))
        distinct_items = als_input_base.select("item_id").distinct().rdd.zipWithIndex().toDF(["item_struct", "itemIndex"]).select(F.col("item_struct.*"), F.col("itemIndex").cast("int"))
        
        # Break Lineage!
        distinct_users.write.mode("overwrite").parquet(f"{self.config['temp_path']}/user_lookup")
        distinct_items.write.mode("overwrite").parquet(f"{self.config['temp_path']}/item_lookup")
        
        user_lookup = self.spark.read.parquet(f"{self.config['temp_path']}/user_lookup")
        self.item_lookup = self.spark.read.parquet(f"{self.config['temp_path']}/item_lookup")

        # 3. Apply Indexes
        self.indexed_train = als_input_base.join(user_lookup, "userId").join(self.item_lookup, "item_id") \
            .withColumn("playtime_logged", F.log1p("total_playtime_combined")) \
            .select("userIndex", "itemIndex", "playtime_logged").repartition(1000).cache()
            
        self.indexed_test = test_filtered.join(user_lookup, "userId").join(self.item_lookup, "item_id") \
            .withColumn("playtime_logged", F.log1p("total_play_time_sec")) \
            .select("userIndex", "itemIndex", "playtime_logged").cache()

    def get_als_data(self):
        """Returns DataFrames optimized for Spark ALS."""
        ground_truth = self.indexed_test.groupBy("userIndex").agg(F.collect_set("itemIndex").alias("actual_items"))
        return self.indexed_train, ground_truth

    def get_lightfm_data(self, sample_fraction=0.05):
        """Samples, extracts features, and builds Scipy matrices for LightFM."""
        print(f"Preparing LightFM Matrices (Sampling {sample_fraction * 100}% of users)...")
        
        # Scrub test data of seen items
        clean_test = self.indexed_test.join(self.indexed_train.select("userIndex", "itemIndex"), on=["userIndex", "itemIndex"], how="left_anti")
        
        # Sample users to fit in driver memory
        sampled_users = self.indexed_train.select("userIndex").distinct().sample(withReplacement=False, fraction=sample_fraction, seed=42)
        
        train_pdf = self.indexed_train.join(sampled_users, "userIndex").toPandas()
        test_pdf = clean_test.join(sampled_users, "userIndex").toPandas()
        
        items_pdf = self.metadata_df.join(self.item_lookup, "item_id").select("itemIndex", "original_language", "Genres").dropDuplicates(["itemIndex"]).toPandas()
        
        # Clean & extract metadata features
        items_pdf['Genres'] = items_pdf['Genres'].fillna('Unknown').astype(str)
        items_pdf['original_language'] = items_pdf['original_language'].fillna('Unknown').astype(str)
        
        valid_items = np.unique(np.concatenate((train_pdf['itemIndex'], test_pdf['itemIndex'])))
        items_pdf = items_pdf[items_pdf['itemIndex'].isin(valid_items)]
        
        def extract_features(lang, genre_data):
            feats = [str(lang)]
            if genre_data.startswith('['):
                try: genre_data = ast.literal_eval(genre_data)
                except: genre_data = genre_data.replace('[', '').replace(']', '').replace("'", "").replace('"', "").split(',')
            else:
                genre_data = genre_data.split(',')
            feats.extend([g.strip() for g in genre_data])
            return list(set(feats))
            
        items_pdf['clean_features'] = items_pdf.apply(lambda row: extract_features(row['original_language'], row['Genres']), axis=1)
        all_features = list(set([feat for sublist in items_pdf['clean_features'] for feat in sublist]))
        
        # Build LightFM Dataset
        dataset = Dataset()
        all_users = np.unique(np.concatenate((train_pdf['userIndex'], test_pdf['userIndex'])))
        dataset.fit(users=all_users, items=valid_items, item_features=all_features)
        
        train_interact, train_weights = dataset.build_interactions(zip(train_pdf['userIndex'], train_pdf['itemIndex'], train_pdf['playtime_logged']))
        test_interact, _ = dataset.build_interactions(zip(test_pdf['userIndex'], test_pdf['itemIndex'], test_pdf['playtime_logged']))
        item_features = dataset.build_item_features((idx, feats) for idx, feats in zip(items_pdf['itemIndex'], items_pdf['clean_features']))
        
        return train_interact, train_weights, test_interact, item_features
    
config = {
    "date": "2026-03-22",
    "train_days": 175,
    "test_days": 30,
    "watch_history_path": "gs://wynk-ml-workspace/projects/rails_reranking/daily_user_watch_history_new",
    "db_path": "gs://wynk-ml-workspace/projects/xstream_nlu/catalog-db/",
    "temp_path": "gs://wynk-ml-workspace/_temp/harshith/als"
}

pipeline = RecommendationDataPipeline(spark, config)
pipeline.load_raw_data()
pipeline.process_and_index_data()

def tune_als(train_data, ground_truth, param_grid):
    print("Starting ALS Hyperparameter Tuning...")
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for params in combinations:
        with mlflow.start_run(run_name=f"ALS_rank_{params['rank']}_reg_{params['regParam']}"):
            mlflow.log_params(params)
            mlflow.log_param("model_type", "ALS")
            
            als = ALS(
                userCol="userIndex", itemCol="itemIndex", ratingCol="playtime_logged",
                implicitPrefs=True, maxIter=10, coldStartStrategy="drop",
                rank=params['rank'], regParam=params['regParam']
            )
            
            model = als.fit(train_data)
            
            # Evaluate
            user_recs = model.recommendForAllUsers(15)
            user_recs_flat = user_recs.select("userIndex", F.col("recommendations.itemIndex").alias("predicted_items"))
            eval_data = user_recs_flat.join(ground_truth, "userIndex").select("predicted_items", "actual_items").rdd.map(tuple)
            
            metrics = RankingMetrics(eval_data)
            map_metric = metrics.meanAveragePrecision
            precision_15 = metrics.precisionAt(15)
            
            mlflow.log_metrics({
                "MAP": map_metric,
                "Precision_at_15": precision_15,
                "Recall_at_15": metrics.recallAt(15)
            })
            print(f"ALS Params: {params} -> MAP: {map_metric:.4f}, P@15: {precision_15:.4f}")

# 1. Fetch data formats
als_train, als_ground_truth = pipeline.get_als_data()

# 2. Define Grids
als_param_grid = {
    "rank": [10, 20, 50, 100, 150, 200],
    "regParam": [0.01, 0.1]
}


tune_als(als_train, als_ground_truth, als_param_grid)

print("Tuning complete! View results on your MLflow UI.")