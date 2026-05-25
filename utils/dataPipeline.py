from datetime import datetime, timedelta

# PySpark SQL imports
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.utils import AnalysisException
from pyspark.sql import SparkSession

# If you use 'col' or 'lower' explicitly without the 'F.' prefix:
from pyspark.sql.functions import col, lower

def connect_spark_session():
    app_name = "RecSysPipeline"
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "4000") \
        .config("spark.executor.memoryOverhead", "4g") \
        .config("spark.sql.files.ignoreCorruptFiles", "true") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.parquet.mergeSchema", "true") \
        .config("spark.executor.failuresValidityInterval", "1h")\
        .config("spark.yarn.executor.failuresValidityInterval", "1h") \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    # Handle both Log4j1 (older Spark) and Log4j2 (Spark 3+)
    log4j = sc._jvm.org.apache.log4j if hasattr(sc._jvm.org.apache, "log4j") else sc._jvm.org.apache.logging.log4j

    # 2. Target the exact noisy Java classes and force them to ERROR level
    log4j.LogManager.getLogger("org.apache.spark.scheduler.DAGScheduler").setLevel(log4j.Level.ERROR)
    log4j.LogManager.getLogger("org.apache.spark.SparkConf").setLevel(log4j.Level.ERROR)

    # Optional: Nuke all org.apache logging just to be safe
    log4j.LogManager.getLogger("org.apache").setLevel(log4j.Level.ERROR)

    print(f"Connecting to Spark Session {app_name}...")
    return spark

# ==========================================
# 1. DATA PREPARATION CLASS
# ==========================================

class DataPreparation:
    """
    Handles data reading, validation, transformations, and index building 
    for recommendation pipelines.
    """
    
    def __init__(self, spark, config, decay=False):
        self.spark = spark
        self.config = config
        self.decay = decay
        
        # Raw & Processed DataFrames
        self.train_df = None
        self.test_df = None
        self.labeled_stats = None
        self.playtime_test = None
        self.click_test = None
        self.click_train = None
        self.combined_train_df = None
        self.combined_test_df = None
        self.metadata_df = None
        self.item_popularity = None

        # Lookup & Index DataFrames
        self.indexed_test = None
        self.indexed_train = None
        self.ground_data = None
        self.item_lookup = None
        self.user_lookup = None

    def _get_valid_paths(self, base_path, start_days_ago, end_days_ago):
        base_date = datetime.strptime(self.config['base_date_str'], "%Y-%m-%d")
        paths_with_metadata = []
        for i in range(start_days_ago, end_days_ago):
            target_date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            paths_with_metadata.append({
                "path": f"{base_path}/day={target_date}", 
                "days_ago": i
            })
        return paths_with_metadata

    def _read_valid_paths_flat(self, path_dicts):
        valid_paths = []
        for p in path_dicts:
            path_string = p["path"]
            try:
                self.spark.read.parquet(path_string).schema
                valid_paths.append(path_string)
            except AnalysisException:
                print(f"Path not found or inaccessible, skipping: {path_string}")
            except Exception as e:
                print(f"Unexpected error validating {path_string}: {e}")
        
        if not valid_paths:
            print("Warning: No valid paths found in the provided range.")
            return None
            
        return self.spark.read.parquet(*valid_paths)

    def _read_and_filter_data(self, daily_watch_history_path):
        base_date_str = self.config["base_date_str"]
        test_days = self.config["test_days"]
        train_days = self.config["train_days"]
        
        test_path_dicts = self._get_valid_paths(daily_watch_history_path, 0, test_days)
        train_path_dicts = self._get_valid_paths(daily_watch_history_path, test_days, test_days + train_days)

        test_df = self._read_valid_paths_flat(test_path_dicts)
        train_df = self._read_valid_paths_flat(train_path_dicts)

        # Apply decay factor globally based on the 'day' column, regardless of data type
        if self.decay:
            train_window = train_days - 1
            train_df = train_df.withColumn("days_ago", F.datediff(F.lit(base_date_str).cast("date"), F.col("day"))) \
                .withColumn("decay_factor", 1.0 - (0.5 * ((F.col("days_ago") - test_days) / max(1, train_window))))
            
            # Test data gets a 1.0 factor to keep schemas aligned
            test_df = test_df.withColumn("decay_factor", F.lit(1.0))

        print(f"Data read from {daily_watch_history_path} completed, filtering for common users...")
        
        train_users = train_df.select("userId").distinct()
        test_users = test_df.select("userId").distinct()
        common_users = train_users.join(test_users, on="userId", how="inner")

        df_train_filtered = train_df.join(common_users, on="userId", how="inner")
        df_test_filtered = test_df.join(common_users, on="userId", how="inner")

        return df_train_filtered, df_test_filtered

    def _aggregate_user_playtime(self, watch_history_df):
        agg_exprs = [F.sum("total_play_time_sec").alias("total_playtime_combined")]
        
        if self.decay:
            # We just grab the max decay factor (most recent watch) for this user-item combo
            agg_exprs.append(F.max("decay_factor").alias("recent_decay_factor"))

        user_stats = watch_history_df.groupBy("userId", "item_id") \
            .agg(*agg_exprs) \
            .filter(F.col("total_playtime_combined").isNotNull() & ~F.isnan("total_playtime_combined"))
        
        print("Playtime combined successfully.")
        return user_stats

    def _assign_confidence_scores(self, train_combined_stats):
        # Confidence is ranked on pure, un-decayed total playtime
        # 1. Create a window specifically to count the total rows per item_id
        count_window = Window.partitionBy("item_id")

        # 2. Add a count column, filter out items with fewer than 5 rows, then drop the helper column
        filtered_stats = train_combined_stats \
            .withColumn("item_count", F.count("*").over(count_window)) \
            .filter(F.col("item_count") >= 5) \
            .drop("item_count")

        # 3. Apply your existing percent_rank logic to the newly filtered DataFrame
        window_spec = Window.partitionBy("item_id").orderBy("total_playtime_combined")
        df_ranked = filtered_stats.withColumn("rank", F.percent_rank().over(window_spec))

        labeled_stats = df_ranked.withColumn("confidence", 
            F.when(F.col("rank") <= 0.20, 0.2)
            .when(F.col("rank") <= 0.40, 1.0)
            .when(F.col("rank") <= 0.60, 2.0)
            .when(F.col("rank") <= 0.80, 3.0)
            .otherwise(4.0)
        ).filter(F.col("confidence") > 0).drop("rank", "total_playtime_combined")

        return labeled_stats
    
    def _get_labeled_playtime_data(self):
        playtime_train, playtime_test = self._read_and_filter_data(self.config["playtime_history_path"])
        self.train_df = playtime_train
        self.test_df = playtime_test
        print("self train and test df have been assigned")
        
        train_combined_stats = self._aggregate_user_playtime(playtime_train)
        test_combined_stats = self._aggregate_user_playtime(playtime_test)
        labeled_stats = self._assign_confidence_scores(train_combined_stats)
        print("Playtime to label completed.")
        return labeled_stats, test_combined_stats
    
    # def _remove_sport_data(self, df):
    #     sport_df = df.filter(lower(col("item_id")).contains("match"))

    def prepare_features_and_metadata(self):
        playtime_train, playtime_test = self._get_labeled_playtime_data()
        self.labeled_stats = playtime_train
        self.playtime_test = playtime_test

        click_train, click_test = self._read_and_filter_data(self.config["click_history_path"])
        self.click_test = click_test
        self.click_train = click_train
        
        if self.decay:
            playtime_cols = ["userId", "item_id", "confidence", "recent_decay_factor"]
            pt_train_features = self.labeled_stats.select(*playtime_cols)
            
            # Group clicks, get max decay factor, assign static 2.0 confidence
            ct_train_features = click_train.filter(F.col("quality_click_flag") == 1) \
                .groupBy("userId", "item_id") \
                .agg(F.max("decay_factor").alias("recent_decay_factor")) \
                .withColumn("confidence", F.lit(2.0)) \
                .select(*playtime_cols)
                
            combined_train_df = pt_train_features.unionByName(ct_train_features) \
                .groupBy("userId", "item_id") \
                .agg(
                    F.max("confidence").alias("confidence"),
                    F.max("recent_decay_factor").alias("recent_decay_factor")
                )
        else:
            playtime_cols = ["userId", "item_id", "confidence"]
            pt_train_features = self.labeled_stats.select(*playtime_cols)
            
            ct_train_features = click_train.filter(F.col("quality_click_flag") == 1) \
                .select("userId", "item_id").distinct() \
                .withColumn("confidence", F.lit(2.0))
                
            combined_train_df = pt_train_features.unionByName(ct_train_features) \
                .groupBy("userId", "item_id") \
                .agg(F.max("confidence").alias("confidence"))

        playtime_test_features = self.playtime_test.select("userId", "item_id")
        click_test_features = self.click_test.filter(F.col("quality_click_flag") == 1).select("userId", "item_id")
        self.combined_test_df = playtime_test_features.unionByName(click_test_features).distinct()\
                    .filter(~F.col("item_id").contains("MATCH") & ~F.col("item_id").contains("HIGHLIGHTS"))
        self.combined_train_df = combined_train_df.filter(~F.col("item_id").contains("MATCH") & ~F.col("item_id").contains("HIGHLIGHTS"))
        
        
        print("Successfully combined training and testing features!")

        base_db_path = f"{self.config['db_path']}{self.config['base_date_str']}"
        tv_df = self.spark.read.parquet(f"{base_db_path}/enriched_tv.parquet")
        movie_df = self.spark.read.parquet(f"{base_db_path}/enriched_movie.parquet")
        
        def clean_meta(df):
            return df.filter((F.col('XstreamContentIds') != F.array()) & (F.col("published") == True)) \
                .withColumn("item_id_exploded", F.explode("XstreamContentIds")) \
                .select(
                    F.col("item_id_exploded").cast("string").alias("item_id"),
                    "title", 
                    F.col('OriginalLanguage').alias('original_language').cast("string"),
                    "Genres"
                )
                    
        self.metadata_df = clean_meta(tv_df).unionByName(clean_meta(movie_df)).distinct()

        if not self.combined_train_df or not self.combined_test_df or not self.metadata_df:
            raise ValueError("Critical Error: One or more DataFrames are empty/None. Check your paths!")

    def add_custom_user_data(self, custom_users, confidence_score=4.0):
        if self.combined_train_df is None:
            raise ValueError("combined_train_df is missing. Run prepare_features_and_metadata() first.")
            
        normalized_data = []
        for user in custom_users:
            user_id = user.get("userId") or user.get("userID") 
            history = user.get("watch_history", [])
            
            if isinstance(history, str):
                items = [item.strip() for item in history.split(",") if item.strip()]
            elif isinstance(history, (list, set, tuple)):
                items = [str(item).strip() for item in history if str(item).strip()]
            else:
                continue
                
            if items:
                normalized_data.append((user_id, items))
                
        if not normalized_data:
            return

        custom_df = self.spark.createDataFrame(
            normalized_data, 
            schema=["userId", "watch_history_array"]
        )
        
        exploded_df = custom_df.select(
            F.col("userId"),
            F.explode("watch_history_array").alias("item_id")
        ).withColumn("confidence", F.lit(float(confidence_score)))
        
        # If decay is active, give manual injections a decay factor of 1.0 (no decay)
        if self.decay:
            exploded_df = exploded_df.withColumn("recent_decay_factor", F.lit(1.0))
            agg_exprs = [F.max("confidence").alias("confidence"), F.max("recent_decay_factor").alias("recent_decay_factor")]
        else:
            agg_exprs = [F.max("confidence").alias("confidence")]
            
        self.combined_train_df = self.combined_train_df.unionByName(exploded_df) \
            .groupBy("userId", "item_id") \
            .agg(*agg_exprs)
            
        print(f"Successfully added {len(normalized_data)} custom users to the training data.")

    def build_indices_and_ground_truth(self, penalize_popularity=True):
        print("Filtering users, aggregating playtime, and building indexes...")
        train_data = self.combined_train_df

        # 1. APPLY TIME DECAY (If enabled)
        if self.decay:
            train_data = train_data.withColumn("confidence", F.col("confidence") * F.col("recent_decay_factor"))
            print("Applied time decay to confidence scores.")

        # ==========================================
        # NEW: POPULARITY PENALTY (INVERSE FREQUENCY)
        # ==========================================

        # 1. Count users per item
        item_popularity = train_data.groupBy("item_id").agg(
            F.count("userId").alias("user_count")
        )

        # 2. Find the bounds
        stats = item_popularity.select(
            F.max("user_count").alias("max_c"), 
            F.min("user_count").alias("min_c")
        ).collect()[0]

        max_c = stats["max_c"]
        min_c = stats["min_c"]

        # 3. Calculate the 1-100 scale
        # Formula: ((count - min) / (max - min)) * 99 + 1
        item_popularity = item_popularity.withColumn(
            "popularity_score", 
            ((F.col("user_count") - min_c) / (max_c - min_c)) * 99.0 + 1.0
        )

        # Optional: Round it to a whole number for cleaner reading
        item_popularity = item_popularity.withColumn(
            "popularity_score", 
            F.round(F.col("popularity_score")).cast("integer")
        )

        # 4. Apply logarithmic penalty (Kept as a small fraction because it's used as a multiplier!)
        item_popularity = item_popularity.withColumn(
            "popularity_penalty", 
            1.0 / F.log10(F.col("user_count") + 10.0)
        )

        self.item_popularity = item_popularity.select(
            "item_id", "user_count", "popularity_score", "popularity_penalty"
        )

        if penalize_popularity:
            print("Calculating item frequencies and applying popularity penalty...")
            
            # Broadcast the small item_popularity dataframe to avoid expensive shuffles
            train_data = train_data.join(
                F.broadcast(self.item_popularity), 
                on="item_id", 
                how="inner"
            )
            
            # Scale the final confidence score using the small fraction penalty
            train_data = train_data.withColumn(
                "confidence", 
                F.col("confidence") * F.col("popularity_penalty")
            ).drop("user_count", "popularity_penalty", "popularity_score")
            
            print("Popularity penalty successfully applied to confidence scores.")

        # 2. FILTER USERS BY ACTIVITY THRESHOLD
        train_data = train_data.withColumn("distinct_content_count", F.count("item_id").over(Window.partitionBy("userId")))
        user_content_counts = train_data.select("userId", "distinct_content_count").distinct()
        
        # (Optional) You had p95_threshold commented out; leaving it as you had it.
        # p95_threshold = user_content_counts.stat.approxQuantile("distinct_content_count", [0.95], 0.001)[0]
        
        als_input_base = train_data.filter(
            (F.col("distinct_content_count") >= self.config["distinct_user_content_threshold"])
            # & (F.col("distinct_content_count") <= p95_threshold)
        )
        
        # 3. BUILD LOOKUPS (USER & ITEM INDICES)
        valid_users = als_input_base.select("userId").distinct()
        test_filtered = self.combined_test_df.join(valid_users, "userId")
        
        distinct_users = valid_users.rdd.zipWithIndex().toDF(["user_struct", "userIndex"]) \
            .select(F.col("user_struct.*"), F.col("userIndex").cast("int"))
        distinct_items = als_input_base.select("item_id").distinct().rdd.zipWithIndex().toDF(["item_struct", "itemIndex"]) \
            .select(F.col("item_struct.*"), F.col("itemIndex").cast("int"))
        
        # Write to disk
        distinct_users.write.mode("overwrite").parquet(f"{self.config['temp_path']}/user_lookup")
        distinct_items.write.mode("overwrite").parquet(f"{self.config['temp_path']}/item_lookup")
        
        self.user_lookup = self.spark.read.parquet(f"{self.config['temp_path']}/user_lookup")
        self.item_lookup = self.spark.read.parquet(f"{self.config['temp_path']}/item_lookup")

        # 4. MAP INDICES BACK TO TRAINING AND TEST DATA
        self.indexed_train = als_input_base.join(self.user_lookup, "userId").join(self.item_lookup, "item_id") \
            .select("userIndex", "itemIndex", "confidence").repartition(1000).cache()
            
        self.indexed_test = test_filtered.join(self.user_lookup, "userId").join(self.item_lookup, "item_id") \
            .select("userIndex", "itemIndex").cache()
            
        # 5. GENERATE GROUND TRUTH FOR EVALUATION
        print("Generating Ground Truth...")
        self.ground_data = self.indexed_test.groupBy("userIndex").agg(F.collect_set("itemIndex").alias("actual_items")).cache()
    def get_als_data(self):
        return self.indexed_train, self.ground_data


    def save_data_checkpoint(self, checkpoint_folder="default_checkpoint"):
        checkpoint_path = f"{self.config['temp_path']}/{checkpoint_folder}"
        print(f"Saving data checkpoint to: {checkpoint_folder}...")
        
        # Note: als_top_k was removed from here because it belongs to the model output now.
        dfs_to_save = {
            "indexed_test": self.indexed_test,
            "indexed_train": self.indexed_train,
            "ground_data": self.ground_data,
            "item_lookup": self.item_lookup,
            "user_lookup": self.user_lookup,
            "combined_train_df": self.combined_train_df,
            "combined_test_df": self.combined_test_df,
            "metadata_df": self.metadata_df,
            "item_popularity": self.item_popularity
        }
            
        for name, df in dfs_to_save.items():
            if df is not None:
                save_dir = f"{checkpoint_path}/{name}"
                print(f"  Saving {name}...")
                df.write.mode("overwrite").parquet(save_dir)
            else:
                print(f"  Warning: {name} is None, skipping.")
                
        print("Checkpoint successfully saved!")

    def load_data_checkpoint(self, checkpoint_folder="default_checkpoint"):
        checkpoint_path = f"{self.config['temp_path']}/{checkpoint_folder}"
        print(f"Loading data checkpoint from: {checkpoint_path}...")
        
        try:
            self.indexed_train = self.spark.read.parquet(f"{checkpoint_path}/indexed_train").cache()
            self.ground_data = self.spark.read.parquet(f"{checkpoint_path}/ground_data").cache()
            self.metadata_df = self.spark.read.parquet(f"{checkpoint_path}/metadata_df")
            self.user_lookup = self.spark.read.parquet(f"{checkpoint_path}/user_lookup")
            self.item_lookup = self.spark.read.parquet(f"{checkpoint_path}/item_lookup")
            self.combined_train_df = self.spark.read.parquet(f"{checkpoint_path}/combined_train_df")
            self.combined_test_df = self.spark.read.parquet(f"{checkpoint_path}/combined_test_df")
            self.indexed_test = self.spark.read.parquet(f"{checkpoint_path}/indexed_test").cache()
            self.item_popularity = self.spark.read.parquet(f"{checkpoint_path}/item_popularity").cache()

            print("Checkpoint loaded successfully! Ready for training or inference.")
        except Exception as e:
            print(f"Critical Error loading checkpoint: {e}")
            raise RuntimeError(f"Failed to load datasets. Ensure checkpoint exists.")
        
    def print_data_funnel_metrics(self):
        """
        The absolute maximum level of matrix diagnostics. 
        Combines raw volumes, distribution percentiles, sparsity, 
        skew inequalities, and implicit signal variance.
        """
        if self.indexed_train is None:
            print("Error: indexed_train not found. Run build_indices_and_ground_truth() first.")
            return

        print("\n" + "="*80)
        print("OMEGA-LEVEL MATRIX DIAGNOSTICS: TOTAL PIPELINE EXPOSURE")
        print("="*80)

        df = self.indexed_train.cache()
        
        # ---------------------------------------------------------
        # 1. THE FOUNDATION (Raw Volumes & Ratios)
        # ---------------------------------------------------------
        unique_interactions = df.count()
        unique_users = df.select("userIndex").distinct().count()
        unique_items = df.select("itemIndex").distinct().count()
        
        total_possible = unique_users * unique_items
        sparsity = (1.0 - (unique_interactions / total_possible)) * 100.0 if total_possible > 0 else 100.0
        user_item_ratio = unique_users / unique_items if unique_items > 0 else 0

        print(f"\n[1. BASE DIMENSIONS & MATRIX SHAPE]")
        print(f"  Filled Matrix Cells:     {unique_interactions:,}")
        print(f"  Unique Users (U):        {unique_users:,}")
        print(f"  Unique Items (I):        {unique_items:,}")
        print(f"  Exact Sparsity:          {sparsity:.6f}%")
        print(f"  Matrix Shape Factor:     {user_item_ratio:.2f} Users for every 1 Item")

        # ---------------------------------------------------------
        # 2. USER ENGAGEMENT (Distributions, Zombies & Whales)
        # ---------------------------------------------------------
        user_counts = df.groupBy("userIndex").agg(F.count("*").alias("u_count")).cache()
        u_q = user_counts.stat.approxQuantile("u_count", [0.0, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0], 0.01)
        
        zombie_users = user_counts.filter("u_count == 1").count()
        cold_users = user_counts.filter("u_count < 5").count()
        binge_quotient = (u_q[7] / unique_items) * 100 if unique_items > 0 else 0

        print(f"\n[2. USER ENGAGEMENT SKEW]")
        print(f"  Distributions (Items per User):")
        print(f"    Min/25th/Median:       {u_q[0]:.0f} / {u_q[1]:.0f} / {u_q[2]:.0f}")
        print(f"    75th/90th/95th:        {u_q[3]:.0f} / {u_q[4]:.0f} / {u_q[5]:.0f}")
        print(f"    99th / Max (Whale):    {u_q[6]:.0f} / {u_q[7]:.0f}")
        print(f"  Anomalies:")
        print(f"    Zombie Users (=1 rec): {zombie_users:,} ({(zombie_users/unique_users)*100:.1f}%)")
        print(f"    Cold Users (<5 recs):  {cold_users:,} ({(cold_users/unique_users)*100:.1f}%)")
        print(f"    Binge Quotient:        Max user watched {binge_quotient:.2f}% of entire catalog")

        # ---------------------------------------------------------
        # 3. ITEM POPULARITY (Blockbusters & One-Hit Wonders)
        # ---------------------------------------------------------
        item_counts = df.groupBy("itemIndex").agg(F.count("*").alias("i_count")).cache()
        i_q = item_counts.stat.approxQuantile("i_count", [0.0, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0], 0.01)

        one_hit_wonders = item_counts.filter("i_count == 1").count()
        cold_items = item_counts.filter("i_count < 5").count()
        
        # Gini Wealth Proxy (Top 10% vs Bottom 50%)
        top_10_threshold = i_q[4]
        bottom_50_threshold = i_q[2]
        
        top_10_traffic = item_counts.filter(F.col("i_count") >= top_10_threshold).agg(F.sum("i_count")).collect()[0][0] or 0
        bottom_50_traffic = item_counts.filter(F.col("i_count") <= bottom_50_threshold).agg(F.sum("i_count")).collect()[0][0] or 0
        
        top_10_pct = (top_10_traffic / unique_interactions) * 100
        bottom_50_pct = (bottom_50_traffic / unique_interactions) * 100

        print(f"\n[3. ITEM POPULARITY SKEW]")
        print(f"  Distributions (Users per Item):")
        print(f"    Min/25th/Median:       {i_q[0]:.0f} / {i_q[1]:.0f} / {i_q[2]:.0f}")
        print(f"    75th/90th/95th:        {i_q[3]:.0f} / {i_q[4]:.0f} / {i_q[5]:.0f}")
        print(f"    99th / Max (Hit):      {i_q[6]:.0f} / {i_q[7]:.0f}")
        print(f"  Anomalies & Wealth Skew:")
        print(f"    One-Hit Wonders:       {one_hit_wonders:,} ({(one_hit_wonders/unique_items)*100:.1f}%)")
        print(f"    Cold Items (<5 users): {cold_items:,} ({(cold_items/unique_items)*100:.1f}%)")
        print(f"    Gini Proxy:            Top 10% of items hoard {top_10_pct:.1f}% of all traffic.")
        print(f"                           Bottom 50% of items share only {bottom_50_pct:.1f}% of traffic.")

        # ---------------------------------------------------------
        # 4. IMPLICIT SIGNAL DEGRADATION (Confidence Analysis)
        # ---------------------------------------------------------
        conf_stats = df.select(
            F.mean("confidence").alias("mean"),
            F.stddev("confidence").alias("stddev"),
            F.max("confidence").alias("max"),
            F.min("confidence").alias("min")
        ).collect()[0]

        decay_graveyard = df.filter("confidence <= 0.1").count()
        top_tier = df.filter("confidence >= 3.0").count()

        print(f"\n[4. IMPLICIT SIGNAL VARIANCE & DEGRADATION]")
        print(f"  Confidence Min / Max:    {conf_stats['min']:.4f} / {conf_stats['max']:.4f}")
        print(f"  Confidence Mean:         {conf_stats['mean']:.4f}")
        print(f"  Confidence StdDev:       {conf_stats['stddev']:.4f} (Must be > 0.2 to matter)")
        print(f"  Decay Graveyard (<=0.1): {decay_graveyard:,} rows ({(decay_graveyard/unique_interactions)*100:.2f}%)")
        print(f"  Top-Tier Saturation:     {top_tier:,} rows ({(top_tier/unique_interactions)*100:.2f}%) are scored >= 3.0")

        # ---------------------------------------------------------
        # 5. EXACT CONFIDENCE SPREAD (Score Bucketing)
        # ---------------------------------------------------------
        print(f"\n[5. EXACT CONFIDENCE SPREAD]")
        # Rounding to 1 decimal to group the endless float variations caused by decay penalties
        conf_dist = df.withColumn("conf_rounded", F.round("confidence", 1)) \
                      .groupBy("conf_rounded").count().orderBy("conf_rounded")
        
        # Taking top 15 most common scores to avoid blowing up the console if math gets weird
        conf_rows = conf_dist.orderBy(F.col("count").desc()).limit(15).collect()
        
        for row in sorted(conf_rows, key=lambda x: x['conf_rounded']):
            pct = (row["count"] / unique_interactions) * 100.0
            print(f"  Score ~{row['conf_rounded']:>4.1f}: {row['count']:>10,} ({pct:>5.1f}%)")

        print("\n" + "="*80 + "\n")
        
        # Cleanup cached DFs
        user_counts.unpersist()
        item_counts.unpersist()
