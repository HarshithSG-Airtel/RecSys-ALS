# PySpark ML Recommendation imports
from pyspark.ml.recommendation import ALS, ALSModel

# Evaluation Metrics
from pyspark.mllib.evaluation import RankingMetrics

# Standard SQL Functions
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ==========================================
# 2. MODEL PIPELINE CLASS
# ==========================================
class ALSModelPipeline:
    """
    Handles training, evaluation, and inference for the ALS recommendation model.
    """
    
    def __init__(self, spark, config, data_prep=None):
        self.spark = spark
        self.config = config
        self.data_prep = data_prep # Takes the prepared data object
        
        # State variables for our model and predictions
        self.als_model = None
        self.als_raw_50 = None     # Used for Coverage
        self.als_novel_15 = None   # Used for Recall/Precision and Inference

    # def _generate_novel_recommendations(self, raw_recs_df, k_novel):
    #     """Filters watched items from the raw 50 recs to return the top 'k_novel' items."""
    #     print(f"Filtering raw recommendations down to {k_novel} novel items per user...")
    #     watched_items_df = self.data_prep.indexed_train.groupBy("userIndex").agg(
    #         F.collect_set("itemIndex").alias("watched_items")
    #     )
        
    #     novel_recs_df = raw_recs_df.join(watched_items_df, on="userIndex", how="left") \
    #         .withColumn("watched_items", F.coalesce(F.col("watched_items"), F.array().cast("array<int>"))) \
    #         .withColumn(
    #             "novel_recs_structs", 
    #             F.expr("filter(recommendations, x -> not array_contains(watched_items, x.itemIndex))")
    #         ) \
    #         .withColumn("top_k_structs", F.expr(f"slice(novel_recs_structs, 1, {k_novel})")) \
    #         .select("userIndex", F.col("top_k_structs.itemIndex").alias("predicted_items"))
        
    #     return novel_recs_df

    def _generate_novel_recommendations(self, raw_recs_df, k_novel, apply_popularity_penalty=True):
        """
        Filters watched items and applies a post-processing popularity penalty 
        to surface niche items while retaining semantic (language/genre) relevance.
        """
        print(f"Filtering raw recommendations down to {k_novel} novel items per user...")
        
        # 1. Get watched items to filter out
        watched_items_df = self.data_prep.indexed_train.groupBy("userIndex").agg(
            F.collect_set("itemIndex").alias("watched_items")
        )

        # 2. Explode the raw recommendations array to access individual ratings
        exploded_recs = raw_recs_df.select(
            "userIndex", 
            F.explode("recommendations").alias("rec")
        ).select(
            "userIndex", 
            F.col("rec.itemIndex").alias("itemIndex"),
            F.col("rec.rating").alias("als_rating")
        )

        # 3. Filter out items the user has already watched
        novel_exploded = exploded_recs.join(watched_items_df, "userIndex", "left") \
            .filter(~F.expr("array_contains(coalesce(watched_items, array()), itemIndex)"))

        # 4. Apply Post-Processing Popularity Penalty
        if apply_popularity_penalty and self.data_prep.item_popularity is not None:
            print("Applying post-processing popularity penalty to ALS scores...")
            
            # Map item_id from popularity table to itemIndex
            popularity_idx = self.data_prep.item_popularity.join(
                self.data_prep.item_lookup, on="item_id", how="inner"
            ).select("itemIndex", "user_count")

            # Join with our novel predictions
            penalized_recs = novel_exploded.join(popularity_idx, on="itemIndex", how="left") \
                .fillna({"user_count": 1.0}) # Fallback for items with no count

            # PENALTY FORMULA: Decrease the score for highly popular items. 
            # You can tune the + 10.0 or the log base to make the penalty harsher/softer.
            penalized_recs = penalized_recs.withColumn(
                "final_score", 
                F.col("als_rating") / F.log10(F.col("user_count") + 10.0)
            )
            sort_column = "final_score"
        else:
            penalized_recs = novel_exploded.withColumn("final_score", F.col("als_rating"))
            sort_column = "final_score"

        # 5. Re-rank based on the new penalized score and take top K
        window_spec = Window.partitionBy("userIndex").orderBy(F.col(sort_column).desc())
        
        top_k_exploded = penalized_recs.withColumn("rank", F.row_number().over(window_spec)) \
            .filter(F.col("rank") <= k_novel)

        # 6. Re-aggregate back into the array format expected by the rest of the pipeline
        novel_recs_df = top_k_exploded.groupBy("userIndex").agg(
            F.collect_list("itemIndex").alias("predicted_items")
        )
        
        return novel_recs_df

    def train_ALS(self, als_params, print_metrics=True):
        train_data = self.data_prep.indexed_train
        k_novel = als_params.get('k', 15)
        
        print("Training ALS Model...")
        als = ALS(
            userCol="userIndex", 
            itemCol="itemIndex", 
            ratingCol="confidence", 
            implicitPrefs=als_params.get("implicitPrefs", True), 
            maxIter=als_params["maxIter"], 
            coldStartStrategy=als_params["coldStartStrategy"],
            rank=als_params["rank"], 
            alpha=als_params["alpha"],
            regParam=als_params["regParam"], 
            
        )

        self.als_model = als.fit(train_data)

        # 1. Generate 50 raw recs for Coverage
        print("Generating 50 raw recommendations for all users...")
        self.als_raw_50 = self.als_model.recommendForAllUsers(50).cache()

        # 2. Generate 15 novel recs for Recall/Precision
        self.als_novel_15 = self._generate_novel_recommendations(self.als_raw_50, k_novel).cache()

        metrics = None
        if print_metrics:
            metrics = self.evaluate_model(k_novel=k_novel)

        return metrics

    def save_model_checkpoint(self, checkpoint_folder="als_checkpoint"):
        """
        Saves the Model, the 50 raw recommendations, and the 15 novel recommendations.
        (Lookups should be saved/handled in DataPreparation class).
        """
        base_path = f"{self.config['temp_path']}/{checkpoint_folder}"
        print(f"Saving ALL ALS artifacts to {base_path}...")
        
        if self.als_model:
            self.als_model.write().overwrite().save(f"{base_path}/als_model")
        if self.als_raw_50:
            self.als_raw_50.write.mode("overwrite").parquet(f"{base_path}/als_raw_50")
        if self.als_novel_15:
            self.als_novel_15.write.mode("overwrite").parquet(f"{base_path}/als_novel_15")
            
        print("Model checkpoint successfully saved!")

    def load_model_checkpoint(self, checkpoint_folder="als_checkpoint"):
        """
        Loads the pre-computed models and recommendations directly from storage.
        """
        base_path = f"{self.config['temp_path']}/{checkpoint_folder}"
        print(f"Loading ALS artifacts from: {base_path}...")
        
        try:
            self.als_model = ALSModel.load(f"{base_path}/als_model")
            self.als_raw_50 = self.spark.read.parquet(f"{base_path}/als_raw_50").cache()
            self.als_novel_15 = self.spark.read.parquet(f"{base_path}/als_novel_15").cache()
            print("Model checkpoint loaded successfully! Ready for inference.")
        except Exception as e:
            print(f"Critical Error loading model checkpoint: {e}")
            raise RuntimeError(f"Failed to load from {base_path}. Ensure you trained and saved the model first.")

    def evaluate_model(self, k_novel=15):
        """Evaluates Coverage using the raw 50 recs, and Recall using the novel 15 recs."""
        if not self.als_raw_50 or not self.als_novel_15:
            raise ValueError("Predictions missing. Please train or load the model first.")

        print("\n" + "="*50)
        print("EVALUATING MODEL METRICS")
        print("="*50)

        # ---------------------------------------------------------
        # METRIC 1: COVERAGE (@50 raw recommendations)
        # ---------------------------------------------------------
        # Explode array of structs -> get the itemIndex
        exploded_recs = self.als_raw_50.select(
            "userIndex", 
            F.explode("recommendations").alias("rec")
        ).select("userIndex", F.col("rec.itemIndex").alias("itemIndex"))

        item_counts = exploded_recs.groupBy("itemIndex").agg(F.count("userIndex").alias("times_recommended"))

        final_counts_df = self.data_prep.item_lookup.join(item_counts, on="itemIndex", how="left") \
            .fillna({"times_recommended": 0})

        total_items = final_counts_df.count()
        covered_items = final_counts_df.filter(F.col("times_recommended") > 0).count()
        
        coverage = covered_items / total_items if total_items > 0 else 0.0

        # ---------------------------------------------------------
        # NEW: RECOMMENDATION FREQUENCY SCORE & PERCENTILES
        # ---------------------------------------------------------
        # 1. Get the max and min times an item was recommended
        stats = item_counts.select(
            F.max("times_recommended").alias("max_c"), 
            F.min("times_recommended").alias("min_c")
        ).collect()[0]

        max_c = stats["max_c"] or 1
        min_c = stats["min_c"] or 0

        # 2. Scale the recommendation counts to a 1-100 score
        if max_c > min_c:
            item_counts = item_counts.withColumn(
                "rec_frequency_score", 
                ((F.col("times_recommended") - min_c) / (max_c - min_c)) * 99.0 + 1.0
            )
        else:
            item_counts = item_counts.withColumn("rec_frequency_score", F.lit(1.0))

        # 3. Calculate Percentiles (using 0.001 relative error for speed/accuracy balance)
        quantiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        percentiles = item_counts.stat.approxQuantile("rec_frequency_score", quantiles, 0.001)

        # 4. Print the distribution
        print("\n--- Recommendation Frequency Percentiles ---")
        for q, p in zip(quantiles, percentiles):
            print(f"{int(q * 100)}th Percentile: {p:.1f} rec_frequency_score(1-100)")

        # ---------------------------------------------------------
        # SANITY STEP: Filter Test users to ensure they exist in Train
        # ---------------------------------------------------------
        ground_truth = self.data_prep.ground_data
        train_users = self.data_prep.indexed_train.select("userIndex").distinct()
        
        # Inner join ensures we only evaluate users who were in both sets
        valid_ground_truth = ground_truth.join(train_users, on="userIndex", how="inner")

        # ---------------------------------------------------------
        # METRIC 2: RECALL / PRECISION (@15 novel recommendations)
        # ---------------------------------------------------------
        joined_data = self.als_novel_15.join(valid_ground_truth, "userIndex").dropna()
        
        eval_df = joined_data \
            .withColumn("hits", F.size(F.array_intersect(F.col("predicted_items"), F.col("actual_items")))) \
            .withColumn("actual_count", F.size(F.col("actual_items"))) \
            .withColumn("recall", F.col("hits") / F.col("actual_count"))
            
        avg_recall = eval_df.select(F.mean("recall")).collect()[0][0] or 0.0
        
        rdd_data = joined_data.select("predicted_items", "actual_items").rdd.map(tuple)
        metrics = RankingMetrics(rdd_data)

        Model_Metrics = {
            "coverage": coverage,
            "avg_recall": avg_recall,
            "precision_k": metrics.precisionAt(k_novel),
            "map": metrics.meanAveragePrecision
        }
        print(f"Catalog Coverage (@50): {Model_Metrics['coverage']:.4f}  ({covered_items} / {total_items} items recommended)")

        print(f"\n[Ranking Performance]")
        print(f"Users Evaluated (Train/Test intersection): {joined_data.count()}")
        print(f"Precision@{k_novel}: {Model_Metrics['precision_k']:.4f}")
        print(f"Recall@{k_novel}:    {Model_Metrics['avg_recall']:.4f}")
        print(f"MAP:             {Model_Metrics['map']:.4f}")
        print("="*50 + "\n")

        return Model_Metrics
    
    def print_omega_model_diagnostics(self, k_novel=15):
        """
        The ultimate SOTA diagnostic for model output behavior.
        Measures Popularity Amplification, Personalization Diversity, and Catalog Resurrection.
        """
        if self.als_novel_15 is None:
            print("Error: ALS novel predictions missing. Train or load model first.")
            return

        print("\n" + "="*80)
        print("OMEGA-LEVEL MODEL DIAGNOSTICS: BEHAVIOR & OUTPUT SKEW")
        print("="*80)

        # 1. Explode recommendations to analyze them at the item level
        exploded_recs = self.als_novel_15.select(
            "userIndex", 
            F.explode("predicted_items").alias("itemIndex")
        ).cache()

        total_recs_given = exploded_recs.count()
        users_evaluated = self.als_novel_15.count()

        # 2. Count how many times each item was recommended
        rec_item_counts = exploded_recs.groupBy("itemIndex").agg(F.count("*").alias("times_recommended")).cache()
        unique_items_recommended = rec_item_counts.count()
        total_catalog_size = self.data_prep.item_lookup.count()
        catalog_coverage_pct = (unique_items_recommended / total_catalog_size) * 100 if total_catalog_size > 0 else 0

        print(f"\n[1. CATALOG COVERAGE & UTILIZATION]")
        print(f"  Users Evaluated:         {users_evaluated:,}")
        print(f"  Total Recs Generated:    {total_recs_given:,}")
        print(f"  Unique Items Suggested:  {unique_items_recommended:,} out of {total_catalog_size:,} ({catalog_coverage_pct:.2f}% Coverage)")

        # 3. RECOMMENDATION SKEW (Is the model just pushing the same 5 items?)
        rec_q = rec_item_counts.stat.approxQuantile("times_recommended", [0.5, 0.90, 0.95, 0.99, 1.0], 0.01)
        
        top_1_threshold = rec_q[3]
        top_1_recs = rec_item_counts.filter(F.col("times_recommended") >= top_1_threshold).agg(F.sum("times_recommended")).collect()[0][0] or 0
        top_1_recs_pct = (top_1_recs / total_recs_given) * 100 if total_recs_given > 0 else 0

        print(f"\n[2. RECOMMENDATION SKEW (Diversity of Output)]")
        print(f"  Median Item Rec'd:       {rec_q[0]:.0f} times")
        print(f"  95th Percentile:         {rec_q[2]:.0f} times")
        print(f"  99th Percentile:         {rec_q[3]:.0f} times")
        print(f"  Max (The Model's Fav):   {rec_q[4]:.0f} times")
        print(f"  -> WARNING ALARM: The top 1% of items consume {top_1_recs_pct:.1f}% of all recommendation slots.")
        if top_1_recs_pct > 30.0:
            print(f"     (Status: CRITICAL. Model has collapsed into a popularity algorithm.)")

        # 4. POPULARITY AMPLIFICATION (Did the model make the 89.7% skew worse?)
        # FIX: Map item_id to itemIndex using the lookup table first!
        original_pop = self.data_prep.item_popularity \
            .join(self.data_prep.item_lookup, on="item_id", how="inner") \
            .select("itemIndex", "popularity_score")
            
        recs_with_pop = rec_item_counts.join(original_pop, on="itemIndex", how="left")
        
        # Calculate weighted popularity (fallback to 1.0 if score is null)
        avg_pop_score = recs_with_pop.withColumn(
            "weighted_pop", 
            F.col("times_recommended") * F.coalesce(F.col("popularity_score"), F.lit(1.0))
        ).agg(F.sum("weighted_pop")).collect()[0][0] / total_recs_given

        print(f"\n[3. POPULARITY BIAS AMPLIFICATION]")
        print(f"  Avg Popularity Score (1-100) of Recommended Items: {avg_pop_score:.2f}")
        if avg_pop_score > 75.0:
            print("  (Model is actively amplifying popularity bias. The rich are getting richer.)")
        else:
            print("  (Model is successfully surfacing middle-tier content.)")

        # 5. CATALOG RESURRECTION (Did the model save the "Bottom 50%"?)
        if self.data_prep.indexed_train:
            train_item_counts = self.data_prep.indexed_train.groupBy("itemIndex").count()
            train_median = train_item_counts.stat.approxQuantile("count", [0.5], 0.01)[0]
            
            bottom_50_items_df = train_item_counts.filter(F.col("count") <= train_median).select("itemIndex")
            bottom_50_recs = rec_item_counts.join(bottom_50_items_df, on="itemIndex", how="inner").count()
            bottom_50_total = bottom_50_items_df.count()
            
            resurrection_rate = (bottom_50_recs / bottom_50_total) * 100 if bottom_50_total > 0 else 0
            
            print(f"\n[4. CATALOG RESURRECTION (The Niche Test)]")
            print(f"  Out of the bottom {bottom_50_total:,} niche items in training...")
            print(f"  The model recommended {bottom_50_recs:,} of them at least once ({resurrection_rate:.1f}% Resurrection Rate).")

        print("\n" + "="*80 + "\n")
        
        exploded_recs.unpersist()
        rec_item_counts.unpersist()

    # ==========================================
    # INFERENCE METHODS 
    # ==========================================
    def _get_titles_for_indices(self, item_indices):
        if self.data_prep.item_lookup is None or self.data_prep.metadata_df is None:
            raise ValueError("Item lookup or metadata table not found in data_prep.")
        
        indices_df = self.spark.createDataFrame([(int(idx),) for idx in item_indices], ["itemIndex"])
        
        titles_df = indices_df.join(self.data_prep.item_lookup, on="itemIndex", how="left") \
            .join(self.data_prep.metadata_df, on="item_id", how="left") \
            .select("title", "item_id", "Genres", "original_language") # Added original_language
            
        return titles_df

    def get_recommendations_userId(self, userId, show_popularity=True):
        if self.als_novel_15 is None:
            raise ValueError("ALS novel predictions not available. Train or load checkpoint first.")
        
        if self.data_prep.combined_train_df is not None and self.data_prep.metadata_df is not None:
            hist = self.data_prep.combined_train_df.filter(F.col("userId") == userId) \
                .join(self.data_prep.metadata_df, on="item_id", how="left") \
                .select("title", "item_id", "Genres", "original_language") # Added original_language
                
            print(f"\nWatch/Click History for userId {userId}:")
            if hist.count() > 0:
                print("Watch History Count: ", hist.count())
                hist.show(truncate=False)
            else:
                print("  - [No title metadata found for history]")
        else:
            print("\n[Warning: combined_train_df or metadata_df missing. Cannot fetch history.]")
        
        user_index_row = self.data_prep.user_lookup.filter(F.col("userId") == userId).select("userIndex").collect()
        
        if not user_index_row:
            print(f"User ID {userId} not found in user lookup.")
            return []
            
        target_index = user_index_row[0]["userIndex"]
        recs = self.als_novel_15.filter(F.col("userIndex") == target_index).select("predicted_items").collect()
        item_popularity = self.data_prep.item_popularity.select("item_id", "popularity_score")

        if recs:
            reco_df = self._get_titles_for_indices(recs[0]["predicted_items"])
            print(f"\nTop Novel Recommendations for userId {userId}:")
            
            # Added original_language to the select statement
            reco_df = reco_df.join(item_popularity, on="item_id", how="left") \
                .select("title", "Genres", "original_language", "popularity_score", "item_id") \
                .orderBy(F.col("popularity_score").desc_nulls_last() if show_popularity else F.lit(0).desc())
            reco_df.show(truncate=False) 
            return reco_df
        else:
            print(f"No recommendations generated by ALS for user index: {target_index}")
            return []

    def get_recommendations_random_user(self, min_content_threshold=1):
        if self.data_prep is None or self.data_prep.combined_train_df is None:
            raise ValueError("Data preparation object or combined_train_df is missing.")

        train_data = self.data_prep.combined_train_df

        valid_users_df = train_data.groupBy("userId") \
            .agg(F.count("item_id").alias("distinct_content_count")) \
            .filter(F.col("distinct_content_count") >= min_content_threshold)

        random_row = valid_users_df.orderBy(F.rand()).first()

        if random_row is None:
            print(f"No users found matching the threshold of {min_content_threshold}.")
            return []

        random_userId = random_row["userId"]
        
        print(f"\n" + "="*50)
        print(f"Randomly Selected User: {random_userId}")
        print(f"User's Historical Item Count: {random_row['distinct_content_count']}")
        print("="*50)

        return self.get_recommendations_userId(random_userId)