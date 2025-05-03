import os
import torch as t
import pandas as pd
import numpy as np
from collections import defaultdict
import gc # Import garbage collector
import sys
sys.path.append("..")
import asyncio
from autointerp.automation import OpenRouterClient,LogProbsClient, Explainer
from autointerp import load, make_quantile_sampler
# Assuming load_header_parquet, get_shards, combine_shards are defined above...

# --- load_features_filtered function definition remains the same ---
def load_header_parquet(path):
    # load header.parquet
    df = pd.read_parquet(path)
    return df

def get_shards(feature_id, header_df):
    # get the shards for a given feature_id
    shards = header_df[header_df["feature_idx"] == feature_id]["shard"].values
    return shards

def download_shard(shard_id, shard_dir):
    # full path to the shard
    shard_path = os.path.join(shard_dir, f"{shard_id}.pt")
    # load the shard
    shard = t.load(shard_path)
    return shard

def combine_shards(shards, key):
    """Combines tensors from a list of shard dictionaries based on a key."""
    tensors_to_cat = [shard[key] for shard in shards if key in shard and isinstance(shard[key], t.Tensor)]
    if not tensors_to_cat:
        # Handle case where key is not found or no tensors are present
        print(f"Warning: No tensors found for key '{key}' in shards.")
        return None # Or return an empty tensor: t.tensor([])
    # Ensure tensors are on the same device before concatenation
    if len(set(t.device for t in tensors_to_cat)) > 1:
        # Example: Move all to the device of the first tensor
        target_device = tensors_to_cat[0].device
        print(f"Warning: Tensors for key '{key}' are on multiple devices. Moving to {target_device}.")
        tensors_to_cat = [tensor.to(target_device) for tensor in tensors_to_cat]
    elif tensors_to_cat: # Check if list is not empty
         target_device = tensors_to_cat[0].device # Keep original device if all same
    else:
        return None # Should not happen if check above passed, but for safety

    try:
        return t.cat(tensors_to_cat)
    except Exception as e:
        print(f"Error during torch.cat for key '{key}': {e}")
        # Optionally, print shapes for debugging:
        # print(f"Shapes: {[t.shape for t in tensors_to_cat]}")
        return None


def load_features_filtered(feature_ids, cache_dir):
    """
    Loads data for a list of features efficiently by loading each required shard
    once, filtering its contents immediately, and discarding the full shard data.
    Assumes 'locations' tensor has feature_id in the 3rd column (index 2).
    """
    header_path = os.path.join(cache_dir, "header.parquet")
    if not os.path.exists(header_path):
        raise FileNotFoundError(f"Header file not found: {header_path}")
    header_df = load_header_parquet(header_path) # Assuming this function exists

    # Ensure feature_ids is a set for efficient lookup
    requested_feature_ids_set = set(feature_ids)
    if not requested_feature_ids_set:
        return {}

    all_shard_ids_needed = set()
    feature_to_shards = defaultdict(list)

    # 1. Identify all unique shards needed for the requested features
    for feature_id in requested_feature_ids_set:
        # Make sure get_shards returns a list or array-like structure
        shard_ids_for_feature = get_shards(feature_id, header_df)
        if shard_ids_for_feature is None or len(shard_ids_for_feature) == 0:
            print(f"Warning: No shards found for feature_id {feature_id}")
            continue
        # Ensure shard_ids are usable as dict keys (e.g., strings or ints)
        valid_shard_ids = [sid for sid in shard_ids_for_feature if sid is not None]
        feature_to_shards[feature_id].extend(valid_shard_ids)
        all_shard_ids_needed.update(valid_shard_ids)


    if not all_shard_ids_needed:
        print("Warning: No shards identified for any of the requested features.")
        return {}

    # Dictionary to hold lists of filtered tensors for each feature
    feature_data_parts = defaultdict(lambda: {
        'locations': [],
        'activations': [],
        'tokens_path': None,
        'model_id': None
    })

    # 2. Load each unique shard, filter immediately, and store parts
    print(f"Need to load {len(all_shard_ids_needed)} unique shards.")
    processed_shards_count = 0
    for shard_id in all_shard_ids_needed:
        shard_path = os.path.join(cache_dir, f"{shard_id}.pt")
        if not os.path.exists(shard_path):
            print(f"Warning: Shard file not found: {shard_path}. Skipping.")
            continue

        try:
            # Load the entire shard dictionary
            # print(f"Loading shard: {shard_id} from {shard_path}")
            loaded_shard = t.load(shard_path, map_location='cpu') # Load to CPU first to potentially save GPU memory

            if not (isinstance(loaded_shard, dict) and
                    'locations' in loaded_shard and 'activations' in loaded_shard and
                    isinstance(loaded_shard['locations'], t.Tensor) and
                    isinstance(loaded_shard['activations'], t.Tensor)):
                print(f"Warning: Shard {shard_id} has unexpected format or missing/invalid tensors. Skipping.")
                del loaded_shard # Clean up
                continue

            shard_locations = loaded_shard['locations']
            shard_activations = loaded_shard['activations']

            # Check tensor dimensions (basic sanity check)
            if shard_locations.ndim < 2 or shard_activations.ndim < 1 or shard_locations.shape[0] != shard_activations.shape[0]:
                 print(f"Warning: Tensor dimension mismatch or unexpected shape in shard {shard_id}. Loc: {shard_locations.shape}, Act: {shard_activations.shape}. Skipping.")
                 del loaded_shard, shard_locations, shard_activations
                 gc.collect() # Clean up before continuing
                 continue


            # Assuming feature index is the 3rd column (index 2) in locations
            if shard_locations.shape[1] < 3:
                print(f"Warning: Locations tensor in shard {shard_id} has fewer than 3 columns ({shard_locations.shape}). Cannot extract feature IDs. Skipping.")
                del loaded_shard, shard_locations, shard_activations
                gc.collect() # Clean up before continuing
                continue

            # Ensure locations[:, 2] is compatible with feature IDs (e.g., long tensor)
            shard_feature_col = shard_locations[:, 2].long()

            # Find which of the *requested* features are present in *this* shard
            # Use unique to avoid redundant checks if a feature appears multiple times
            features_present_in_shard = t.unique(shard_feature_col)

            # Efficiently find intersection using tensor operations if possible, or sets
            # Convert requested_feature_ids_set to a tensor for isin
            requested_fids_tensor = t.tensor(list(requested_feature_ids_set), dtype=t.long)
            mask_features_present = t.isin(features_present_in_shard, requested_fids_tensor)
            relevant_feature_ids_in_shard = features_present_in_shard[mask_features_present].tolist()


            # Filter and distribute data for relevant features
            if relevant_feature_ids_in_shard:
                # Create a mask for all relevant features at once using the original feature column
                combined_mask = t.isin(shard_feature_col, t.tensor(relevant_feature_ids_in_shard, dtype=t.long))

                # Apply mask to get only relevant rows from the original shard tensors
                relevant_locations = shard_locations[combined_mask]
                relevant_activations = shard_activations[combined_mask]

                # Now distribute these filtered rows to the correct feature_id list
                for fid in relevant_feature_ids_in_shard:
                     # Ensure fid is the correct type (e.g., int) if used as dict key
                     fid_key = int(fid)
                     # Create mask specific to this feature within the *already filtered* tensors
                     feature_mask = (relevant_locations[:, 2].long() == fid) # Use .long() for comparison
                     feature_data_parts[fid_key]['locations'].append(relevant_locations[feature_mask])
                     feature_data_parts[fid_key]['activations'].append(relevant_activations[feature_mask])

                     # Store metadata once per feature
                     if feature_data_parts[fid_key]['tokens_path'] is None:
                         feature_data_parts[fid_key]['tokens_path'] = loaded_shard.get('tokens_path')
                         feature_data_parts[fid_key]['model_id'] = loaded_shard.get('model_id')

                del relevant_locations, relevant_activations, combined_mask # Free filtered tensors

            # Crucially, delete the large loaded objects and run GC
            del loaded_shard, shard_locations, shard_activations, features_present_in_shard, shard_feature_col
            processed_shards_count += 1
            if processed_shards_count % 10 == 0: # Optional: Collect garbage periodically
                 gc.collect()

        except Exception as e:
            print(f"Error processing shard {shard_id} from {shard_path}: {e}")
            # Ensure cleanup even if error occurs mid-processing
            if 'loaded_shard' in locals(): del loaded_shard
            if 'shard_locations' in locals(): del shard_locations
            if 'shard_activations' in locals(): del shard_activations
            gc.collect()
            continue

    # Clear cache if using GPU tensors were involved
    if t.cuda.is_available():
        t.cuda.empty_cache()
    gc.collect() # Final collection

    # 3. Combine the filtered parts for each feature
    features_data = {}
    print("Combining filtered data...")
    for feature_id, parts in feature_data_parts.items():
        if parts['locations'] and parts['activations']: # Check if data was found
            try:
                # Combine tensors; ensure combine_shards handles potential errors/empty lists
                combined_locations = t.cat(parts['locations']) if parts['locations'] else None
                combined_activations = t.cat(parts['activations']) if parts['activations'] else None


                # Skip if combination failed or resulted in empty tensors
                if combined_locations is None or combined_activations is None or combined_locations.shape[0] == 0:
                    print(f"Warning: No valid data combined for feature {feature_id}. Skipping.")
                    continue


                # Basic check after concatenation
                if combined_locations.shape[0] != combined_activations.shape[0]:
                    print(f"Warning: Mismatch after concatenating for feature {feature_id}. Loc: {combined_locations.shape}, Act: {combined_activations.shape}. Skipping feature.")
                    continue

                # Check for essential metadata
                if parts['tokens_path'] is None or parts['model_id'] is None:
                     print(f"Warning: Missing 'tokens_path' or 'model_id' for feature {feature_id}. Skipping feature.")
                     continue


                features_data[feature_id] = {
                    'locations': combined_locations,
                    'activations': combined_activations,
                    'tokens_path': parts['tokens_path'],
                    'model_id': parts['model_id']
                }
                # print(f"Feature {feature_id}: Combined {combined_locations.shape[0]} activations.")


            except Exception as e:
                print(f"Error combining tensors for feature {feature_id}: {e}")
                # Clean up potentially partially combined tensors
                if 'combined_locations' in locals(): del combined_locations
                if 'combined_activations' in locals(): del combined_activations
                gc.collect()
                continue # Skip this feature if combination fails
        else:
            print(f"Warning: No data parts collected or combined successfully for feature {feature_id}.")


    if not features_data:
         print("Warning: No data could be successfully processed for any requested feature.")

    print(f"Finished loading. Returning data for {len(features_data)} features.")
    return features_data


# --- Constants ---
EXPLAINER_MODEL = "openai/gpt-4o-mini"
# FEATURE_PATH = "/disk/u/koyena/llama-8b-cache-sae-lens-with-slimpajama-L7/model.layers.7" # Might not be needed if CACHE_DIR is sufficient
CACHE_DIR = "/disk/u/koyena/llama-8b-cache-sae-lens-with-slimpajama/model.layers.15"

l15_bottom_50_ascending_zero = [9496, 25488, 20728, 15686, 29936, 18691, 17357, 26817, 26760, 21835, 18531, 3527, 4068, 16392, 1616, 10897, 22697, 18900, 30262, 31382, 18995, 15318, 24731, 32146, 6129, 9980, 5182, 20844, 7078, 24080, 15390, 10578, 1446, 663, 26945, 8491, 9880, 3192, 4481, 4190, 16090, 5760, 25060, 6179, 1407, 4635, 25640, 11925, 21162, 25272]

async def explain():
    print(f"Using explainer model: {EXPLAINER_MODEL}")
    print(f"Loading data from cache: {CACHE_DIR}")
    client = OpenRouterClient(EXPLAINER_MODEL)
    explainer = Explainer(client=client)

    print(f"Loading data for {len(l15_bottom_50_ascending_zero)} features...")
    # feature_data_map: Dict[int, Dict[str, Any]]
    feature_data_map = load_features_filtered(l15_bottom_50_ascending_zero, CACHE_DIR)

    if not feature_data_map:
        print("No feature data loaded. Exiting.")
        return

    sampler = make_quantile_sampler(n_examples=10, n_quantiles=1)
    loaded_features_list = []

    print("Processing loaded feature data...")
    for feature_id, single_feature_data in feature_data_map.items():
        # Basic check for valid data structure for 'load'
        if not all(k in single_feature_data for k in ['locations', 'activations', 'tokens_path', 'model_id']) or \
           single_feature_data['locations'] is None or single_feature_data['activations'] is None or \
           single_feature_data['tokens_path'] is None or single_feature_data['model_id'] is None:
            print(f"Skipping feature {feature_id} due to missing essential data components.")
            continue

        # Check for empty tensors
        if single_feature_data['locations'].shape[0] == 0:
            print(f"Skipping feature {feature_id} due to empty tensors.")
            continue

        # Check if tokens file exists before calling load, as load might try to access it
        tokens_file_path = single_feature_data['tokens_path']
        if not os.path.exists(tokens_file_path):
            print(f"Skipping feature {feature_id}: tokens file not found at {tokens_file_path}")
            continue

        try:
            # Assuming load processes the data dict for the features within it
            # Pass the specific data dictionary for this feature.
            # The FeatureIndex returned by load is iterable (often yields one Feature)
            # print(f"Calling autointerp.load for feature {feature_id}...")
            feature_index_obj = load(CACHE_DIR, sampler, data=single_feature_data) # Pass single feature data
            loaded_features_list.extend(list(feature_index_obj)) # Collect Feature objects
            # print(f"Successfully processed feature {feature_id}.")

        except KeyError as e:
             print(f"KeyError while calling autointerp.load for feature {feature_id}: {e}. Check structure of 'single_feature_data'.")
             continue
        except FileNotFoundError as e:
             print(f"FileNotFoundError during autointerp.load for feature {feature_id}: {e}. Check paths within data.")
             continue
        except Exception as e:
             print(f"Unexpected error during autointerp.load for feature {feature_id}: {type(e).__name__} - {e}")
             # Consider adding traceback here for debugging
             # import traceback; traceback.print_exc()
             continue # Skip this feature on error

    if not loaded_features_list:
        print("No features were successfully processed by autointerp.load. Exiting.")
        return

    print(f"\nGenerating explanations for {len(loaded_features_list)} processed features...")
    tasks = [
        explainer(feature)
        for feature in loaded_features_list # Iterate through collected Feature objects
    ]

    try:
        explanations = await asyncio.gather(*tasks)
        print("\n--- Explanations ---")
        # Zip explanations with the corresponding Feature objects
        for explanation, feature in zip(explanations, loaded_features_list):
            print(f"Feature ID: {feature.index}", flush=True)
            print("Explanation:")
            print(explanation)
            print("-" * 100, flush=True)
    except Exception as e:
        print(f"\nError during explanation generation: {type(e).__name__} - {e}")
        # Add more details if needed, e.g., which feature failed if gather allows partial results
        # import traceback; traceback.print_exc()


if __name__ == "__main__":
    print("Starting explanation script...")
    try:
        asyncio.run(explain())
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
        # import traceback; traceback.print_exc()
    print("Explanation script finished.")
