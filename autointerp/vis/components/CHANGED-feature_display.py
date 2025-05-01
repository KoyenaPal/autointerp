from typing import List, Dict

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from ...base import Example, Feature
from ..backend import InferenceResult
from .base import Component


TOOLTIP = """
<style>
.tooltip {
    position: relative;
    display: inline-block;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 120px;
    background-color: black;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px 0;

    /* Position the tooltip */
    position: absolute;
    z-index: 1000;
    top: 100%;
    left: 120%;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
}

.activating-example {
    margin: 3px 0;
    padding: 3px;
    background-color: #f5f5f5;
}
</style>
"""

ACTIVATING_EXAMPLE_WRAPPER = """
<div class="activating-example">
    {example}
</div>
"""

HIGHLIGHTED_TOKEN_WRAPPER = """
<span class="tooltip">
    <span style="background-color: rgba({color}, {opacity:.2f})">{token}</span>
    <span class="tooltiptext">{activation:.2f}</span>
</span>
"""


class FeatureDisplay(Component):
    def __init__(self):
        self.feature_display = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                height="100%",
                border="1px solid #ddd",
                padding="10px",
            )
        )

        super().__init__(self.feature_display)

    def _display_inference_example(self, query_result: InferenceResult):
        inference_html = self._example_to_html(query_result.inference_example)
        display(HTML(ACTIVATING_EXAMPLE_WRAPPER.format(example=inference_html)))

        display(HTML("<hr>"))

        return query_result.feature

    def display(self, features_dict: Dict[str, List[InferenceResult | Feature]]):
        """Display the top features for the selected tokens."""
        with self.feature_display:
            clear_output()

            display(HTML(TOOLTIP))
            # Iterate through the dictionary items (hookpoint and the list of results)
            for hookpoint, results_list in features_dict.items():

                display(HTML(f"<h2>{hookpoint}</h2>"))

                # Iterate through the list of InferenceResult or Feature objects
                for query_result in results_list:
                    # --- DEBUGGING START ---
                    print(f"DEBUG: Processing query_result of type: {type(query_result)}")
                    if isinstance(query_result, str):
                        print(f"DEBUG: String value: {query_result}")
                    # --- DEBUGGING END ---

                    # Original problematic code:
                    try: # Add try-except to potentially get more info if it fails
                        index = (
                            query_result.index
                            if isinstance(query_result, Feature)
                            else query_result.feature.index
                        )
                        display(HTML(f"<h4>Feature {index}</h4>"))
                    except AttributeError as e:
                        print(f"ERROR accessing index for query_result: {query_result}")
                        print(f"Error details: {e}")
                        # Optionally re-raise or handle
                        raise e # Re-raise to see the original traceback context

                    if isinstance(query_result, InferenceResult):
                        # This returns the underlying Feature object after displaying inference info
                        feature_obj = self._display_inference_example(query_result)
                    elif isinstance(query_result, Feature):
                         # If it's already a Feature object
                        feature_obj = query_result
                    else:
                        # Handle the unexpected string case gracefully if needed,
                        # otherwise the error above will stop execution.
                        print(f"WARN: Skipping unexpected item of type {type(query_result)}")
                        continue # Skip this item

                    # Use the potentially updated feature_obj
                    if feature_obj.activating_examples:
                        # Display activating examples directly
                        for example in feature_obj.activating_examples:
                            example_html = self._example_to_html(example)
                            display(
                                HTML(
                                    ACTIVATING_EXAMPLE_WRAPPER.format(
                                        example=example_html,
                                    )
                                )
                            )

    def _example_to_html(
        self,
        example: Example,
        threshold_ratio: float = 0.0,  # Renamed threshold for clarity
    ) -> str:
        str_tokens = example.str_tokens
        activations = example.activations

        result = []
        # Calculate max absolute activation for normalization, handle potential empty tensor
        if activations.numel() == 0:
            max_abs_act = 0.0
        else:
            max_abs_act = activations.abs().max().item()

        # Calculate absolute threshold based on ratio
        threshold = max_abs_act * threshold_ratio

        # Define base colors (RGB)
        positive_color = "0, 102, 204"  # Blue
        negative_color = "204, 51, 51"  # Red

        for i in range(len(str_tokens)):
            activation_val = activations[i].item()
            abs_activation = abs(activation_val)

            if abs_activation > threshold:
                # Calculate opacity based on activation value (normalized between 0.2 and 1.0)
                # Avoid division by zero if max_abs_act is 0
                opacity = 0.2 + 0.8 * (abs_activation / max_abs_act) if max_abs_act > 0 else 0.2

                if activation_val > 0:
                    color = positive_color
                else:
                    # Use negative color even for exactly zero if threshold is zero and we want to highlight negatives
                    color = negative_color

                result.append(
                    HIGHLIGHTED_TOKEN_WRAPPER.format(
                        token=str_tokens[i],
                        color=color,
                        opacity=opacity,
                        activation=activation_val,
                    )
                )
            else:
                result.append(str_tokens[i])
        return "".join(result)
