import sys
sys.path.append("..")
import asyncio
from autointerp.automation import OpenRouterClient,LogProbsClient, Explainer
from autointerp import load, make_quantile_sampler

EXPLAINER_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
FEATURE_PATH = "/share/u/koyena/llama-8b-cache-sae-lens/model.layers.15/0.pt"

async def explain():
    client = OpenRouterClient(EXPLAINER_MODEL)
    explainer = Explainer(client=client)

    sampler = make_quantile_sampler(n_examples=2, n_quantiles=1)
    features = load(FEATURE_PATH, sampler)
    tasks = [
        explainer(feature)
        for feature in features[:1]
    ]
    explanations = await asyncio.gather(*tasks)
    print(explanations)

if __name__ == "__main__":
    asyncio.run(explain())

