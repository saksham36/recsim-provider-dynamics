# coding=utf-8
# coding=utf-8

from recsim.simulator import recsim_gym
from recsim.simulator import environment

from provider import ProviderSampler
from user import LTSUserModel, LTSUserState, LTSResponse


def clicked_engagement_reward(responses, doc_sampler, alpha=0.5):
    """Calculates the total clicked watchtime from a list of responses.
    Args:
      responses: A list of LTSResponse objects
    Returns:
      reward: A float representing the total watch time from the responses
    """
    print("In clicked_engagement reward")
    reward = 0.0
    for response in responses:
        if response.clicked:
            reward += response.engagement
    reward = alpha* reward +\
     (1-alpha)*sum([doc_sampler.providers_engagement[provider_id] for provider_id in doc_sampler.available_providers])/len(doc_sampler.providers)
    return reward


def create_environment(env_config, users, providers):
    """Creates a long-term satisfaction environment."""

    user_model = LTSUserModel(
        users=users,
        slate_size=env_config['slate_size'],
        user_state_ctor=LTSUserState,
        response_model_ctor=LTSResponse)

    provider_sampler = ProviderSampler(providers)

    ltsenv = environment.Environment(
        user_model,
        provider_sampler,
        env_config['num_candidates'],
        env_config['slate_size'],
        resample_documents=env_config['resample_documents'])

    return recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)
