# coding=utf-8
# coding=utf-8

import numpy as np
from recsim import document
from gym import spaces


class Provider(document.AbstractDocument):
    def __init__(self, provider_id, feature):
        self.feature = feature
        # doc_id is an integer representing the unique ID of the provider
        super(Provider, self).__init__(provider_id)

    def create_observation(self):
        return np.array([self.feature])

    @staticmethod
    def observation_space():
        return spaces.Box(shape=(20,), dtype=np.float32, low=0.0, high=1.0)

    def __str__(self):
        return "Provider {} with feature {}.".format(self._doc_id, self.feature)


class ProviderSampler(document.AbstractDocumentSampler):
    def __init__(self, providers, doc_ctor=Provider,engagement_decay=0.99, engagement_threshold=0.5, **kwargs):
        self._doc_count = 0
        self.providers = providers
        self.providers_engagement = np.ones(providers.shape[0])
        self.available_providers = np.arange(0, providers.shape[0])
        self.engagement_decay = engagement_decay
        self.engagement_threshold = engagement_threshold
        super(ProviderSampler, self).__init__(doc_ctor, **kwargs)

    def sample_document(self):
        doc_features = {}
        while True:
            provider_id = self._rng.choice(self.available_providers)
            if provider_id in self.available_providers:
                break

        doc_features['provider_id'] = provider_id
        doc_features['feature'] = self.providers[provider_id]
        self._doc_count += 1
        return self._doc_ctor(**doc_features)

    def reset_sampler(self):
        print(20*'*')
        print(f"Resetting ProviderSampler")
        super(ProviderSampler, self).reset_sampler()
        self.available_providers = np.arange(0, self.providers.shape[0])
        self.providers_engagement = np.ones(self.providers.shape[0])
        


    def update_state(self, providers, responses):
        """Update document state (if needed) given user's (or users') responses."""
        response_clicked_ids = [i for i, response in enumerate(responses) if response.clicked]
        response_clicked_engagements = [response.engagement for i, response in enumerate(responses) if response.clicked]
        clicked_provider_ids = [provider.doc_id() for provider in np.array(providers)[response_clicked_ids]]
        response_engagement_dict = dict()
        for i, idx in enumerate(clicked_provider_ids):
          response_engagement_dict[idx] = response_clicked_engagements[i]

        for provider_id in self.available_providers:
          self.providers_engagement[provider_id] *=  self.engagement_decay
          if provider_id in clicked_provider_ids:
            self.providers_engagement[provider_id] +=response_engagement_dict[provider_id]

        mask = np.ones(len(self.available_providers), dtype=bool)
        mask[[idx for idx in range(len(mask)) if self.providers_engagement[idx] < self.engagement_threshold]] = False
        self.available_providers = self.available_providers[mask,...]
        return len(self.available_providers) < len(responses) # slate

# ProviderCandidateSet = document.CandidateSet()
# for i in range(V.shape[0]):
#   doc = Provider(i,V[i])
#   ProviderCandidateSet.add_document(doc)

# # # Verify ProviderSampler
# sampler = ProviderSampler(V)
# for i in range(5): print(sampler.sample_document())
# d = sampler.sample_document()
# print("Documents have observation space:", d.observation_space(), "\n"
#       "An example realization is: ", d.create_observation())
