# coding=utf-8
# coding=utf-8

import numpy as np
from collections import deque
from recsim import user
from gym import spaces
from absl import logging
import gin.tf


class LTSUserState(user.AbstractUserState):
    """Class to represent users.
    See the LTSUserModel class documentation for precise information about how the
    parameters influence user dynamics.
    Attributes:
      memory_discount: rate of forgetting of latent state.
      sensitivity: magnitude of the dependence between latent state and
        engagement.
      innovation_stddev: noise standard deviation in latent state transitions.
      stddev: standard deviation of engagement with clickbaity content.
      previous_providers: Deque of providers selected previously
      previous_window: Number of previous providers to consider
      net_positive_exposure: starting value for NPE (NPE_0).
      preference: starting prefernce (PR_0)
      step_size: step size for updating preference
      time_budget: length of a user session.
    """

    def __init__(self, memory_discount, sensitivity, innovation_stddev, step_size,
                 stddev, previous_providers, previous_window, net_positive_exposure, preference, time_budget
                 ):
        """Initializes a new user."""
        # Transition model parameters
        ##############################
        self.memory_discount = memory_discount
        self.sensitivity = sensitivity
        self.innovation_stddev = innovation_stddev
        self.step_size = step_size
        self.previous_providers = deque(maxlen=previous_window)
        # Engagement parameters
        self.stddev = stddev

        # State variables
        ##############################
        self.net_positive_exposure = net_positive_exposure
        self.satisfaction = 1 / \
            (1 + np.exp(-sensitivity * net_positive_exposure))

        self.preference = preference/np.linalg.norm(preference)

        self.time_budget = time_budget

    def create_observation(self):
        """The observation generally includes (noisy) information about the 
            user's reaction to the content and potentially clues about the 
            user's latent stateUser's state is not observable."""
        return np.array([self.preference])

    @staticmethod
    def observation_space():
        return spaces.Box(shape=(20,), dtype=np.float32, low=0.0, high=np.inf)


@gin.configurable
class LTSStaticUserSampler(user.AbstractUserSampler):
    """Generates user with identical predetermined parameters.
       Users are selected from set of available users from dataset
    """
    _state_parameters = None

    def __init__(self,
                 users,
                 user_ctor=LTSUserState,
                 memory_discount=0.7,
                 step_size=0.01,
                 sensitivity=0.01,
                 innovation_stddev=0.05,
                 stddev=1.0,
                 previous_window=10,
                 time_budget=60,
                 **kwargs):
        """Creates a new user state sampler."""
        self.users = users
        self.num_users = users.shape[0]
        logging.debug('Initialized LTSStaticUserSampler')
        self._state_parameters = {'memory_discount': memory_discount,
                                  'sensitivity': sensitivity,
                                  'innovation_stddev': innovation_stddev,
                                  'step_size': step_size,
                                  'stddev': stddev,
                                  'previous_window': previous_window,
                                  'time_budget': time_budget
                                  }
        super(LTSStaticUserSampler, self).__init__(user_ctor, **kwargs)

    def sample_user(self):
        starting_pr = self.users[self._rng.choice(self.num_users)]
        # starting_npe = ((self._rng.random_sample() - .5) *
        #                 (1 / (1.0 - self._state_parameters['memory_discount'])))
        starting_npe = 0.
        self._state_parameters['net_positive_exposure'] = starting_npe
        self._state_parameters['preference'] = starting_pr
        self._state_parameters['previous_providers'] = deque(
            maxlen=self._state_parameters['previous_window'])
        return self._user_ctor(**self._state_parameters)


class LTSResponse(user.AbstractResponse):
    """Class to represent a user's response to a provider.
    Attributes:
      engagement: real number representing the degree of engagement with a
        provider (e.g. watch time).
      clicked: boolean indicating whether the item was clicked or not.
    """

    # The maximum degree of engagement.
    MAX_ENGAGEMENT_MAGNITUDE = 5.0

    def __init__(self, clicked=False, engagement=0.0):
        """Creates a new user response for a provider.
        Args:
          clicked: boolean indicating whether the item was clicked or not.
          engagement: real number representing the degree of engagement with a
            provider (e.g. watch time).
        """
        self.clicked = clicked
        self.engagement = engagement

    def __str__(self):
        return '[' + str(self.engagement) + ']'

    def __repr__(self):
        return self.__str__()

    def create_observation(self):
        return {
            'click':
                int(self.clicked),
            'engagement':
                np.clip(self.engagement, 0,
                        LTSResponse.MAX_ENGAGEMENT_MAGNITUDE)
        }

    @classmethod
    def response_space(cls):
        # `engagement` feature range is [0, MAX_ENGAGEMENT_MAGNITUDE]
        return spaces.Dict({
            'click':
                spaces.Discrete(2),
            'engagement':
                spaces.Box(
                    low=0.0,
                    high=LTSResponse.MAX_ENGAGEMENT_MAGNITUDE,
                    shape=tuple(),
                    dtype=np.float32)
        })


class LTSUserModel(user.AbstractUserModel):
    """Class to model a user with long-term satisfaction dynamics.
    Implements a controlled continuous Hidden Markov Model of the user having
    the following components.
      * State space: M dimensional real number, termed preference
        (abbreviated PR);
      * transition dynamics: net_positive_exposure is updated according to:
        clickbait_score = np.dot(provider.feature, self._user_state.preference)
        NPE_(t+1) := memory_discount * NPE_t
                     + 2 * (clickbait_score - .5)
                     + N(0, innovation_stddev);
        User preference is updated according to:
        PR_(t+1) :=  norm(PR_t + step_size * (clickbait_score* provider.feature) * recency bias)
      * observation space: a nonnegative real number, representing the degree of
        engagement, eg. econds watched from recommended video. An observation
        is drawn from a log-normal distribution with mean
        (provider.feature^T PR_T)
        An individual user is thus represented by the combination of parameters
        (step_size, innovation_stddev, NPE, PR, sensitivity, prev_clicked_count, stddev), which are encapsulated in LTSUserState.
      Args:
        slate_size: An integer representing the size of the slate
        user_state_ctor: A constructor to create user state.
        response_model_ctor: A constructor function to create response. The
          function should take a string of doc ID as input and returns a
          LTSResponse object.
        seed: an integer as the seed in random sampling.
    """

    def __init__(self,
                 users,
                 slate_size,
                 user_state_ctor=None,
                 response_model_ctor=None,
                 seed=0):
        if not response_model_ctor:
            raise TypeError('response_model_ctor is a required callable.')

        super(LTSUserModel, self).__init__(
            response_model_ctor,
            LTSStaticUserSampler(users, user_ctor=user_state_ctor, seed=seed), slate_size)

    def is_terminal(self):
        """Returns a boolean indicating if the session is over."""
        return self._user_state.time_budget <= 0

    def recency_bias(self, provider_feature):
        self._user_state.previous_providers.append(provider_feature)
        previous_aggregate = np.zeros_like(provider_feature)
        for idx, feature in enumerate(self._user_state.previous_providers):
            previous_aggregate += feature / \
                (len(self._user_state.previous_providers)-idx)
        previous_aggregate = previous_aggregate / \
            np.linalg.norm(previous_aggregate)
        return 1 - np.dot(previous_aggregate, provider_feature)

    def update_state(self, slate_providers, responses):
        """Updates the user's latent state based on responses to the slate.
        Args:
          slate_providers: a list of Providers representing the slate
          responses: a list of LTSResponses representing the user's response to each
            provider in the slate.
        """
        for provider, response in zip(slate_providers, responses):
            if response.clicked:
                innovation = np.random.normal(
                    scale=self._user_state.innovation_stddev)
                clickbait_score = np.dot(
                    provider.feature, self._user_state.preference)
                # net_positive_exposure = (self._user_state.memory_discount
                #                          * self._user_state.net_positive_exposure
                #                          - 2.0 * (clickbait_score - 0.5)
                #                          + innovation
                #                          )
                satisfaction = 10*clickbait_score * self.recency_bias(provider.feature)
                self._user_state.net_positive_exposure += satisfaction

                # self._user_state.net_positive_exposure = net_positive_exposure

                preference = self._user_state.preference + self._user_state.step_size * \
                    clickbait_score * provider.feature * \
                    self.recency_bias(provider.feature)

                self._user_state.preference = preference / \
                    np.linalg.norm(preference)
                # satisfaction = 1 / (1.0 + np.exp(-self._user_state.sensitivity
                #                                  * net_positive_exposure)
                #                     )
                self._user_state.satisfaction = satisfaction
                self._user_state.time_budget -= 1
                return

    def simulate_response(self, providers):
        """Simulates the user's response to a slate of providers with choice model.
        Args:
          providers: a list of Provider objects.
        Returns:
          responses: a list of LTSResponse objects, one for each document.
        """
        # List of empty responses
        responses = [self._response_model_ctor() for _ in providers]
        # User always clicks the first item.
        selected_index = 0
        self.generate_response(
            providers[selected_index], responses[selected_index])
        return responses

    def generate_response(self, provider, response):
        """Generates a response to a clicked provider.
        Args:
          provider: an Provider object.
          response: an LTSResponse for the provider.
        Updates: response, with whether the provider was clicked, liked, and how
          much of it was watched.
        """
        response.clicked = True
        # engagement_loc = np.dot(provider.feature, self._user_state.preference)
        engagement_loc = self._user_state.satisfaction
        engagement_scale = self._user_state.stddev
        log_engagement = np.random.normal(loc=engagement_loc,
                                          scale=engagement_scale)
        response.engagement = np.exp(log_engagement)

# # Verify UserSampler
# import matplotlib.pyplot as plt
# sampler = LTSStaticUserSampler(users=U)
# starting_npe = []
# for i in range(1000):
#   sampled_user = sampler.sample_user()
#   starting_npe.append(sampled_user.net_positive_exposure)
# _ = plt.hist(starting_npe)
