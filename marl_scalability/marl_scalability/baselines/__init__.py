# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from smarts.zoo.registry import register
from .sac.sac.policy import SACPolicy
from .ppo.ppo.policy import PPOPolicy
from .dqn.dqn.policy import DQNPolicy
from .dqn_discrete.dqn_discrete.policy import DiscreteDQNPolicy
from .ppo_discrete.ppo_discrete.policy import DiscretePPOPolicy
from .sac_discrete.sac_discrete.policy import DiscreteSACPolicy
from smarts.core.controllers import ActionSpaceType
from marl_scalability.baselines.agent_spec import BaselineAgentSpec

register(
    locator="sac-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Continuous, policy_class=SACPolicy, **kwargs
    ),
)
register(
    locator="ppo-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Continuous, policy_class=PPOPolicy, **kwargs
    ),
)
register(
    locator="dqn-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Continuous, policy_class=DQNPolicy, **kwargs
    ),
)
register(
    locator="dqn_discreteRGB-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Lane, image_agent=True, policy_class=DiscreteDQNPolicy, **kwargs
    ),
)
register(
    locator="dqn_discrete-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Lane, policy_class=DiscreteDQNPolicy, **kwargs
    ),
)
register(
    locator="sac_discrete-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Lane, policy_class=DiscreteSACPolicy, **kwargs
    ),
)
register(
    locator="sac_discreteRGB-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Lane, image_agent=True, policy_class=DiscreteSACPolicy, **kwargs
    ),
)
register(
    locator="ppo_discrete-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Lane, policy_class=DiscretePPOPolicy, **kwargs
    ),
)
register(
    locator="ppo_discreteRGB-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Lane, image_agent=True, policy_class=DiscretePPOPolicy, **kwargs
    ),
)
