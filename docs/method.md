# Method

[Back to README](../README.md)

Entropy measures uncertainty at a state; polarity measures the direction in which
a sampled update changes that uncertainty. With intrinsic entropy tendency
$\mathcal{T}(s_t,y_t)$ and advantage $A$, the effective polarity is

$$
\mathcal{P}(s_t,y_t,A)=A\,\mathcal{T}(s_t,y_t).
$$

Positive polarity indicates entropy-expanding pressure; negative polarity indicates
entropy-contracting pressure. The advantage matters: the same token can belong to
either branch when the training signal changes sign.

PAPO rescales token advantages according to this branch assignment:

$$
\widetilde A_t =
\begin{cases}
A\,\omega_{\mathrm{pos}}, & \mathcal{P}_t>0,\\
A\,\omega_{\mathrm{neg}}, & \mathcal{P}_t<0,\\
A, & \mathcal{P}_t=0.
\end{cases}
$$

After a neutral warmup, an entropy-slope controller adjusts the relative branch
weights. Rapid entropy decay calls for stronger exploration protection. As the
decay slows, the controller restores reward-aligned exploitation. A late-stage
entropy gate returns the weights to one. This preserves both branches while
adapting their influence to the training phase.

The implementation separates [polarity statistics](../src/recipe/papo/dp_actor.py),
[branch selection and advantage reweighting](../src/recipe/papo/control_policy.py),
and the [entropy-slope controller](../src/recipe/papo/adaptive_entropy.py).
The symmetric controller is selected by `adaptive_sym_topk`; its settings are in
`trainer.entropy_mask.adaptive`.
