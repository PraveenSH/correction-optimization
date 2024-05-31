# correction-optimization
Align LLMs by minimizing correction required for the generated response through a teacher

# Introduction

Recent advancements in Large Language Models (LLMs) have focused on aligning these models with human preferences. Approaches like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) achieve this alignment by optimizing for human preferences data. RLHF does this by learning an explicit reward model and uses RL to optimize policy while DPO directly optimizes policy without the need of a reward model.

An alternative method to align LLM responses involves utilizing correction mechanisms. Corrections offer more explicit feedback compared to scalar rewards. Instead of directly optimizing for human preferences, our approach aims to minimize the required amount of correction. This correction is determined by comparing the model's response with a corrected version provided by a teacher, which can be either a human or a more accurate model.

The correction mechanism can serve as a form of regularization, preventing excessive alterations to the model, whereas preference optimization algorithms have the potential to significantly alter the model's behavior based on the training samples in preference data.

This model is output of a POC experiment using above hypothesis.
In this experiment, the base model is initialized from a merged model -  https://huggingface.co/prhegde/merge-aanaphi-phi2-orage-3b. Mistral-7b-instruct is used as a teacher model.
These models are selected with consideration for the compute resource constraints.

## Proposed Approach
The proposed methodology involves the following steps:
1. Generate a response $y$ for a given prompt $x$ from a policy to be optimized, denoted as $\pi(y/x)$.
2. Request the teacher to produce the corrected version $y'$ of $y$ denoted as $\tau(y'/y, x)$. The teacher can be a human or a more precise LLM.
3. Assume a distribution function $f$ that gives the likelihood of given text. For a prompt $x$ and a corresponding response $y$, $f(y, x)$ can be written as $P(x)*P(y/x)$. Log-likelyhood becomes $\log(P(x)) + \log(P(y/x))$.
4. Calculate the amount of correction required as a difference between the above distribution $f(y, x) - f(y', x) = \log(P(y/x)) - \log(P(y'/x))$.
5. The distribution $P$ can be parameterized through the policy $\pi$. For training this model, $P$ is directly set to $\pi$.
7. Optimize the policy $\pi$ to minimize the necessary correction = $min_{\pi} ( \log(\pi(y/x) - \log(\pi(y’/x) )$

## Domain specific custom objective
This framework allows for the selection of \\(P\\), offering the flexibility to choose. If additional prior assumptions are available, they can be integrated. For instance, a prior concerning the distribution of response lengths could be included, limiting the model to produce responses of a certain length. If $P(y)$ = $\pi(y)$ * $l(y)$, where $l(y)$ is a prior specific to a target domain, the optimization function becomes $min_{\pi} ( \log(\pi(y/x)) - \log(\pi(y’/x) ) + \log(l(y)) - \log(l(y’)) $. This indicates the aim to minimize the extra loss specific to the target domain.
   
## Connection with Direct Preference Optimization (DPO) and Contrastive Preference Learning (CPL)
The proposed approach has a direct connection to the [DPO](https://arxiv.org/pdf/2305.18290) and [CPL](https://arxiv.org/pdf/2310.13639) frameworks.
If we set the rejected sample to the generated sample $y$ and the accepted sample to the corrected sample $y'$, and choose $P$ to be $\pi/\pi_{old}$, then the DPO objective function becomes equivalent to the proposed objective function. Same can be observed with the CPL objective.


## Future Experiments
This hypothesis requires further evaluation using larger models, which will require increased computational resources. If you are interested in collaborating on these experiments, please get in touch.

Email: connectpraveenhegde@gmail.com

Linkedin: https://www.linkedin.com/in/praveen-hegde-7548a12b
