# LLM-Safety-Acceleration

A repo for our new research: **Fast and Accurate Unethical Rejection For LLMs Based on Residual Stream Classification**

## Abstract

Residual streams of LLMs are proved to have an ethical representation. In this paper, we present a novel algorithm which classifies the input is ethical or not. Our method performs safety classification at lower layers and has low computational overhead. Based on our approach, we introduce a plug-in defense framework. Mitigating novel jailbreaks only requires updating the classification algorithm, without tuning the model. Since rejection responses are often stylized, we propose a leap-layer inference acceleration method. This approach can reduce the computational overhead when responding to harmful inputs, making LLMs more CO2-friendly. 