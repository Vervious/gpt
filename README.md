# Hacking on GPT

> [!NOTE]
> This is an ongoing documentation of my exploration of the GPT (2) architecture. The primary goal is to answer the following questions:
> - Why does it work?
> - Can we do better?
> - Is the architecture theoretically motivated, and do we have reason to believe that it is a "best" learning algorithm? 


> [!TIP] 
> Here are some observations that I thought are interesting:
> - GPT is magic and feels almost perfect.
> - The output of `attn` may not need to be put into the residual. As long as it is fed as input into the `mlp`, the network still performs as well (at least at this scale), albeit slightly slower to train.
> - The structure of GPT reminds me almost of an advanced combinator calculus. If I had to prove the expressivity of the architecture, I would start there.
> - The `mlp` component in particular is quite flexible; it can be replaced by many fun variants, described below.
> - We can learn to "gate" between passing the entire residual, or a combination of the residual and block output, or exclusively the block output; however, this doesn't improve perplexity and also increases training time.


> [!IMPORTANT] 
> I started this project without much background knowledge of the literature (as a cryptographer by training). The documentation for many experiments is also very loose, since it was not indended to be shared publicly. Many early experiments are not documented at all. Many of the observations may seem easy or, alternatively, surprising. On later perusal of the literature, most have been discovered already.


### Useful commands


```torchrun --standalone --nproc_per_node=1 train_gpt2_ben.py```

```pip freeze -l > requirements.txt```

```source setup.sh```

```screen -dm bash -c 'torchrun --standalone --nproc_per_node=8 train_gpt2.py > log/screen.txt 2>&1'```

```screen -r```

```screen -dm bash -c 'torchrun --standalone --nproc_per_node=1 train_gpt2_ben.py > screen.txt 2>&1'```

Setup:

`pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126`

## Notes

### Early Experiments

Lost to time. These included a lot of basic explorations of removing the residual, `mlp(x)*attn(x)`, etc.

### Early fun: Computing different losses on a layer-by-layer basis

> [!NOTE]
> *Notes in retrospect*: This series of experiments probably makes more sense to do on a token-by-token basis, not a layer-by-layer basis. Here, I am conflating the language, and the computation. (Yes, this series of experiments fails. It is not very motivated.)

*Hypothesis*: One of my early motivations was to somehow make the model "self-consistent". That is, putting on an RL hat, perhaps we can interpret subsequent layers as giving "feedback" to earlier layers (in retrospect, this turns out to be a bad way of reasoning about it). In any case, imagine a world where a block wants to output a signal such that the subsequent block minimizes its "surprise" (i.e. `-log(1 - Pr[max])`).


We start off by computing an additional loss at every layer, and then later appending it to the final loss. Setting `block_loss = -log(1 - pr[max])` (the lower the confidence, the closer the block_loss is to 0; the higher the confidence, the more negative the block_loss (so when added to the loss, the loss gets lower, which is "good").). Special care is taken to compute the "evaluator" block under `torch.no_grad`.

It doesn't work very well at all! (*Notes in retrospect:* Graph lost to time.) One observed problem is that the confidence is being pushed very high on early layers, with no bearing on the final layer that outputs. Then, we could maybe investigate some early termination technique that samples from the embeddings as soon as the confidence is high enough.


##### 1-noise

*Hypothesis*: I suspect one issue was that the was that the `block_loss` was too big relative to the real signal; let us normalize it (i.e. for each layer, `-1 * crossentropy / n_layers`), but it turns out that this loss signal is still very noisy, so learning is not very good.

##### 2-test

*Hypothesis*: 
In this one, we set `losses += _block_loss / self.config.n_layer`, where `_block_loss = F.cross_entropy(_logits.view(-1, _logits.size(-1)), _targets.view(-1))` and then `_block_loss = torch.log(1 - torch.exp(-1*_block_loss))`, namely it is positive feedback only. As we can see, it doesn't ever get to good training error, but it is better than noise.

> [!NOTE]
> *Retroactive Note*: I forgot what was going on here.


##### 3-test

*Hypothesis*: Continuing on this "self-reward using confidence as heuristic" idea,
we want to incentivize high confidence (else loss accumulates forever?) whilst penalizing wrong answers. When there is no environmental feedback (i.e. when targets is None), the loss should just be the self-confidence. Whenever confidence is high, there is some probability of terminating the line of thought (which is good), yet also a chance of accumulating loss in prior steps. Then the model should learn to be confident early.

`loss_ = (xe * _confidence * _mask_BT).mean()`

> [!NOTE]
> *Retroactive Note*: I again don't remember what is going on, but we did try this early termination thing, and the conclusion is that it is relatively pointless. In retrospect, it reminds me of Universal Transformers (excluding the per-layer loss idea).

##### 4-test

*Hypothesis*: Previously, we evaluated the output of each layer against the target. 
Now, we don't evaluate the true loss against the target until the network is actually ready to output:

```
xe_factor = ((xe - 1) * _just_triggered + 1)
loss_ = (xe_factor * _confidence * _mask_BT).mean()
```

This has the effect of punishing confidence early on. Unfortunately, the result is still subpar.


##### 5-test

> [!NOTE]
> *Retroactive Note*: Another unmotivated experiment in retrospect, but it has the property of having the first graph that I saved. I remain curious how these ideas would play out at a token generation level (if not already used in these chain-of-thought settings or during post-training).

*Hypothesis*: Let's rethink using confidence of subsequent layers to penalize / reward earlier computation:
- Certainly, one should punish confidently wrong answers. But what if there is no target? Ask the next layer if wrong or not wrong. 

- On confidence: should we reward confidence? (By my simulateability theory, predictable actions are not interesting to me. So if the robot's action was predictable to me, that action is not interesting. But this seems to be different. There is a distinction between predictability and distinguishability. Also, "did I expect this" from a verifier's point of view, is different from "would I have done the same thing", because note that in self talk, the answer to the latter is always "yes". Maybe heuristically, if I am highly confident, then I did expect it -- I know how to act in return to maximize the true reward; if I am not at all highly confident, then I did not expect the answer at all (it doesn't look like giberish either), and have no clue and no confidence in how to act. Thus, we should reward low confidence, or punish high confidence.)

```xe_factor_prev = ((_xe_prev - 1) * _just_triggered_prev + 1)
loss_ = (xe_factor_prev * _confidence * _mask_BT_prev).mean()
```

Result: 

![loss plot](img/5-test.png)

(Strangely linear, but also outputs "the" a hundred times. On further debugging, penalizing confidence appears to cause this behavior. Why?)

##### 6-test

A few experiments:

- `6-test-1`: Rerun the "confidence of target" experiment with GPT learning rate.

- `0-noearly`: the same as vanilla GPT (but reusing weights) without our early termination mechanism. 

- `0-original` does not re-use any weights, and needs a smaller learning rate to converge properly.


Does confidence reinforcement even make sense at a high level? Recall what each layer does:

- the attention module: each embedding is the weighted sum of multiple embeddings from the previous layer in its context window, computed according to some "attention matrix".

- the MLP module is essentially a fact retrieval system; the 4A x A weight matrix + ReLU (or other non-linearity) can be (perhaps) thought of as evaluating a giant |4A|-number set of if statements on A-dimensional embeddings; the A x 4A projection matrix perhaps then "adds" facts to the original embedding (via the residual) depending on which "if statements" passed.

Our general hypothesis in this section was that good training data is only one part of learning; acts of "self-reflection" or "self-consistency" are also very important to learning. (Somehow, the model should make two predictions and check whether they are consistent, or be able to evaluate its own quality/consistency independenty of generating predictions.)

Note that the subsequent logit it generates is indeed such an assessment. 
The problem with using confidence as part of the loss is that...

```
A stream of   text that is
  of     text that is   next
  of     text that is   next
```

Well, we are in some sense in the wrong namespace; why are we using steps of the computation, to reward overall behavior? The computation hasn't even had time to finish.
<!-- strangely, the third layer is guessing not the third layer, but continuing to guess the second layer... If i don't feed the residual back in after applying the attention layer, performance is hurt substantially. Why is the residual so important in that case? Is the (previous) embedding itself really a very deep short term memory? Why can't inclusion of the past be learned? -->

<!-- I guess because we add value matrices instead of the embedding itself, the past gets destroyed. (How necessary is the value matrix? Shouldn't the MLP setup deal with that already. Perhaps the value matrices should just be the embedding itself.) -->

Note that the residual connection and c_proj are extremey important, and I do not know why. The value matrix does not seem so important. (perhaps we can get rid of c_proj?)

What if we reward tokens that are equal to the next token in the previous layer? Does it still have the same namespace issue?

> [!NOTE]
> *Retroactive Note*. I see no reason why intermediate computations should reflect the structure of "next token prediction."

### Experimenting with Signal Propagation

##### 8-experiments

> [!NOTE]
> *Retroactive Note*. A few experiments are lost, and at some point I started experimenting with multiple copies of the attn and mlp layers. Throughout, `all layer loss' refers to evaluating the output of each layer against the target in addition to the usual loss; later, we will see that this is both slower and slightly counterproductive (but the results should still generalize to the usual notion of loss). Initially, I thought all layer loss would help with signal propagation and lessen the need for residuals (indeed it does).

<!-- Note that here, x at first has already had a LayerNorm applied, whereas `res` refers to the actual skip (so `x = LayerNorm(res)`).  -->


> [!NOTE]
> *Retroactive Note*. At some point here, I realized by accident that doing `x = x + attn(LN(x)) * mlp(LN(x))` converges faster than the standard architecture, though the perplexity is a bit worse. For good measure, I retroactively ran the following comparison (the purple here is 8 layers instead of 12):
> ```
> Transformer, max LR 0.0006 n_layer 12
> Setting:
> ==details======
>  machine_code
> class MultExecute(nn.Module):
>     def __init__(self, config):
>         super().__init__()
>         self.mlp = MLP(config)
>     def forward(self, program, attn):
>         return self.mlp(program) * attn
> ----------------
>  machine_modules
>         self.compiler = BenCompilerNoOp(config)
>         self.execute = MultExecute(config)
> ----------------
>  block_logic
>         y = self.ln_1(x)
>         attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
>         program = self.compiler(y)
>         machineOutput = self.execute(program, attn)
>         newx = x + machineOutput
> ========
> VALUEMATRIX=True
> REUSE_WEIGHTS=False
> MLP_SCALE=4
> ATTENTION_SINK=False
> ATTENTION_MASK =False
> IDENTITY_LOSS=False
> CODE_MODE=False
> ```
> ![caption](img/18-axm.jpg)
>
> Also, note that the performance of `attn()*mlp()` is even more comparible to GPT if every block reuses weights, which is a regime that I was initially fascinated with (and forgot to turn off in my experiments).


The problem with `x = res*attn2(x) + attn(x), x = x + mlp(LN(x))` seems to be that the residuals are blowing up the more we train it. Same with `res*attn2(x) + attn(x), x * mlp2(LN(x)) + mlp(LN(x))`. (All Layer Loss). We see residuals like

```SIZE COMPARISON prev 2.5294246673583984 next 1.1539545059204102
SIZE COMPARISON prev 2.8568525314331055 next 1.1331876516342163
SIZE COMPARISON prev 2.7836289405822754 next 1.121557354927063
SIZE COMPARISON prev 3.347931385040283 next 1.1270164251327515
SIZE COMPARISON prev 5.639773368835449 next 1.1290345191955566
SIZE COMPARISON prev 15.21493911743164 next 1.1317602396011353
SIZE COMPARISON prev 60.20911407470703 next 1.1316640377044678
SIZE COMPARISON prev 284.3788757324219 next 1.1313221454620361
SIZE COMPARISON prev 1445.6865234375 next 1.1314541101455688
SIZE COMPARISON prev 7582.7626953125 next 1.1323127746582031
SIZE COMPARISON prev 40300.05078125 next 1.133501410484314
SIZE COMPARISON prev 216047.5625 next 1.1349073648452759
```

For something like
`x = x + self.attn(x)  x = x + self.mlp(self.ln_2(x))` (all layer loss) (note, using x instead of res) we see residuals like

```
SIZE COMPARISON prev 18.42753028869629 next 0.8354735374450684
SIZE COMPARISON prev 12.65165901184082 next 0.8403578996658325
SIZE COMPARISON prev 12.434549331665039 next 0.8322869539260864
SIZE COMPARISON prev 13.066632270812988 next 0.8329403400421143
SIZE COMPARISON prev 12.981801986694336 next 0.8330138921737671
SIZE COMPARISON prev 12.918157577514648 next 0.8329639434814453
SIZE COMPARISON prev 12.923635482788086 next 0.8329555988311768
SIZE COMPARISON prev 12.929137229919434 next 0.8329416513442993
SIZE COMPARISON prev 12.931181907653809 next 0.832944393157959
SIZE COMPARISON prev 12.930807113647461 next 0.8329448699951172
SIZE COMPARISON prev 12.930171966552734 next 0.8329458236694336
SIZE COMPARISON prev 12.92995834350586 next 0.832942008972168
```

For vanilla GPT (as usual shared weights), all layer loss, we see by step 1100 (slowly shrinking over more steps)

```
SIZE COMPARISON prev 0.8780975341796875 next 1.1142359972000122
SIZE COMPARISON prev 1.2143694162368774 next 1.110939383506775
SIZE COMPARISON prev 1.652571439743042 next 1.1084067821502686
SIZE COMPARISON prev 2.119270086288452 next 1.1072710752487183
SIZE COMPARISON prev 2.6001691818237305 next 1.1067460775375366
SIZE COMPARISON prev 3.0836727619171143 next 1.1065006256103516
SIZE COMPARISON prev 3.564718246459961 next 1.1063616275787354
SIZE COMPARISON prev 4.042917251586914 next 1.1062694787979126
SIZE COMPARISON prev 4.51833438873291 next 1.1062010526657104
SIZE COMPARISON prev 4.991661548614502 next 1.1061511039733887
SIZE COMPARISON prev 5.463232040405273 next 1.1061171293258667
SIZE COMPARISON prev 5.933814525604248 next 1.1060969829559326
```

and by step 4250:

```SIZE COMPARISON prev 1.5941141843795776 next 1.3911638259887695
SIZE COMPARISON prev 2.485067367553711 next 1.3869950771331787
SIZE COMPARISON prev 3.4810280799865723 next 1.3849396705627441
SIZE COMPARISON prev 4.4364542961120605 next 1.3832675218582153
SIZE COMPARISON prev 5.394400596618652 next 1.3827459812164307
SIZE COMPARISON prev 6.328531265258789 next 1.3822684288024902
SIZE COMPARISON prev 7.25205135345459 next 1.3819595575332642
SIZE COMPARISON prev 8.162321090698242 next 1.381672739982605
SIZE COMPARISON prev 9.063191413879395 next 1.3814265727996826
SIZE COMPARISON prev 9.955363273620605 next 1.3812041282653809
SIZE COMPARISON prev 10.840224266052246 next 1.3810186386108398
SIZE COMPARISON prev 11.719019889831543 next 1.3808491230010986
```

Most of the contribution seems to come from the attention step. In particular e.g. by step 799 for the same experiment as above:

```
SIZE COMPARISON prev 0.9725479483604431 mid 0.05692768841981888 next 1.1076934337615967
SIZE COMPARISON prev 1.208902359008789 mid 1.1301352977752686 next 1.109763503074646
SIZE COMPARISON prev 1.4090938568115234 mid 1.3024053573608398 next 1.111130714416504
SIZE COMPARISON prev 1.5330252647399902 mid 1.396193027496338 next 1.1153910160064697
SIZE COMPARISON prev 1.7066140174865723 mid 1.5469791889190674 next 1.11979341506958
SIZE COMPARISON prev 2.1395297050476074 mid 1.9626308679580688 next 1.1215941905975342
SIZE COMPARISON prev 3.0498666763305664 mid 2.85837459564209 next 1.1217771768569946
SIZE COMPARISON prev 4.754446029663086 mid 4.55536413192749 next 1.1214094161987305
SIZE COMPARISON prev 7.8321757316589355 mid 7.630302429199219 next 1.121267318725586
SIZE COMPARISON prev 13.233830451965332 mid 13.03246784210205 next 1.1213200092315674
SIZE COMPARISON prev 22.627168655395508 mid 22.428388595581055 next 1.1217448711395264
SIZE COMPARISON prev 38.77415466308594 mid 38.57970428466797 next 1.1223392486572266
```

Next, we see if layernorm learnable scale/shift parameters are actualy necessary. Probably not! For res + attn(x), without learnable parameters, by step 499:

```
SIZE COMPARISON prev 1.3563252687454224 mid 0.4865388572216034 next 1.0006482601165771
SIZE COMPARISON prev 1.7924420833587646 mid 1.4573729038238525 next 1.0006499290466309
SIZE COMPARISON prev 2.284381628036499 mid 1.9278755187988281 next 1.0006506443023682
SIZE COMPARISON prev 2.821475028991699 mid 2.4290308952331543 next 1.0006510019302368
SIZE COMPARISON prev 3.3831615447998047 mid 2.9755988121032715 next 1.000651240348816
SIZE COMPARISON prev 3.9586188793182373 mid 3.545346736907959 next 1.0006513595581055
SIZE COMPARISON prev 4.539757251739502 mid 4.125799179077148 next 1.000651478767395
SIZE COMPARISON prev 5.122259140014648 mid 4.709800720214844 next 1.0006515979766846
SIZE COMPARISON prev 5.703899383544922 mid 5.293674945831299 next 1.0006515979766846
SIZE COMPARISON prev 6.283486843109131 mid 5.875919342041016 next 1.0006515979766846
SIZE COMPARISON prev 6.860944747924805 mid 6.455946922302246 next 1.0006515979766846
SIZE COMPARISON prev 7.435906887054443 mid 7.033243179321289 next 1.0006515979766846
```


If we use res*attn(x), weirdly (and as previously seen), the residuals blow up. For LN without learnable parameters (everylayer loss, shared weights), by step 3800:

```
SIZE COMPARISON prev 0.9520694017410278 mid 0.05660898983478546 next 1.0006455183029175
SIZE COMPARISON prev 1.1140660047531128 mid 1.0007506608963013 next 1.0006468296051025
SIZE COMPARISON prev 1.1631948947906494 mid 0.9358635544776917 next 1.000647783279419
SIZE COMPARISON prev 2.937286853790283 mid 2.5033302307128906 next 1.0006508827209473
SIZE COMPARISON prev 23.77895736694336 mid 23.278114318847656 next 1.0006515979766846
SIZE COMPARISON prev 211.52383422851562 mid 211.03501892089844 next 1.0006515979766846
SIZE COMPARISON prev 1932.359130859375 mid 1931.881103515625 next 1.0006515979766846
SIZE COMPARISON prev 17739.107421875 mid 17738.642578125 next 1.0006515979766846
SIZE COMPARISON prev 162568.78125 mid 162568.328125 next 1.0006515979766846
SIZE COMPARISON prev 1477890.75 mid 1477890.25 next 1.0006515979766846
SIZE COMPARISON prev 13287474.0 mid 13287474.0 next 1.0006515979766846
SIZE COMPARISON prev 117777120.0 mid 117777120.0 next 1.0006515979766846
```

Perhaps this means that attn(x) outputs very small numbers, and they are trying to compensate... generally very confusing.


I wonder if res + attn(ln(res)) is at heart performing a "substitutition" into res. And x + mlp(ln(x)) evaluates an if statement and... maybe it should be

```y = res + attn(ln(res))
x = x + mlp(ln(y))
```

and we don't add y back in?

Well, it turns out this sucks, similarly to x = x + attn(x):

![loss plot](img/10-resescapei.png)

What about

```y = res + attn(ln(res))
x = res + mlp(ln(y))
```

well:

![loss plot](img/10-x-escapei.png)


(performance seems truly worse) so it is really quite important that the logits directly get the output of the attention layer (instead of solely feeding the attention layer into the mlp and then the output).

Note that for double attention `res*attn2(x) + attn(x), x + mlp(LN(x))`, the residuals still blow up:

```
@ 949 train 5.2751 , allloss: 64.1261, confloss: 0.0000, targetloss: 0.0000, earlystop: 0.000, earlystopdict: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lr:5.9999e-04, norm:7.9968, dt: 1769.77ms, tok/sec: 74061.71, flops:30.47, batch-reuse:1
SIZE COMPARISON prev 1.0659868717193604 next 1.0947023630142212
SIZE COMPARISON prev 1.798586368560791 next 1.1034561395645142
SIZE COMPARISON prev 2.1790177822113037 next 1.107276439666748
SIZE COMPARISON prev 2.6335883140563965 next 1.105468511581421
SIZE COMPARISON prev 4.98344612121582 next 1.1039302349090576
SIZE COMPARISON prev 14.011809349060059 next 1.1064351797103882
SIZE COMPARISON prev 56.432098388671875 next 1.106866478919983
SIZE COMPARISON prev 249.68592834472656 next 1.1070586442947388
SIZE COMPARISON prev 1164.308837890625 next 1.1072487831115723
SIZE COMPARISON prev 5611.69482421875 next 1.1074464321136475
SIZE COMPARISON prev 27626.546875 next 1.1076771020889282
SIZE COMPARISON prev 138020.09375 next 1.1078872680664062
```

For `res + attn(x), x * mlp(LN(x))`, it still blows up, but a little slower:

```
@ 949 train 5.3970 , allloss: 65.0632, confloss: 0.0000, targetloss: 0.0000, earlystop: 0.000, earlystopdict: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lr:5.9999e-04, norm:8.4142, dt: 1606.70ms, tok/sec: 81578.15, flops:35.93, batch-reuse:1
SIZE COMPARISON prev 0.6984971761703491 mid 0.6713976860046387 next 1.0006380081176758
SIZE COMPARISON prev 0.9332523345947266 mid 0.7942314147949219 next 1.0006457567214966
SIZE COMPARISON prev 1.5732736587524414 mid 1.0734591484069824 next 1.0006494522094727
SIZE COMPARISON prev 2.531172752380371 mid 1.7262670993804932 next 1.0006508827209473
SIZE COMPARISON prev 4.456131935119629 mid 2.7139768600463867 next 1.0006513595581055
SIZE COMPARISON prev 7.704921245574951 mid 4.647169589996338 next 1.0006515979766846
SIZE COMPARISON prev 14.081094741821289 mid 7.897726058959961 next 1.0006515979766846
SIZE COMPARISON prev 25.32900619506836 mid 14.263557434082031 next 1.0006515979766846
SIZE COMPARISON prev 47.37249755859375 mid 25.50238800048828 next 1.0006515979766846
SIZE COMPARISON prev 87.77811431884766 mid 47.53529357910156 next 1.0006515979766846
SIZE COMPARISON prev 166.31536865234375 mid 87.93147277832031 next 1.0006515979766846
SIZE COMPARISON prev 313.864501953125 mid 166.46051025390625 next 1.0006515979766846
...
@ 3499 train 4.1087 , allloss: 50.1864, confloss: 0.0000, targetloss: 0.0000, earlystop: 0.000, earlystopdict: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lr:5.9982e-04, norm:7.7492, dt: 1606.40ms, tok/sec: 81593.42, flops:35.93, batch-reuse:1
SIZE COMPARISON prev 0.9271938800811768 mid 0.6703000068664551 next 1.000638723373413
SIZE COMPARISON prev 1.5912479162216187 mid 1.1378358602523804 next 1.000647783279419
SIZE COMPARISON prev 2.6288633346557617 mid 1.8008546829223633 next 1.0006506443023682
SIZE COMPARISON prev 7.2939934730529785 mid 2.786419630050659 next 1.0006515979766846
SIZE COMPARISON prev 53.24920654296875 mid 7.418212890625 next 1.0006517171859741
SIZE COMPARISON prev 499.09014892578125 mid 53.380401611328125 next 1.0006515979766846
SIZE COMPARISON prev 4886.146484375 mid 499.2305908203125 next 1.0006515979766846
SIZE COMPARISON prev 48644.1953125 mid 4886.2861328125 next 1.0006517171859741
SIZE COMPARISON prev 489250.40625 mid 48644.3359375 next 1.0006515979766846
SIZE COMPARISON prev 4965097.5 mid 489250.53125 next 1.0006515979766846
SIZE COMPARISON prev 50753656.0 mid 4965097.5 next 1.0006515979766846
SIZE COMPARISON prev 521954624.0 mid 50753656.0 next 1.0006515979766846
```

What about

```y = res + attn(ln(res))
x = y + mlp(ln(res))
```

i.e. how important is it that the output of attention gets fed into the MLP? It turns out, this works surprisingly well; i.e. attention and MLP both seem to be additive, *independent* components:

![loss plot](img/10-resmlpi.png)

```
@ 4099 train 4.1100 , allloss: 50.7263, confloss: 0.0000, targetloss: 0.0000, earlystop: 0.000, earlystopdict: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lr:5.9975e-04, norm:4.9580, dt: 1626.09ms, tok/sec: 80605.68, flops:35.50, batch-reuse:1
SIZE COMPARISON prev 1.7276793718338013 mid 0.828337550163269 next 1.0006475448608398
SIZE COMPARISON prev 2.3453421592712402 mid 1.9509013891220093 next 1.0006494522094727
SIZE COMPARISON prev 3.0417964458465576 mid 2.5768418312072754 next 1.0006505250930786
SIZE COMPARISON prev 3.7537474632263184 mid 3.24953031539917 next 1.0006508827209473
SIZE COMPARISON prev 4.469127655029297 mid 3.953681230545044 next 1.0006511211395264
SIZE COMPARISON prev 5.175540447235107 mid 4.6598124504089355 next 1.0006513595581055
SIZE COMPARISON prev 5.872206687927246 mid 5.358777046203613 next 1.0006513595581055
SIZE COMPARISON prev 6.5586748123168945 mid 6.048742294311523 next 1.000651478767395
SIZE COMPARISON prev 7.23720645904541 mid 6.730024337768555 next 1.0006515979766846
SIZE COMPARISON prev 7.907989501953125 mid 7.40459680557251 next 1.0006515979766846
SIZE COMPARISON prev 8.572383880615234 mid 8.07198715209961 next 1.0006515979766846
SIZE COMPARISON prev 9.231319427490234 mid 8.7335205078125 next 1.0006515979766846
```

Note that the above doesn't matter if we change the order of mlp and attn, namely

```y = res + mlp(ln(res))
x = y + attn(ln(res))
```

is equivalent.

What happens if we remove the "all layer loss", and compute loss as per the usual method?

![loss plot](img/10-resmlp-singlei.png)

It is in fact, better; the all layer loss is utterly useless.

But note that the std does not blow up as much, weirdly:

```
@ 4399 train 3.7913 , allloss: 3.7913, confloss: 0.0000, targetloss: 0.0000, earlystop: 0.000, earlystopdict: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lr:5.9971e-04, norm:0.6440, dt: 411.78ms, tok/sec: 318308.01, flops:140.18, batch-reuse:1
SIZE COMPARISON prev 2.829397439956665 mid 2.238248825073242 next 1.0006511211395264
SIZE COMPARISON prev 2.9154844284057617 mid 2.7668395042419434 next 1.0006510019302368
SIZE COMPARISON prev 2.937965154647827 mid 2.9097206592559814 next 1.0006510019302368
SIZE COMPARISON prev 2.847860336303711 mid 2.835615634918213 next 1.0006510019302368
SIZE COMPARISON prev 2.772952079772949 mid 2.7625179290771484 next 1.0006508827209473
SIZE COMPARISON prev 2.7469887733459473 mid 2.7138304710388184 next 1.0006508827209473
SIZE COMPARISON prev 2.775296211242676 mid 2.707706928253174 next 1.0006508827209473
SIZE COMPARISON prev 2.8555359840393066 mid 2.748220682144165 next 1.0006508827209473
SIZE COMPARISON prev 2.987206220626831 mid 2.838392496109009 next 1.0006510019302368
SIZE COMPARISON prev 3.1633992195129395 mid 2.9731945991516113 next 1.0006511211395264
SIZE COMPARISON prev 3.425197124481201 mid 3.1695663928985596 next 1.000651240348816
SIZE COMPARISON prev 3.9492619037628174 mid 3.533745765686035 next 1.0006513595581055
```

What happens if we 2x the attention component?

![loss plot](img/10-resmlp-single-2xi.png)

```
@ 2449 train 4.2471 , allloss: 4.2471, confloss: 0.0000, targetloss: 0.0000, earlystop: 0.000, earlystopdict: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lr:5.9991e-04, norm:0.7454, dt: 411.36ms, tok/sec: 318631.23, flops:140.32, batch-reuse:1
SIZE COMPARISON prev 3.474045753479004 mid 2.998915672302246 next 1.0006513595581055
SIZE COMPARISON prev 3.860865592956543 mid 3.8047633171081543 next 1.0006513595581055
SIZE COMPARISON prev 3.176105499267578 mid 3.17417573928833 next 1.0006511211395264
SIZE COMPARISON prev 2.7696032524108887 mid 2.7842602729797363 next 1.0006510019302368
SIZE COMPARISON prev 2.601201057434082 mid 2.6080410480499268 next 1.0006508827209473
SIZE COMPARISON prev 2.55460524559021 mid 2.5488529205322266 next 1.0006508827209473
SIZE COMPARISON prev 2.535594940185547 mid 2.500566244125366 next 1.0006508827209473
SIZE COMPARISON prev 2.5360474586486816 mid 2.458188056945801 next 1.0006508827209473
SIZE COMPARISON prev 2.598814010620117 mid 2.465818405151367 next 1.0006508827209473
SIZE COMPARISON prev 2.779794216156006 mid 2.578892707824707 next 1.0006510019302368
SIZE COMPARISON prev 3.112057685852051 mid 2.8377723693847656 next 1.0006511211395264
SIZE COMPARISON prev 3.612070083618164 mid 3.259913444519043 next 1.0006513595581055
```

It is worse... why? But it eventually converges, so it doesn't really matter. What if we 0.5x it?

![loss plot](img/10-resmlp-single-halfxi.png)

```
@ 1999 train 4.1020 , allloss: 4.1020, confloss: 0.0000, targetloss: 0.0000, earlystop: 0.000, earlystopdict: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lr:5.9994e-04, norm:0.6649, dt: 422.70ms, tok/sec: 310084.43, flops:136.56, batch-reuse:1
val: loaded 100000000 tokens (first shard)
val: 1 epoch (1 shard) = 12207 mini-batches
validation loss: 4.2778
SIZE COMPARISON prev 1.7215176820755005 mid 1.1063441038131714 next 1.0006499290466309
SIZE COMPARISON prev 2.114750385284424 mid 1.966301441192627 next 1.0006506443023682
SIZE COMPARISON prev 1.981388807296753 mid 2.1172070503234863 next 1.0006502866744995
SIZE COMPARISON prev 1.8269994258880615 mid 1.9413310289382935 next 1.00065016746521
SIZE COMPARISON prev 1.6911734342575073 mid 1.7758653163909912 next 1.0006499290466309
SIZE COMPARISON prev 1.6034241914749146 mid 1.6564953327178955 next 1.0006498098373413
SIZE COMPARISON prev 1.5692648887634277 mid 1.589362621307373 next 1.0006496906280518
SIZE COMPARISON prev 1.5858174562454224 mid 1.5691030025482178 next 1.0006496906280518
SIZE COMPARISON prev 1.6530719995498657 mid 1.5916287899017334 next 1.0006498098373413
SIZE COMPARISON prev 1.7865718603134155 mid 1.6776154041290283 next 1.0006500482559204
SIZE COMPARISON prev 2.007704973220825 mid 1.845852017402649 next 1.000650405883789
SIZE COMPARISON prev 2.3220303058624268 mid 2.0987911224365234 next 1.0006507635116577
```

It is identical... So it seems that attention is not particularly useful for early gains; it's really the MLP that matters. If we skip out on attention completely:

![loss plot](img/10-resmlp-single-noattni.png)


```
@ 1699 train 5.8143 , allloss: 5.8143, confloss: 0.0000, targetloss: 0.0000, earlystop: 0.000, earlystopdict: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lr:5.9996e-04, norm:0.2052, dt: 272.28ms, tok/sec: 481386.85, flops:212.00, batch-reuse:1
SIZE COMPARISON prev 3.032069206237793 mid 0.034484002739191055 next 1.0006475448608398
SIZE COMPARISON prev 4.604878902435303 mid 3.032069206237793 next 1.0006513595581055
SIZE COMPARISON prev 4.4857869148254395 mid 4.604878902435303 next 1.0006513595581055
SIZE COMPARISON prev 4.478532791137695 mid 4.4857869148254395 next 1.0006513595581055
SIZE COMPARISON prev 4.600770473480225 mid 4.478532791137695 next 1.000651478767395
SIZE COMPARISON prev 4.848290920257568 mid 4.600770473480225 next 1.0006513595581055
SIZE COMPARISON prev 5.205094337463379 mid 4.848290920257568 next 1.0006513595581055
SIZE COMPARISON prev 5.654345989227295 mid 5.205094337463379 next 1.0006515979766846
SIZE COMPARISON prev 6.184695720672607 mid 5.654345989227295 next 1.0006513595581055
SIZE COMPARISON prev 6.788667678833008 mid 6.184695720672607 next 1.000651478767395
SIZE COMPARISON prev 7.458611965179443 mid 6.788667678833008 next 1.0006515979766846
SIZE COMPARISON prev 8.18480110168457 mid 7.458611965179443 next 1.0006517171859741
```

It is worse! Whew. Our efforts are validted. But note that the residual still grows... Here, prev is the size of newres, x is the size of ln(prev), and mid is the size of prevres (as opposed to prevres + attn). So MLP is definitely adding a component (perhaps over and over again). 


What happens (for fun) if we do `y = res  + self.attn(x)*self.mlp(x)`:

![loss plot](img/10-resmlp-single-attnxmlpi.png)

```
@ 6349 train 3.8610 , allloss: 3.8610, confloss: 0.0000, targetloss: 0.0000, earlystop: 0.000, earlystopdict: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lr:5.9941e-04, norm:0.7167, dt: 409.91ms, tok/sec: 319757.28, flops:140.82, batch-reuse:1
SIZE COMPARISON prev 4.092894554138184 mid 4.092894554138184 next 1.0006513595581055
SIZE COMPARISON prev 4.809565544128418 mid 4.809565544128418 next 1.0006513595581055
SIZE COMPARISON prev 4.6843671798706055 mid 4.6843671798706055 next 1.0006513595581055
SIZE COMPARISON prev 4.60699462890625 mid 4.60699462890625 next 1.0006513595581055
SIZE COMPARISON prev 4.567028999328613 mid 4.567028999328613 next 1.0006513595581055
SIZE COMPARISON prev 4.567410945892334 mid 4.567410945892334 next 1.0006513595581055
SIZE COMPARISON prev 4.6050896644592285 mid 4.6050896644592285 next 1.0006513595581055
SIZE COMPARISON prev 4.691596508026123 mid 4.691596508026123 next 1.000651478767395
SIZE COMPARISON prev 4.89960241317749 mid 4.89960241317749 next 1.0006513595581055
SIZE COMPARISON prev 5.414493083953857 mid 5.414493083953857 next 1.0006515979766846
SIZE COMPARISON prev 6.540674686431885 mid 6.540674686431885 next 1.0006515979766846
SIZE COMPARISON prev 8.01706314086914 mid 8.01706314086914 next 1.0006517171859741
rank 0 sample 0: A Poem for you! Roses are red, Potatoes are 
```

It seems to be the best yet.

What happens if we only feed in `attn(x) + mlp(x)` into the attn and mlp components of the next layer, and not the res? (This one crashes).

What happens if we remove the value matrix, and instead just use the identity matrix, and them sum together the output of all of the heads?


![loss plot](img/10-resmlp-single-axm-novaluei.png)

```
@ 4699 train 3.9265 , allloss: 3.9265, confloss: 0.0000, targetloss: 0.0000, earlystop: 0.000, earlystopdict: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lr:5.9967e-04, norm:0.6336, dt: 1324.04ms, tok/sec: 98993.98, flops:42.88, batch-reuse:1
INFO nextres 33.397308349609375 attn*mlp 33.40809631347656 layernormed 1.0006517171859741
	 attn_hist -29.4375<tensor([ 45., 134., 166., 263., 119.,  41.])>29.25 mlp_hist -10.4375<tensor([ 24., 123., 402., 207.,  10.,   2.], dtype=torch.bfloat16)>15.0625
INFO nextres 193.25694274902344 attn*mlp 195.94659423828125 layernormed 1.0006517171859741
	 attn_hist -75.375<tensor([  1.,  11., 467., 249.,  38.,   2.])>72.75 mlp_hist -58.5<tensor([ 74., 122., 255., 280.,  33.,   5.], dtype=torch.bfloat16)>80.0
INFO nextres 120.01077270507812 attn*mlp 82.8893814086914 layernormed 1.0006517171859741
	 attn_hist -52.5<tensor([ 14., 545., 197.,  10.,   1.,   1.])>116.25 mlp_hist -18.125<tensor([  7.,   3., 272., 278., 190.,  17.], dtype=torch.bfloat16)>6.28125
INFO nextres 71.63121032714844 attn*mlp 55.632476806640625 layernormed 1.0006517171859741
	 attn_hist -45.75<tensor([ 14., 526., 212.,  12.,   2.,   2.])>102.75 mlp_hist -13.0625<tensor([  7.,  24., 362., 226., 137.,  12.], dtype=torch.bfloat16)>5.21875
INFO nextres 44.969993591308594 attn*mlp 30.2081298828125 layernormed 1.0006517171859741
	 attn_hist -49.875<tensor([  8., 436., 308.,  12.,   2.,   2.])>106.5 mlp_hist -7.6875<tensor([ 11., 276., 183., 214.,  74.,  10.], dtype=torch.bfloat16)>3.8125
INFO nextres 29.09838104248047 attn*mlp 18.429052352905273 layernormed 1.0006517171859741
	 attn_hist -45.375<tensor([ 27., 598., 130.,  10.,   1.,   2.])>109.5 mlp_hist -4.8125<tensor([155., 192., 158., 181.,  71.,  11.], dtype=torch.bfloat16)>2.78125
INFO nextres 19.71997833251953 attn*mlp 12.138340950012207 layernormed 1.0006517171859741
	 attn_hist -35.8125<tensor([132., 586.,  42.,   6.,   1.,   1.])>112.5 mlp_hist -3.65625<tensor([210., 152., 167., 177.,  56.,   6.], dtype=torch.bfloat16)>2.609375
INFO nextres 15.027619361877441 attn*mlp 7.450224876403809 layernormed 1.0006517171859741
	 attn_hist -34.5<tensor([121., 542.,  94.,   9.,   1.,   1.])>99.0 mlp_hist -3.25<tensor([166., 180., 143., 212.,  57.,  10.], dtype=torch.bfloat16)>2.09375
INFO nextres 13.641393661499023 attn*mlp 5.161359786987305 layernormed 1.0006517171859741
	 attn_hist -36.75<tensor([ 47., 202., 477.,  38.,   2.,   2.])>73.875 mlp_hist -3.1875<tensor([ 13., 253., 178., 238.,  70.,  16.], dtype=torch.bfloat16)>1.609375
INFO nextres 15.402214050292969 attn*mlp 6.81701135635376 layernormed 1.0006517171859741
	 attn_hist -36.5625<tensor([ 28., 136., 418., 123.,  58.,   5.])>44.25 mlp_hist -4.125<tensor([  4.,  26.,  81., 354., 270.,  33.], dtype=torch.bfloat16)>1.5703125
INFO nextres 19.019399642944336 attn*mlp 8.641536712646484 layernormed 1.0006517171859741
	 attn_hist -30.0<tensor([ 65., 130., 418.,  95.,  55.,   5.])>43.5 mlp_hist -5.03125<tensor([ 24., 156., 308.,  90., 113.,  78.], dtype=torch.bfloat16)>5.65625
INFO nextres 20.741458892822266 attn*mlp 7.100666046142578 layernormed 1.0006517171859741
	 attn_hist -34.3125<tensor([ 21., 186., 411., 108.,  38.,   4.])>50.625 mlp_hist -5.34375<tensor([ 23., 328., 159.,  73.,  66., 120.], dtype=torch.bfloat16)>8.1875
```

It is almost as good, but definitely not as good (I wonder if because we got rid of that one projection matrix).



Recall that in combinator calculuses, we need to be able to (1) copy arguments (inverting this operation is kind of the point of compression / learning a computation) and (2) apply arguments to each other (programmability). When inverting, I suspect it is just pattern match replacement (is this the MLP). Magnitude is somehow important for the encoding. Attention combines tokens into single embeddings? Is this some manifestation of programmability. Or, is everything truly in the forward direction. Attention takes a weighted sum of prior tokens.

Somehow, addition feels like an application / one-step evaluation / perhaps it refers to the depth of the tree, and each embedding dimension is like a possible subtree at each level. Or does attention give tree structure.

Somehow, a sequence of embeddings represents code. The attention component learns how individual tokens (read subtrees) programmatically act on other tokens (other subtrees). (But what is this value matrix?) Note that addition does not distinguish between left and right subtrees, or have an order, so how do we deal with it...  


General framework:
- Backprop performs memorization of substitution rules.
- regularization through LN limits "how much" we can memorize (limiting standard deviation).
- forward pass performs the actual computation.

Try again:
- the embedding itself encodes things like its position in the tree... and also its subtree... which is itself a sum of embeddings... (how is this possible? We only give it a 1D positional embedding.)
- The MLP maps nodes (i.e. subtrees) to compressed inverses.
- Attention then joins more nodes together as appropriate. (Why not join all nodes together? Well, maybe MLP cannot distinguish such a big sum? It shouldn't actually matter..., MLP should be able to distinguish. So why?) (If we sum everything together at every T, then the MLP triggers for every location T.) So somehow, we want the MLP to trigger selectively and in the right order.

But how come the MLP can't pick out specific patterns by itself? Why does it need the attention... what does element-wise multiplication mean, between two embeddings? A masked embedding? Attention is the mask (does it generate only boolean outputs)? MLP is the mask? why do we need to mask  the output of the mlp? What does magnitude of embedding mean?

An embedding does not see its sibling trees. Attention computes (given the embeddings) which  other nodes each on acts on / is connected to, outputting a parent node (the sum of the child nodes). The MLP layer takes as input an embedding (subtree) and inverts it (ideally to something more compressed). But the result is added on. Why is it added on? It should instead replace the embedding (but maybe not the whole thing, of only a subtree was inverted?)


Question:

what happens if i backprop every layer, but only propagate the gradient one layer

termination when it no longer updates / when it converges



## On Backprop

Note that the all_layer generally does seem slightly worse. For instance:

```testname = "10-resmlp-single-axm-novalue-copy-alllayer"
basename = "10-resmlp-single-axm-novalue"
```

![loss plot](img/10-resmlp-single-axm-novalue-copy-alllayeri.png)

With all-layer loss, the mlp size does seem smaller, and the graphs seemed more centered.

Let us now try it without deep propagation, i.e. all layer but only propagating the gradient one step:


## RMSNorm

Now, going bak to single layer loss, using the axm architecture, what does it look like with RMSNorm?


## On MLP

One hypothesis is that the MLP performs a copy of the embedding in a memorized location, i.e. it rotates it, or adds a positional embedding (and maybe this is why the residual is important?)

Try MLP(x+LN(y))

Position embeddings are definitely additive. If we switch from `x = tok_emb + pos_emb` to `x = tok_emb * pos_emb`, the capability of the model to compute is completely destroyed:

![loss plot](img/11-axm-multposi.png)

If we do `y = self.attn(x), z=self.mlp(x*y)*y`, the capability is slightly weakened, but not fully destroyed (it probably also converges?)

![loss plot](img/11-axm-mlpinputmodi.png)

Adjusting the input again, `y = self.attn(x), z=self.mlp(x+y)*y`

![loss plot](img/11-axm-mlpinputmod2i.png)

So really, we want to feed in just x into the mlp. (Why?)

Now, let's see what happens if we don't feed in the residual to future layers, but we do add the positional embedding back in at every layer:

![loss plot](img/11-axm-addposembagain-noresi.png)

Nope! The residual is clearly important for reasons other than the initial positional embedding.

What if we do

```
attn = self.attn(y, y)
mlp = self.mlp(x)
y = attn*mlp
x = res + y
newres = x
x = self.ln(x)
```

This crashes because values quickly go to infinity or zero.


What if we do: `y = KQ(x) @ mlp(x)` instead? Well, it turns out worse. So somehow the fact that we are doing a linear combination of `x` and not `mlp(x)` is important. But note, mlp is not introduced anywhere else! (We took it away from the `attn*mlp` term.) And it still does seem to be able to learn. I.e.
```
mlp = self.mlp(x)
attn = self.attn(x, mlp)
y = attn
x = res + y
newres = x
x = self.ln(x)
```
![loss plot](img/11-axm-attnofzi.png)


What if we do `y = KQ(mlp(x)) @ x`. Then, as before, it performs worse, but it is not fatal:
```
mlp = self.mlp(x)
attn = self.attn(mlp, x)
y = attn
x = res + y
newres = x
x = self.ln(x)
```
![loss plot](img/11-a-kqmlp.png)


Out of curiosity, what if we get rid of mlp entirely. It turns out both of the above experiments look exactly like this one! So certainly the non-linearity is essential, but it does not show up immediately.
```
attn = self.attn(x, x)
y = attn
x = res + y
newres = x
x = self.ln(x)
```
![loss plot](img/11-a-nomlp.png)


What if run attn on mlp only; this is much worse than nomlp, and also much worse than self.attn(mlp,x). Of all of these experiments (without including the mlp component elsewhere), `self.attn(mlp, x)` is for some reason the best (and `self.attn(mlp, mlp)` is the worst).
```
mlp = self.mlp(x)
attn = self.attn(mlp, mlp)
y = attn
x = res + y
newres = x
x = RMSNorm(x, ELEMENTWISEAFFINE={ELEMENTWISEAFFINE}), 
```
![loss plot](img/11-a-mlpasinput.png)


Does it matter at all what I feed into attn?
```
mlp = self.mlp(x)
attn = self.attn(mlp, x)
y = attn*mlp
x = res + y
newres = x
x = RMSNorm(x, ELEMENTWISEAFFINE={ELEMENTWISEAFFINE}), 
```
![loss plot](img/11-axm-mlpvalue.png)


Directly comparing the previous to the vanilla experiment... (orange is attn(x, x))
```
mlp = self.mlp(x)
attn = self.attn(x, x)
y = attn*mlp
x = res + y
newres = x
x = RMSNorm(x, ELEMENTWISEAFFINE={ELEMENTWISEAFFINE}), 
```
![loss plot](img/11-axm-xvalue-vs-mlp.png)


so, yeah. I'm stumped. The best still seems to be `x = LN(res + attn(x)*mlp(x))`. For good measure, let's try feeding only attn into the mlp input: (This one should be very bad if I remember correctly)
```
attn = self.attn(x, x)
mlp = self.mlp(attn)
y = attn*mlp
x = res + y
newres = x
x = RMSNorm(x, ELEMENTWISEAFFINE={ELEMENTWISEAFFINE}), 
```
![loss plot](img/11-axm-mlponattn.png)


Oh hey. it's surprisingly not so bad, but it is definitely worse. When was it bad? Perhaps when it was `attn + mlp`. Let's try:
```
attn = self.attn(x, x)
mlp = self.mlp(attn)
y = attn+mlp
x = res + y
newres = x
x = RMSNorm(x, ELEMENTWISEAFFINE={ELEMENTWISEAFFINE}), 
```
![loss plot](img/11-apm-mlponattn.png)

For good measure, let's run the vanilla GPT with RMSNorm and no value matrix:
```
attn = self.attn(x, x)
mlp = self.mlp(x)
y = attn+mlp
x = res + y
newres = x
x = RMSNorm(x, ELEMENTWISEAFFINE={ELEMENTWISEAFFINE}), 
```
![loss plot](img/11-apm.png)


In general, it seems important to feed the whole `x` into the mlp and attention layers. In some sense, `x` represents an entire embedding, whereas attn and mlp perhaps generate only part of it. 

Now, one last time, why is the residual so important? What happens again if I do this:
```
attn = self.attn(x, x)
mlp = self.mlp(x)
y = attn*mlp
x = y
newres = x
x = RMSNorm(x, ELEMENTWISEAFFINE={ELEMENTWISEAFFINE}), 
```
![loss plot](img/11-axm-nores.png)

It doesn't work particularly well, but what happens if I turn All Layer Loss back on?

![loss plot](img/11-axm-nores-alllayer.png)


So surprisingly, removing the residual is fine if we do `attn*mlp`, as long as we have all layer loss. It does seem to converge a little slower. If we do `attn + mlp`, what happens?

(todo)


I do think that from first principles, it should be as follows (with the residual, because the mlp is allowed to be a `no-op'.). But, concretely, this doesn't work as well for some reason (maybe it converges over time?):
```
attn = self.attn(x, x)
mlp = self.mlp(attn)
y = mlp
x = y + res
newres = x
x = RMSNorm(x, ELEMENTWISEAFFINE={ELEMENTWISEAFFINE}), 
```
![loss plot](img/11-mlponly-mlponattn.png)


Or, maybe even `y*res`? Nope, the loss here is stuck and doesn't change, so there is some numerical problem.
```
attn = self.attn(x, x)
mlp = self.mlp(attn)
y = mlp
x = y * res
newres = x
x = RMSNorm(x, ELEMENTWISEAFFINE={ELEMENTWISEAFFINE}), 
```
![loss plot](img/11-mlponly-mlponattn-times.png)


Let's try this one again but train it longer. IT does eventually converge to the same place, I think!
```
attn = self.attn(x, x)
mlp = self.mlp(attn)
y = mlp
x = y + res
newres = x
x = RMSNorm(x, ELEMENTWISEAFFINE={ELEMENTWISEAFFINE}), 
```
![loss plot](img/11-mlponly-mlponattn-value.png)

Generally, we need a way for mlp to express a `no-op`. Also, one question is why taking the attention out actually makes it train faster.

# Bringing the forward pass together

One question is, how come we don't just sum together the entire context window? Well, the thing is that some tokens attend to other tokens selectively, and not others. So it could be that `[A B]` does something, but `[C B]` does not, and we should really only sum tokens together when we know it (potentially) does something. (Or, does attention compute instead whether `A` and `B` are positioned together / near each other?) (In that case, this whole key query interpretation is kind of ridiculous, and can maybe be replaced by inverses of the position embedding.)

So, continuing this line of thought, if attention computes whether two tokens are positioned adjacent to each other (whcih would be a function of their *positions* only and not necessarily their semantic content, unless the two are conflated, which they may well be in natural language), then mlp should compute its *replacement*, namely, if `C = [A B]`, then MLP should learn `C`. This has the property that it describes both the inverse computation and the forward computation; i.e. if we are trying to compress the output of some computation, I think that this compression map can be learned exactly by learning `C` (i.e. learning some language for the forward computation). [Not going to lie, this is somewhat trippy.] 

Importantly, it could also be that `[A B]` is a no-op, that is, it is already in its most simplified form, and so `[A B] = [A B]`. It could also be that `[A B C D E] = [A B' C D E]` is mostly a no-op, but there is some small operation. But, it seems that `[A B]` should fully replace the separate `A` and `B` from the previous layer (i.e. applying positional properties should not result in a no-op). The no-op could be implemented using a gate much like in an LSTM. Alternatively, a simple addition may be enough (not sure how to do a full replacement). So, I think, from these vague "principles", I expect the following architecture to be ideal (in terms of expressiveness, and interpretability, but not sure about optimizability):

```
x = self.attn(x, x) + x # (is this +x necessary? Or +res? Or nothing?)
midx = x
mlp = self.mlp(x)
gate = sigmoid(self.mlp2(x))
x = x*gate + mlp
newres = x
x = RMSNorm(x, ELEMENTWISEAFFINE={ELEMENTWISEAFFINE}), 
```

The RMSNorm may not even be necessary. Is this additional mlp necessary, or can it just be learned via the embedding matrix? Is the residual previously necessary, only to allow for the implementation of no-op? (Previously, the norm of attn(x) is 10x that of mlp(x), but 1/4 of that of res.)

## Experiments

First, we try, without all layer loss. Note that it is exceptionally important to normalize the input to the MLP, else the gradient quickly approaches NaN. During optimization, it is also important that the MLP get a piece of the real residual signal, without it being fully destroyed by attention (why?), or alternatively we use all layer loss.
```
attn = self.attn(x, x) + x
y = self.ln(attn)
mlp = self.mlp(y)
midx = mlp
y = torch.sigmoid(self.mlp2(mlp))
x = mlp + res*(1 - y)
newres = x
x = self.ln(x)
```

Frankly, it's not clear how well the previous worked... but the curve is interesting, it certainly converges.

![loss plot](img/12-gate-mlpofattn.png)



Let's test vanilla architectural changes. Consider transformers as conditional summation. The hypothesis is that this should perform just as well as our usual `mlp*attn` achitecture:
```
class Gate(DualModule):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # smoother ReLU, also maybe helps dead neurons empirically?
        # (should just delete dead neurons)
        self.gelu = nn.GELU(approximate='tanh')  # should just use 'none' if not trying to copy GPT2
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = x.sum(dim=-1, keepdim=True) # sum over the last dimension
        x = torch.sigmoid(x)
        return x
```
```
attn = self.attn(x, x)
mlp = self.gate(x)
midx = mlp
y = mlp*attn
x = y + res
newres = x
x = self.ln(x)
```

![loss plot](img/12-gate-axg.png)

But unfortunately, it doesn't! In fact, the curve looks less nice than the prior experiment, and I'm not even sure that it converges. So we really lose a lot of expressitivity; the hypothesis is just wrong. To verify that it isn't some issue with some other part of the code, let's just run the original `mlp*attn` architecture again. It seems fine, no issues:
```
attn = self.attn(x, x)
mlp = self.mlp(x)
midx = mlp
y = mlp*attn
x = y + res
newres = x
x = self.ln(x)
```

![loss plot](img/12-axm.png)


I wonder how much of the gate experiment is just loss in parameters, i.e. if we add in the same number of parameters and then do a summation, does it help?  It turns out ot make no difference at all, compared to the `12-gate-axg` experiment.
```
class Gate(DualModule):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # smoother ReLU, also maybe helps dead neurons empirically?
        # (should just delete dead neurons)
        self.gelu = nn.GELU(approximate='tanh')  # should just use 'none' if not trying to copy GPT2
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = x.sum(dim=-1, keepdim=True) # sum over the last dimension
        x = torch.sigmoid(x)
        return x
```
```
attn = self.attn(x, x)
mlp = self.gate(x)
midx = mlp
y = mlp*attn
x = y + res
newres = x
x = self.ln(x)
```
![loss plot](img/12-gate-axg-moreparams.png)


This points to the conclusion that the mlp does not just serve as a gate; it is in fact memorizing useful information. This begs the question: is attention itself serving as a gate? No, it seems that the attention component is not just functioning as a gate. (The outcome reminds me of the outcome of removing attention entirely.)
```
attn = self.attn(x, x)
mlp = self.mlp(x)
midx = mlp
gate = attn.sum(dim=-1, keepdim=True)
y = mlp*gate
x = y + res
newres = x
x = self.ln(x)
```
![loss plot](img/12-axm-attentiongate.png)


Let me also feed in the gate through a sigmoid to make sure we are coming to the correct conclusion. The conclusion remains unchanged:
```
attn = self.attn(x, x)
mlp = self.mlp(x)
midx = mlp
gate = torch.sigmoid(attn.sum(dim=-1, keepdim=True))
y = mlp*gate
x = y + res
newres = x
x = self.ln(x)
```
![loss plot](img/12-axm-attentiongate-sigmoid.png)




Let's keep trying random things. First, let's try adding attn back into the residual, in hopes that it makes optimization faster (it does not make optimization faster):
```
attn = self.attn(x, x)
mlp = self.mlp(x)
midx = mlp
y = mlp*attn
x = y + attn + res
newres = x
x = self.ln(x)
```
![loss plot](img/12-scratchpad.png)


Is this really the best version of this architecture? Let's check if giving mlp access to attn helps with anything... No, it in fact penalizes things. (wtf) Probably the more pure the residual signal, the better at first...
```
attn = self.attn(x, x)
mlp = self.mlp(x + attn)
midx = mlp
y = mlp*attn
x = y + res
newres = x
x = self.ln(x)
```
![loss plot](img/12-scratchpad-1.png)


Let's check if giving mlp only access to attn hurts anything... (It hurts things a little bit)
```
attn = self.attn(x, x)
mlp = self.mlp(attn)
midx = mlp
y = mlp*attn
x = y + res
newres = x
x = self.ln(x)
```
![loss plot](img/12-scratchpad-2.png)



TODO: original transformer basline:
```
attn = self.attn(x, x)
mlp = self.mlp(self.ln(attn + res))
midx = mlp
y = mlp
x = y + attn + res
newres = x
x = self.ln(x)
```
![loss plot](img/12-baseline.png)



Let's train a larger model just for fun (value matrix is set to True)
```
attn = self.attn(x, x)
mlp = self.mlp(x)
midx = mlp
y = mlp*attn
x = y + res
newres = x
x = self.ln(x)
```
![loss plot](img/13-larger.png)

TODO train a baseline GPT-medium (copy Kaparthy code)


Let's "rotate" the mlp before applying it (this is probably redundant, because the last layer of the MLP probably already does this). It doesn't seem to work super well, but it does converge (does it get better?) (value matrix is set to True)
```
attn = self.attn(x, x)
mlp = self.rotator(self.mlp(x))
midx = mlp
y = mlp*attn
x = y + res
newres = x
x = self.ln(x)
```
![loss plot](img/13-rotator.png)


Let's try the rotator again but with more warmup time (from `100` from `10`). It still doesn't seem to bring any improvement. (value matrix is set to True)
```
attn = self.attn(x, x)
mlp = self.rotator(self.mlp(x))
midx = mlp
y = mlp*attn
x = y + res
newres = x
x = self.ln(x)
```
![loss plot](img/13-rotator-slowerlr.png)


Let's set `n_embd = 1296` (and also increase warmupsteps to `100` from `10`). More embed is not substantially better! (TODO try normal transformers, is it substantially better there?) (value matrix is set to True)
```
attn = self.attn(x, x)
mlp = self.mlp(x)
midx = mlp
y = mlp*attn
x = y + res
newres = x
x = self.ln(x)
```
![loss plot](img/13-moreembd.png)


Let's go back to `n_embd = 768` (and also increase warmupsteps to `100` from `10`). Optimization is a little slower but it ends up in the same place. (value matrix is set to True)
```
attn = self.attn(x, x)
mlp = self.mlp(x)
midx = mlp
y = mlp*attn
x = y + res
newres = x
x = self.ln(x)
```
![loss plot](img/13-axm-100warmup.png)


Let me try to reproduce axm again... wait, why is it different? (value matrix is set to True) (Warmup steps `10`).
```
attn = self.attn(x, x)
mlp = self.mlp(x)
midx = mlp
y = mlp*attn
x = y + res
newres = x
x = self.ln(x)
```
![loss plot](img/13-axm-scratchpad.png)


Maybe I have to turn off the value matrix... Indeed it was because of the value matrix. (Warmup steps `10`). Yes this is closer (discrepancy may be due to RMSNorm, I forget now.)
```
attn = self.attn(x, x)
mlp = self.mlp(x)
midx = mlp
y = mlp*attn
x = y + res
newres = x
x = self.ln(x)
```
![loss plot](img/13-axm-novalue.png)



Let's send warmup steps back to `100` and turn the value matrix back on just for faster training.  What happens if we use x instead of res? It is simply a lost cause, optimization-wise, even with learning rate set to 1/4 of the usual (`1.5e-4` instaed of `6e-4`.) (It may eventually converge to the same place, I think, eventually? But I don't have enough patience to find out.) **TODO: why is this harder to optimize?**
```
attn = self.attn(x, x)
mlp = self.mlp(x)
midx = mlp
y = mlp*attn
x = y + x
newres = x
x = self.ln(x)
```
![loss plot](img/13-axm-experiment.png)


I suspect that the mlp is deleting "semantic dimensions" of note (but this only works if dimensions align...). Does the size of mlp output scale with the size of the residual that we are trying to attenuate? What if we try to help it along, directly have it attenuate res (hopefully), and add attn separately? There appears to be no tangible benefit.
```
attn = self.attn(x, x)
mlp = self.mlp(x)
midx = mlp
y = mlp*res
x = y + res + attn
newres = x
x = self.ln(x)
```
![loss plot](img/13-axm-deletion.png)


Let's revisit vanilla attention, with indpenednet blocks, copying a bit from the original Kaprthy code. Wait, it does much better! Why?
```
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```
![loss plot](img/13-baseline.png)


Let's make sure that our multiplication schematic is still competitive when not reusing weights. Vanilla is still true. Unlike previously, this seems to be a complete no-op. (Maybe this only helps in the re-using weight regime?)
```
y = self.ln_1(x)
x = x + self.attn(y)*self.mlp(y)
```
![loss plot](img/13-baseline-axm.png)

Let's turn on the original, reusing weights.
```
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=True
MLP_SCALE=4
```
![loss plot](img/13-baseline-reuseweights.png)

Clearly reusing weight is what is wrong. So it seems like we need to probably increase the size of the mlp matrix correspondingly (from 4 to something larger, i.e. 4*NUMLAYERS to match the original number of weights) if we do decide to reuse weights.

Let's turn on multiplication re-using wieghts again, just to see if munltiplication helps in the re-using weights regime. (Yes, it does improve optimization a little bit) Note that removing the value matrix in fact also improves things substantially during optimization (but is slower):
```
y = self.ln_1(x)
x = x + self.attn(y)*self.mlp(y)
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=True
MLP_SCALE=4
```
![loss plot](img/13-axm-reuseweights.png)
![loss plot](img/13-axm-reuseweights-novaluecomp.png)

Now, setting MLP_SCALE=4*12=48, to match the weights lost due to removal. Our hypothesis is wrong, and increasing the size of the MLP doesn't really help! 
```
y = self.ln_1(x)
x = x + self.attn(y)*self.mlp(y)
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=True
MLP_SCALE=48
```
![loss plot](img/13-axm-reuseweights-48mlp.png)


To make sure, let's try `MLP_SCALE=48` for addition too. The performance impact of not reusing weights persists:
```
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=True
MLP_SCALE=48
```
![loss plot](img/13-apm-reuseweights-48mlp.png)


So somehow, it seems that we want to learn a different MLP for each layer. TODO



Returning to the non-reusing-weights regime, what happens if we specifically have mlp negate embeddings that contribute to attention. This performs decidely worse:
```
y = self.ln_1(x)
mlp=self.mlp(y)
x = x + self.attn(y,mlp*y)
======== 
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-axm-targetednegate.png)


I don't understand though; isn't that mathematically equivalent to (note that this experiment, the no-value no-sharing-weights, is the best yet):
```
y = self.ln_1(x)
mlp=self.mlp(y)
x = x + self.attn(y,y)*mlp
======== 
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-axm-novalue.png)

No, they are not equivalent; the difference is that mlp also has dimension `(B, T, C)`. And when we do element-wise multiplication, in the first experiment `attn(y, mlp*y)`, each contribution is multiplied by the source token MLP weight, whereas when we do `attn(y, y)*mlp`, the final output is weighted by the destination token MLP.

Now let's try: (TODO try putting second LN bak in and running attn on output of mlp). ONe hypothesis is that MLP exists exclusively to ``negate'' certain priveleged dimensions of the residual, and maybe htis architecture would facilitate that. But clearly, it does not facilitate it, it seems to be a long-term penalty:
```
y = self.ln_1(x)
mlp=self.mlp(y)
x = x + mlp*x + self.attn(y,y)
======== 
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-r+rxm+a.png)


Orthogonally, I would like to get of the residual. Ideally, `attn(x)` can learn to replace it, but the problem is that it runs itself through a softmax, so the weight of the identity will always decrease. We need to allow the weight of the original to stay the same (i.e. in computation, we need to allow union in addition to replacement, and replacement is handled by the MLP). Can we avoid running the attn through softmax? (This doesn't work)
```
y = self.ln_1(x)
mlp=self.mlp(y)
x = 2*self.attn(y,y)*mlp
======== 
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-2axm.png)


Let's try running attention through a sigmoid instead (It deosn't work)
```
y = self.ln_1(x)
mlp=self.mlp(y)
x = self.sigmoidattn(y,y)*mlp
======== 
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-sigmoidaxm.png)


Adding x back: it seems anyways extremely important, that self.attn always picks "the best ones to combine" (maybe to prevent too much information pollution). Otherwise, there seems not to be any incentive not to send every weight to be equal to 1: (TODO what if we set every weight to be 1)
```
y = self.ln_1(x)
mlp=self.mlp(y)
x = self.sigmoidattn(y,y)*mlp+x
======== 
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-sigmoidaxm+x.png)


Just for fun, just force the attention diagonal to always be 1 (keeping sigmoid):
```
y = self.ln_1(x)
mlp=self.mlp(y)
x = self.sigmoiddiagattn(y,y)*mlp+x
======== 
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-sigmoidaxm-diag.jpg)

Removing the `+x`:
```
y = self.ln_1(x)
mlp=self.mlp(y)
x = self.sigmoiddiagattn(y,y)*mlp
======== 
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-sigmoidaxm-diag-nox.jpg)


Alright now, switching gears entirely back to something more complicated. This is my best blind guess at an architecture currently. The outcome is that it converges to the same place but is decidely slower to optimize in the very beginning:
```
y = self.ln_1(x)
newemb = self.attn(y) + x
mlp=self.mlp(self.ln_2(newemb))
x = mlp*newemb + x
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-complicated.jpg)


Does delaying the input of mlp by a layer change anything at all? It optimizes faster, but then seems to fail and get worse later, even decongealing after step 4000. So input delay to MLP doesn't matter and maybe it in fact hurts, but maybe ensuring there is a large residual component to multiply into mlp helps with optimization early on but maybe not later? (So mlp maybe isn't attenuating x? I'm thoroughly confused.)
```
y = self.ln_1(x)
newemb = self.attn(y) + x
mlp=self.mlp(y)
x = mlp*newemb + x
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-complicated-2.jpg)


Another experiment. Note that here, I forgot to pass the residual into mlp. It doesn't realy converge to the same spot...
```
y = self.ln_1(x)
newemb = self.attn(y)
mlp=self.mlp(self.ln_2(newemb))
x = mlp*newemb + x
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-complicated-3.jpg)


Let's feed the residual back in. Compared to the first "complicated" experiment, here mlp only multiplies `self.attn(y)`, and not `x`; it is worse than the first complicated experiment. Compared to the `axm` experiments, here mlp additionally gets an un-attenuated attn component as input, which is clearly a penalty. So for some reason, it is is important to explicitly attenuate only the attention component, and not the residual.
```
y = self.ln_1(x)
newemb = self.attn(y)
mlp=self.mlp(self.ln_2(newemb + x))
x = mlp*newemb + x
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-complicated-4.jpg)

Experiment 5:
```
y = self.ln_1(x)
newemb = self.attn(y)
mlp=self.mlp(self.ln_2(newemb))
x = mlp*(newemb + x) + x
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-complicated-5.jpg)

Experiment 6:
```
y = self.ln_1(x)
newemb = self.attn(y)
mlp=self.mlp(self.ln_2(newemb))
x = mlp*(x) + x
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-complicated-6.jpg)


The takeaways: (Experiment 2) `mlp(x) * (self.attn(x) + x)` seems to optimize faster than (Experiment 1) `mlp(self.attn(x)+x) * (self.attn(x) + x)` at first, but later it really fails. (Experiment 3) `mlp(self.attn(x)) * (self.attn(x))` only is atrocious, worse than both. (Experiment 4) `mlp(self.attn(x)+x) * (self.attn(x))` is overall just slightly worse than Experiment 1, thouguh has a tiny optimization bump in the very beginning... (Experiment 5) `mlp(self.attn(x)) * (self.attn(x) + x)` optimizes slowly at the beginning, and doesn't look like it will converge to a good place (TODO run for longer)... (Experiment 6) `mlp(self.attn(x)) * (x)` is more stable than 5 but is even worse worse than Experiment 3. All are worse than (13-baseline-axm)  `mlp(x) * self.attn(x)`.

Takeaways: MLP needs to see the residual, not just attention. It doesn't care about the current attention component at all. It also shouldn't multiply the residual itself (and it should multiply attention only) -- it converges faster but really odd things happen later down the road...


## More Semantics


Let's remove the 1s from the diagonal of attention. The motivation here is, we are already copying x, there is no need to copy it again (but wait, what if there is no application to do, i.e. it is not near anything. I suspect this will not do as well. Instead, we should do the next experiment.) (I accidentally deleted this run...)
```
self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size),diagonal=-1).bool().view(1, 1, config.block_size, config.block_size))
========
y = self.ln_1(x)
attn = self.attn(y)
mlp=self.mlp(y)
x = mlp*attn + x
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-nodiagonal.jpg)


Now, we use the usual attention causality (lower triangular mask = True), but when summing things up, we omit the value contribution from the identity token (so the mlp(x) does not apply/multiply on it). This also requires turning the value matrix off... The result is equivalent to the one before! (What does this mean?)
```
y = y*self.nodiagonal[:,:,T,T] # delete the self contribution
========
y = self.ln_1(x)
attn = self.attn(y)
mlp=self.mlp(y)
x = mlp*attn + x
========
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/13-removediagonal.jpg)

In any case, the takeaway is that in this architecture, it is important to run `mlp*x` as well. Maybe, because we always add the residual back to itself, `mlp*x` is performing leftover cleanup of the previous residual, over and over and over again.

What happens if I add a constant contribution from the identity token? I.e. `mlp(x)*(attn+x)`. That was the `13-complicated-2` experiment from above. It does amazingly in the beginning and then odd things happen. Experiment 1 `mlp(self.attn(x)+x) * (self.attn(x) + x)` does slightly less well in the beginning, but eventually seems to converge? Let me try to run this experiment again.

Note in `13-complicated-2` it is `attn + x` but here it is `attn + y` (layer normed). As we found previously, it really does hurt long term performance, a little bit!
```
y = self.ln_1(x)
attn = self.attn(y)
mlp = self.mlp(y)
x = mlp*(attn+y) + x
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/14-complicated-2.jpg)


In an ideal world, we zero out the self contribution in attention, and add in a whole unit of it afterwards:
```
y = self.ln_1(x)
attn = self.zerodattn(y)
mlp = self.mlp(y)
x = mlp*(attn+y) + x
========
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/14-zerodiagonal-addback.jpg)

Note that, in general, multiplication seems slightly worse than addition.

Unrelatedly, let me try to do the "DELETE_SELF_CONTRIBUTION" faster in the value matrix regime. (Our implementation is not faster.) But note that DELETE_SELF_CONTRIBUTION has no effect at all:
```
y = self.ln_1(x)
attn = self.attn(y)
mlp = self.mlp(y)
x = mlp*(attn) + x
========
DELETE_SELF_CONTRIBUTION=True
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/14-zerodiagonal-addback-2.jpg)


TODO when doing addition attention, what happens if I run `mlp(x)` instead of `mlp(LN(attn + res))`?

TODO is mlp allowing us to attenuate res? I.e. deletion instead of replacement?


## Takeaways so far

Attention as positional applicator, should be allowed to no-op. It essentially computes `neighbors(x)`

`MLP(x) * y `applies `x` to `y = neighbors(x)` in a way specified by `x`. The result is added to the computation graph (`x`).

I.e. imagine left application (mirroring traditional visualizations of combinator calculus, so below, x is called on y)

```
        ?
    y       x
        prevy   prevx
            ppy     ppx
                pppy    pppx
```

What about multiple arguments? I.e. traditionally
```
        ?
     ?      z
  ?     y  
S   x
```
reduces to
```
      ?
  ?       ?
x  y    x   z
```
Here, we have something like
```
    ?
z       ?
    y      ?
        x     S
```
and what maybe happens is that x is just added to the residual, and so is y, while they wait for the final argument, and then finally `MLP(S+x+y)*z + (S+x+y)` computes the application (But in this case why do we need to add `S+x+y` back to the residual? Maybe to defer it, and try again, to make optimization easier? The more times the better I guess? But then `z` should be added back in in the next layer. And it is, through the residual from the previous column.) Now, why might we want to do `MLP(x)*(attn(x) + x)`, instead of just `MLP(x)*attn(x)`. Well, maybe the effect of MLP in prior layers was incomplete, so we want to make sure that MLP can keep cleaning it up?

I really do think we should use zero'd attention, but it is really quite too slow, and the non-zero'd version is a good approximation.


## Reusing Weights

Our goal is to figure out how to re-use weights, since that seems to be a central tenet of my computational theory.

First, let me try a new version of all-train, where we compute a loss at every layer, but only propagate the loss one layer deep each time. This works really poorly (somewhat expected, since we're not reusing weights, so how can they hope to learn anything quickly?). This shows that with poor oprimizatbility, it's hard to distinguish that from poor expressivity.

```
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```
![loss plot](img/14-alltrain.jpg)

Evven reusing weights, it is horribe:

```
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```
![loss plot](img/14-alltrain-reuseweights.jpg)


I guess it is important for backprop to be somewhat deep. Let me remove the detach(), and stop reusing weights (and interestingly, it's not any slower than with the detach(), TODO why is that?). Generally, computing loss at every layer is simply worse than computing only at one layer.

```
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```
![loss plot](img/14-alltrain-deep.jpg)


Let's talk about mlp matrices. Here, we map y to a (CxC)-matrix M (using the appropriate decomposition), and then directly apply M to x. Here, we use an inner matrix of size 48x48.
```
y = self.ln_1(x)
attn = self.attn(y)
mlp, bias = self.fatmlp(y)
M = self.matrixfromparams(mlp)
x = M @ attn + bias + x
========
DELETE_SELF_CONTRIBUTION=False
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/14-mlpmatrix.jpg)


Let's get rid of the bias and increase the size of the inner matrix to `MLPMAT_INNER_SIZE = 128`. This isn't very good at all! The bias seems to be quite important. 
```
y = self.ln_1(x)
attn = self.attn(y)
mlp, bias = self.fatmlp(y)
M = self.matrixfromparams(mlp)
x = M @ attn + x
========
DELETE_SELF_CONTRIBUTION=False
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/14-mlpmatrix-nobias-moreinner.jpg)

Let's add the bias back but keep `MLPMAT_INNER_SIZE = 128`. This does well. Is this because of more parameters, or due to the more expressive MLP? Why is bias so important? Maybe it allows for replacing the applicator.
```
y = self.ln_1(x)
attn = self.attn(y)
mlp, bias = self.fatmlp(y)
M = self.matrixfromparams(mlp)
x = M @ attn + bias + x
========
DELETE_SELF_CONTRIBUTION=False
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/14-mlpmatrix-moreinner.jpg)

One question is what happens if we keep the same size for fatmlp (i.e. are the parameters more useful for somewhere else)

One question (TODO) is how to encode multiple application? In other words, a copy operation.

Multiple applications (x + y)? Apply to same attendees? Hmm. Should really be one applicator per column. And in my opinion, a column should comprise either an applicator or an applicatee; a token shoud not able to be BOTH (unless in some parallel superposition?) (Perhaps dot product attn(x) with x to see if it is a no-op, first starting in the no-value regime).

Let's try to get rid of the residual... we revert to `mlp*attn` just to make this experiment fast. Note that sometimes the dot product is negative --- why? Note that the value matrix is set to false. Which means that other words that we are attending to have opposite embeddings of the applicator. It turns out this is completely broken (and surely the same thing is achieved just by forwarding along attn...)
```
y = self.ln_1(x)
attn = self.attn(y, y) 
app = torch.linalg.vecdot(attn, y,dim=-1).unsqueeze(-1) 
mlp = self.mlp(y)
# NOTE sometimes negative dot product (why?)
app = (torch.sigmoid(app) - 0.5) * 2  # app is -1 or 1
x = mlp * attn + app * x
========
DELETE_SELF_CONTRIBUTION=False
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/14-axm-appselector.jpg)


On that note, I am curious, if attn alone is enough of a residual (answer: it is not.) The plot looks similar to if we had used y instead of attn; maybe attn is destructive because of the layer norm, not because it is an attention component. (perhaps mlp is wiping it out)
```
y = self.ln_1(x)
attn = self.attn(y, y)
mlp = self.mlp(y)
x = mlp * attn + attn
========
DELETE_SELF_CONTRIBUTION=False
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/14-axm-plusattn-nores.jpg)


We should probably normalize app... else everything probably has large dot product (and I suspect layer norm doesn't normalize it enough). (Um, what did I do) (Also, instead of sigmoid, should just use tanh)
```
y = self.ln_1(x)
attn = self.attn(y, y) 
siz = torch.linalg.vecdot(y, y,dim=-1).unsqueeze(-1)
app = torch.linalg.vecdot(attn, y,dim=-1).unsqueeze(-1) / siz
# print(app[-1,-1,-1].item())
mlp = self.mlp(y)
# NOTE sometimes negative dot product (why?)
app = (torch.sigmoid(app) - 0.5) * 2  # app is -1 or 1
x = mlp * attn + app * x
========
DELETE_SELF_CONTRIBUTION=False
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
```
![loss plot](img/14-axm-appselector-2.jpg)


Trying app selection with our weird matrix conglomerate (this is exceedingly slow). It is simultatenousy promising and unpromising, and I'm not patient enough to run it all the way (should really optimize app computation if we are going to run more experiments.)
```
y = self.ln_1(x)
attn = self.attn(y,y)
siz = torch.linalg.vecdot(y, y,dim=-1).unsqueeze(-1) # (B, T, 1)
app = torch.linalg.vecdot(attn, y,dim=-1).unsqueeze(-1) / siz
app = (torch.sigmoid(app) - 0.5) * 2  # [-1, 1]
m, bias = self.fatmlp(y)
M = self.applymat(m, attn) #(B, T, 3*C), (B, T, C) -> (B, T, C)
x = M + app * bias + app * x
======
DELETE_SELF_CONTRIBUTION=False
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
MLPMAT_INNER_SIZE=128
```
![loss plot](img/14-axm-appselector-3.jpg)


Honestly, we should also flip the app on the bias. Ideally app is 0 to 1. (Also how do we support this computation with the value matrix?) Well, this turns out to be strictly worse than `mlpmatrix-moreinner`...
```
y = self.ln_1(x)
attn = self.attn(y,y)
siz = torch.linalg.vecdot(y, y,dim=-1).unsqueeze(-1) # (B, T, 1)
app = torch.linalg.vecdot(attn, y,dim=-1).unsqueeze(-1) / siz # may be greater than 1
app = (torch.sigmoid(torch.abs(app)) - 0.5) * 2  # [0, 1]
m, bias = self.fatmlp(y)
M = self.applymat(m, attn) #(B, T, 3*C), (B, T, C) -> (B, T, C)
x = (1-app) * (M + bias) + app * x
======
DELETE_SELF_CONTRIBUTION=False
VALUEMATRIX=False
REUSE_WEIGHTS=False
MLP_SCALE=4
MLPMAT_INNER_SIZE=128
```
![loss plot](img/14-axm-appselector-4.jpg)


TODO: pull app out directly from KQ matrix, I think i  have a bug somewhere.


Let's exeriment with mlpmatrix-moreinner some more. Here, we want to know why `axm` works but here we need the bias. What if we `bias x attn`.

```
y = self.ln_1(x)
attn = self.attn(y,y)
siz = torch.linalg.vecdot(y, y,dim=-1).unsqueeze(-1) # (B, T, 1)
app = torch.linalg.vecdot(attn, y,dim=-1).unsqueeze(-1) / siz # may be greater than 1
app = (torch.sigmoid(torch.abs(app)) - 0.5) * 2  # [0, 1]
m, bias = self.fatmlp(y)
M = self.applymat(m, attn) #(B, T, 3*C), (B, T, C) -> (B, T, C)
x = M + bias*attn + x
======
DELETE_SELF_CONTRIBUTION=False
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MLPMAT_INNER_SIZE=128
```
![loss plot](img/14-mlpmatrix-bxa.jpg)



Let's exeriment with mlpmatrix-moreinner some more...
```
y = self.ln_1(x)
attn = self.attn(y,y)
siz = torch.linalg.vecdot(y, y,dim=-1).unsqueeze(-1) # (B, T, 1)
app = torch.linalg.vecdot(attn, y,dim=-1).unsqueeze(-1) / siz # may be greater than 1
app = (torch.sigmoid(torch.abs(app)) - 0.5) * 2  # [0, 1]
m, bias = self.fatmlp(y)
M = self.applymat(m, attn) #(B, T, 3*C), (B, T, C) -> (B, T, C)
x = M + bias + x
======
DELETE_SELF_CONTRIBUTION=False
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MLPMAT_INNER_SIZE=128
```
![loss plot](img/14-mlpmatrix-experiment.jpg)


I am curious what happens if we omit attn entirely from the residual stream in the vanilla architecture. It eventually does converge!  It is just slightly harder to optimize. (So clearly MLP needs to map the output of attention only.) (Note that in the log, `perc(-)` starts off near 1 and then gets smaller and smaller, closer to `0.6`; this corresponds to the number of `(B,T)` s.t. the sum of self attention scores (over the 12 heads) is less than 1. This number starts off huge, but eventually it seems that it is getting smaller; so, the overall sum is getting larger, so it is slowly learning to attend more and more to the self? Confusing.)
```
y = self.ln_1(x)
attn = self.attn(y)
x = x + self.mlp(self.ln_2(x + attn))
======
DELETE_SELF_CONTRIBUTION=False
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MLPMAT_INNER_SIZE=128
```
![loss plot](img/15-baseline.jpg)


Again, let me omit the diagonal from attn, but here for vanilla transformer. It works well (but unclear if the gains are substantial, or increasing.)
```
y = self.ln_1(x)
attn = self.attn(y)
x = x + attn
x = x + self.mlp(self.ln_2(x))
======
DELETE_SELF_CONTRIBUTION=True
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MLPMAT_INNER_SIZE=128
```
![loss plot](img/14-nodiagonal-baseline.jpg)


What if we do both, i.e. omit the diagonal, and also exclude attn from the residual? This converges, and eventually does even slightly  better than the original...
```
y = self.ln_1(x)
attn, _ = self.attn(y, y)
x = x + self.mlp(self.ln_2(x + attn))
======
DELETE_SELF_CONTRIBUTION=True
MEASURE_SELF_CONTRIBUTION=True
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MLPMAT_INNER_SIZE=128
```
![loss plot](img/14-nodiagonal-noattn.jpg)

Note that at this point I also silently normalized the `VALUE_MATRIX=True` code to divide the sum of values by the number of heads.


And again, what happens if we omit the residual entirely from the mlp. We have tried this before, and it takes longer to converge (but it should converge?) But this also breaks our intuition of needing to run `self.mlp(on the applicator)`. So maybe this one shouldn't converge at all. (After experiment) the results are:
```
y = self.ln_1(x)
attn, scores = self.attn(y, y)
x = x + self.mlp(attn)
======
DELETE_SELF_CONTRIBUTION=True
MEASURE_SELF_CONTRIBUTION=True
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MLPMAT_INNER_SIZE=128
```
![loss plot](img/14-nodiagonal-noattn-nomlpres.jpg)


(TODO I did not run this experiment)
And now, instead of feeding attn as input into mlp, we apply our mlp thing instead. This experiment we've run before, except now delete self contribution is true.
```
y = self.ln_1(x)
attn, scores = self.attn(y, y)
m, bias = self.fatmlp(y)
M = self.applymat(m, attn)
x = x + M + bias
======
DELETE_SELF_CONTRIBUTION=True
MEASURE_SELF_CONTRIBUTION=True
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MLPMAT_INNER_SIZE=128
```
![loss plot](img/14-nodiagonal-noattn-nomlpres.jpg)


I am curious if the last layer of the MLP needs a bias in vanilla transformers:
```
========
y = self.ln_1(x)
attn, score = self.attn(y, y)
x = x + attn
x = x + self.mlp(self.ln_2(x))
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=16384
MLPMAT_INNER_SIZE=128
DELETE_SELF_CONTRIBUTION=False
```
![loss plot](img/14-nooutlinearbiasmlp.jpg)

## Principled Experiments

This is how I would currently design our network. The assumption is that, if a token is not an applicator, i.e. it is not applying itself to anything, it should just be shoveled up another level / compute a no-op, because there is no application to be done. The only issue may be that of the layer norm... maybe resx should add `x` instead and not `y`. Generally we should ensure `bias` is initialized to zero, and `M` starts my computing the identity. Should M compute a basic matrix rotation, or should it compute a full blown MLP? I think it should maybe compute a full blown MLP --- currently, it doesn't, which is maybe why the residual is useful, as now we can apply the nonlinearity in the next layer. (But how do we initialize an MLP to idenitty? Is this why we need the residual? Qn: is the attention residual important? I posit that the answer is No, based on prior experiments where I fed attn directly into MLP as input and never aded it to the residual stream. E.g. `img/14-nodiagonal-noattn.jpg`). (Resx is broken -- need to initialize, so just use x for now) Hmmm... it's not great, it's not horrible. 
```
y = self.ln_1(x)
attn, resx, scores = self.attn(y, y)
hiddenBias, fParams, bParams = self.compiler(y)
machineOutput = self.execute(attn, fParams, bParams, hiddenBias)
x = x + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=True
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=True
```
![loss plot](img/16-principle-1.jpg)


Does more parameters help?
```
y = self.ln_1(x)
attn, resx, scores = self.attn(y, y)
hiddenBias, fParams, bParams = self.compiler(y)
machineOutput = self.execute(attn, fParams, bParams, hiddenBias)
x = x + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=True
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=128
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=True
```
![loss plot](img/16-principle-2.jpg)

Note that, without zero-ing the diagonal, performance is actually kind of horrible:
```
y = self.ln_1(x)
attn, resx, scores = self.attn(y, y)
hiddenBias, fParams, bParams = self.compiler(y)
machineOutput = self.execute(attn, fParams, bParams, hiddenBias)
x = x + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=True
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
```
![loss plot](img/16-principle-3.jpg)


Zero'ing the diagonal... wait, it is incredible fragile. Sometimes it works sometimes it doesnt, I really have no idea what is going on:
```
y = self.ln_1(x)
attn, resx, scores = self.attn(y, y)
hiddenBias, fParams, bParams = self.compiler(y)
machineOutput = self.execute(attn, fParams, bParams, hiddenBias)
x = x + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=True
EXTRACT_SELF_CONTRIBUTION=False
```
![loss plot](img/16-principle-4.jpg)

Why might a machine that only executes matrix multiplications (+ bias) be as expressive as one that executes an entire MLP? I guess the matrix multiplication (+ bias) is fed into the next layer anyways, which has a mlp/nonlinearity?

Traditionally, with residual, say the desired mapping is `H(x)`, with the residual, we can imagine it wanting to learn `H(x) - x`, which should not be any harder, but the idenitty `H(x) = x` is easier to learn. 


Also, all things considered, it seems quite reasonable to learn some function `f(x, attn)` instead, maybe by feeding it directly into `mlp(x || attn)`. If `attn` is small, perhaps we want this to return the identity, and then it is easier to directly learn `x = x + mlp(x || attn)`. But it seems like a chicken and egg; to replace `x` entirely then `mlp(x || attn)` needs to learn `-x`.

An alternative is to initialize the matrix M in `M@x+b` to the identity matrix, and `b` to the 0 vector; but what about the non-linearity?



Let me try just concatenating `x + mlp(LN(x) || attn(LN(x)))`. This doesn't perform that great, or better, but it doesn't perform worse, because it does converge..
```
(compiler just outputs y, execute outputs mlp(y || attn))
y = self.ln_1(x)
attn, resx, scores = self.attn(y, y)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = x + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
```
![loss plot](img/16-principle-5.jpg)

Doing due diligence, same experiment as before, but `MLP_SCALE=8`. It doesn't appear to improve the base model substantially.
```
(compiler just outputs y, execute outputs mlp(y || attn))
y = self.ln_1(x)
attn, resx, scores = self.attn(y, y)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = x + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=8
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
```
![loss plot](img/16-principle-6.jpg)



Let's try large mlp scale with vanilla transformer. Note that tit doesn't have the same impact as `14-mlpmatrix-moreinner`, but note that the latter has `700M` parameters versus our `520M` parameters. This one still doesn't appear to improve the base model.
```
self.compiler = BenCompilerNoOp(config)
self.execute = VanillaExecute(config)
y = self.ln_1(x)
attn, resx, scores = self.attn(y, y)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = x + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=32
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
```
![loss plot](img/16-vanilla-mlpscale-32.jpg)


Am curious how large we can go, can we get to `700M` parameters? Let's set MLP_SCALE=64, we get `973,701,120` parameters. TLDR; MLP is enough, we don't need our fancy compiler stuff. (Would it help with reusing parameters?)
```
self.compiler = BenCompilerNoOp(config)
self.execute = VanillaExecute(config)
y = self.ln_1(x)
attn, resx, scores = self.attn(y, y)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = x + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=64
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False

```
![loss plot](img/16-vanilla-mlpscale-64.jpg)


What about --- rotate it, and add a bias (computed from attn)... Anyways...

## Attention Sinks


Q: why does magnitude of the residual matter so much? Layer norming it completely destroys it... Even halving it, completely destroys it. (I imagine that it gets smaller and smaller every step, converging to zero.)


New experiment! I'm trying to figure out if we can avoid passing the entirety of res along. Note that all layer loss is on, and reusing weights is true. We have added a "Zero Sink" as well (todo: figure out how to make this computation more efficient, I don't know why it is so slow)
```
self.compiler = BenCompilerNoOp(config)
self.execute = VanillaExecute(config)
y = self.ln_1(x)
attn, resx, scores = self.attn(y, y)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = y + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=True
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=True
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
```
![loss plot](img/16-attentionsink.jpg)

If we turn reusing weights off, 
```

self.compiler = BenCompilerNoOp(config)
self.execute = VanillaExecute(config)
y = self.ln_1(x)
attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = y + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=True
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
```
![loss plot](img/16-attentionsink-noreuse.jpg)

Let me see if I can extract `resw` efficiently by commandeering a direction in the embedding dimensionality. The hope is that by  using `resw`, we can zero out the contribution from the residual (noop) completely if the attention is high enough. We keep this optimization for all future attention sink experiments: (This is not a fair comparison, because we also use `xWeights` now)

```
self.compiler = BenCompilerNoOp(config)
self.execute = VanillaExecute(config)
y = self.ln_1(x)
attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = xWeights*y + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=True
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
```
![loss plot](img/16-attentionsink-noreuse-efficientresw.jpg)

Now, use MLPConcat instead:
```
self.compiler = BenCompilerNoOp(config)
self.execute = BenExecute(config)
y = self.ln_1(x)
attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = xWeights*y + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=True
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
```
![loss plot](img/16-sinkgate-mlpconcat.jpg)


What if I don't layer norm the attention before feeding it into the mlp? (Otherwise same as above)

![loss plot](img/16-sinkgate-mlpconcat-nonorm.jpg)

Should really find a way to start with the residual with weight 1 (instead of 0)...

Let's try the same sinkgate but with vanilla execution instead:
```
========
self.compiler = BenCompilerNoOp(config)
self.execute = VanillaExecute(config)
y = self.ln_1(x)
attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = xWeights*y + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=True
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
```
![loss plot](img/16-sinkgate-vanilla-2.jpg)

JUst for fun... Let me turn REUSE_WEIGHTS on and all_layer_loss off, and feed in the entire residual again. It looks almost identical to `16-sinkgate-vanilla-2`, which has reuse_weights off and all_layer_loss on and the weighted residual. (TODO: why?)
```
========
self.compiler = BenCompilerNoOp(config)
self.execute = VanillaExecute(config)
y = self.ln_1(x)
attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = x + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=True
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
```
![loss plot](img/16-sinkgate-debug.jpg)


Now, let's turn on reuse weights but also do the xWeights again.
```
========
self.compiler = BenCompilerNoOp(config)
self.execute = VanillaExecute(config)
y = self.ln_1(x)
attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = xWeights*y + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=True
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
```
![loss plot](img/16-sinkgate-debug-2.jpg)



## ON training in parallel

Is training in parallel on `T` tokens fundamental? What happens is that at all but the last position, we are now awarding pass-through... perhaps carrying more information than necessary to compute the next token, but perhaps the information is useful for future tokens down the line. So this is like creating some "current understanding state" instead of next token prediction.

## The Identity Mechanism

First, we reduce the learning rate, and run the vanilla experiment
```
self.compiler = BenCompilerNoOp(config)
self.execute = VanillaExecute(config)
y = self.ln_1(x)
attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = x + machineOutput
======== 
max_lr = 0.25*6e-4
min_lr = max_lr * 0.1
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
IDENTITY_LOSS=False
```
![loss plot](img/17-identity.jpg)


Now, we run an experiment where we do the weighting thing again...
```
self.compiler = BenCompilerNoOp(config)
self.execute = VanillaExecute(config) 
========
y = self.ln_1(x)
attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = xWeights * x + (1-xWeights)*machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
IDENTITY_LOSS=True
```
![loss plot](img/17-identity-test.jpg)



Surprisingly, if I get rid of the `(1 - xWeights)`, things are substantially worse. I suspect that this is because the mlp component is additive? Or is the attn component itself just too large (because of the value matrix?). I suspect that it is entirely becaues of the MLP (outdominating the residual component)
```
self.compiler = BenCompilerNoOp(config)
self.execute = VanillaExecute(config) 
========
y = self.ln_1(x)
attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = xWeights * x + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
IDENTITY_LOSS=True
```
![loss plot](img/17-identity-test-no-1minus.jpg)

Now, we switch to `BenExecute`, which currently looks like
```
class BenExecute(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
    
    def forward(self, program, attn):
        return self.mlp(program, attn) # self.ln_2(attn)
```
Note that we do not layernorm the attn signal, in hopes that if it is attenuated, the mlp will recognize that and attenuate its own output as well.
```
self.compiler = BenCompilerNoOp(config)
self.execute = BenExecute(config) 
========
y = self.ln_1(x)
attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = xWeights * x + machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
IDENTITY_LOSS=True
```

![loss plot](img/17-identity-test-no-1minus-mlpconcat.jpg)


Next, we add the layer norm in BenExecute and the `(1-xWeights)` back in. The motivation is to hopefully increase the residual faster, and to make it easier for the attention component to zero itself out. Surprisingly, this absolutely fails.
```
self.compiler = BenCompilerNoOp(config)
self.execute = BenExecute(config) 
========
y = self.ln_1(x)
attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = xWeights * x + (1-xWeights)*machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
IDENTITY_LOSS=True
```
![loss](img/17-identity-test-yes-1minus-mlpconcat.jpg)

Why did the previous fail? Is it because we put the layer norm back? Or is it because we added the `(1-xWeights)`? Let's remove the layer norm again from BenExecute. It works now, but note the strange time when the standard deviation of `output` went out of control:

![loss](img/17-identity-test-yes-1minus-mlpconcat-noln.jpg)

That seems to be why. Now, it seems that mlpconcat is not very good, but in prior experiments it has usually converged to vanilla transformers, and is just slower to train. But it fits my mental model better, so let's stick with it for now, and try to up the training rate:

Let's see if this is still stable, if we return the learning rate to the original of 6e-4; also we switch from cosine similarity to a norm of the difference:

![loss](img/17-yes-1minus-mlpconcat-noln-faster.jpg)


The norm of the output in the previous experiment was very high. What if we layer norm it? It starts off bad but eventually converges. Also note the horizontal section in the beginning while we wait for `fracRes` to converge to some high number like 0.8:
```
class BenExecute(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
    
    def forward(self, program, attn):
        return self.ln_2(self.mlp(program, attn))
```
![loss](img/17-yes-1minus-mlpconcat-noln-faster-2.jpg)

Hmmmmm. I suspect such a program is just too complicated to learn? What happens if I push the identity loss much higher? It is rather unstable... I'm not sure it ever converges at this training rate.
```
_xtraloss = _xtraloss + 2*torch.linalg.norm(_x - _in, dim=-1).mean()
```
![loss](img/17-yes-1minus-mlpconcat-noln-faster-3.jpg)

What happens if I switch it to the 1-norm? I also remove the layer norm wrapping the output.
```
_xtraloss = _xtraloss + torch.linalg.norm(_x - _in, dim=-1, ord=1).mean()
```
![loss](img/17-yes-1minus-mlpconcat-noln-faster-4.jpg)


Let's do something fun, setting `n_layer=48`, and using the infinity norm: (Note that when i use `ord=2`, it doesn't even make it past a loss of `10`. This somehow feels like an important observation, but at this point I don't know what to make of it)
```
_xtraloss = _xtraloss + torch.linalg.norm(_x - _in, dim=-1, ord=float('inf')).mean()
self.compiler = BenCompilerNoOp(config)
self.execute = BenExecute(config) 
========
y = self.ln_1(x)
attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = xWeights * x + (1-xWeights)*machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=True
MLP_SCALE=8
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
IDENTITY_LOSS=True
```
![loss](img/17-1minus-mlpconcat-crazystuff.jpg)

It's a little screwy! It seems to be getting caught when activating the effects of attention.
Set `n_layer=12` just to speed it up a little bit, and also `newx = xWeights * y + (1 - xWeights) * machineOutput`. This is awful, which is odd, because usually layer norming `x` (`y = LN(x)`) would make the standard deviation of `x` larger, not smaller.

![loss](img/17-1minus-mlpconcat-crazystuff-2.jpg)

Let's try the same thing but pop the learning rate lower. Note that ord = 2, in this experiment (The outcome is horrible) (12 layers)
```
Experiment description: Transformer, max LR 0.00015
self.compiler = BenCompilerNoOp(config)
self.execute = BenExecute(config) 
========
y = self.ln_1(x)
attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
program = self.compiler(y)
machineOutput = self.execute(program, attn)
x = xWeights * y + (1-xWeights)*machineOutput
======== 
VALUEMATRIX=True
REUSE_WEIGHTS=True
MLP_SCALE=2
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
IDENTITY_LOSS=True
```
![loss](img/17-1minus-mlpconcat-crazystuff-3.jpg)


Let me try the same thing with ord = inf. It is infinitely better. (Why??) But it cannot seem to get the residual fraction to be high. (12 layers)

![loss](img/17-1minus-mlpconcat-crazystuff-4.jpg)

So let me swap back the x for y and set ord=2. This is pushing up the residual fraction again. But, it diverges.
```
Transformer, max LR 0.00015
Setting:
========
        self.compiler = BenCompilerNoOp(config)
        self.execute = BenExecute(config)
========
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = xWeights * x + (1 - xWeights) * machineOutput
========
                x, metadata = block(x,print_weights=print_weights,step=i)
                _x_total = x
                _in = x.detach()
                _x, _ = block(_in,print_weights=False) # Do again... lol
                _xtraloss = _xtraloss + torch.linalg.norm(_x - _in, dim=-1, ord=2).mean()
========
VALUEMATRIX=True
REUSE_WEIGHTS=True
MLP_SCALE=2
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
IDENTITY_LOSS=True
```
![caption](img/17-1minus-mlpconcat-crazystuff-5.jpg)

Same thing but ord=inf again. Still having trouble getting the residual fraction up:
```
Transformer, max LR 0.00015
Setting:
========
        self.compiler = BenCompilerNoOp(config)
        self.execute = BenExecute(config)
========
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = xWeights * x + (1 - xWeights) * machineOutput
========
                x, metadata = block(x,print_weights=print_weights,step=i)
                _x_total = x
                _in = x.detach()
                _x, _ = block(_in,print_weights=False) # Do again... lol
                _xtraloss = _xtraloss + torch.linalg.norm(_x - _in, dim=-1, ord=float('inf')).mean()
========
VALUEMATRIX=True
REUSE_WEIGHTS=True
MLP_SCALE=2
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
IDENTITY_LOSS=True
```
![caption](img/17-1minus-mlpconcat-crazystuff-6.jpg)


Same thing but with cosine similarity. This one is much nicer. (Similar to the y + infinite ordinality)
```
Transformer, max LR 0.00015
Setting:
========
        self.compiler = BenCompilerNoOp(config)
        self.execute = BenExecute(config)
========
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = xWeights * x + (1 - xWeights) * machineOutput
========
                x, metadata = block(x,print_weights=print_weights,step=i)
                _x_total = x
                _in = x.detach()
                _x, _ = block(_in,print_weights=False) # Do again... lol
                _xtraloss = _xtraloss + (1 - F.cosine_similarity(_x, _in, dim=-1).mean())
========
VALUEMATRIX=True
REUSE_WEIGHTS=True
MLP_SCALE=2
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
IDENTITY_LOSS=True
```
![caption](img/17-1minus-mlpconcat-crazystuff-7.jpg)

Same thing, but MLP_SCALE back to 4.
```
Transformer, max LR 0.00015 n_layer 12
Setting:
==machine code======
        self.compiler = BenCompilerNoOp(config)
        self.execute = BenExecute(config)
==machine modules======
        self.compiler = BenCompilerNoOp(config)
        self.execute = BenExecute(config)
==block logic======
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = xWeights * x + (1 - xWeights) * machineOutput
==loss computation======
                x, metadata = block(x,print_weights=print_weights,step=i)
                _x_total = x
                _in = x.detach()
                _x, _ = block(_in,print_weights=False) # Do again... lol
                _xtraloss = _xtraloss + (1 - F.cosine_similarity(_x, _in, dim=-1).mean())
========
VALUEMATRIX=True
REUSE_WEIGHTS=True
MLP_SCALE=4
MEASURE_SELF_CONTRIBUTION=False
NEW_ALL_LAYER_LOSS=False
MATRIX_NUM_PARAMS=4096
MLPMAT_INNER_SIZE=64
DELETE_SELF_CONTRIBUTION=False
EXTRACT_SELF_CONTRIBUTION=False
ATTENTION_SINK=True
IDENTITY_LOSS=True
```
![caption](img/17-1minus-mlpconcat-crazystuff-8.jpg)

Reuse weights back to false. This will converge to `17-identity-test`.
```
Transformer, max LR 0.00015 n_layer 12
Setting:
==machine code======
class BenExecute(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
        # self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)

    
    def forward(self, program, attn):
        return self.mlp(program, attn)
==machine modules======
        self.compiler = BenCompilerNoOp(config)
        self.execute = BenExecute(config)
==block logic======
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = xWeights * x + (1 - xWeights) * machineOutput
==loss computation======
                x, metadata = block(x,print_weights=print_weights,step=i)
                _x_total = x
                _in = x.detach()
                _x, _ = block(_in,print_weights=False) # Do again... lol
                _xtraloss = _xtraloss + (1 - F.cosine_similarity(_x, _in, dim=-1).mean())
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
ATTENTION_SINK=True
IDENTITY_LOSS=True
```
![caption](img/17-1minus-mlpconcat-crazystuff-9.jpg)


Same thing, but with higher learning rate. And n_layers to 8.
```
Transformer, max LR 0.0006 n_layer 8
Setting:
==machine code======
class BenExecute(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
        # self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)

    
    def forward(self, program, attn):
        return self.mlp(program, attn)
==machine modules======
        self.compiler = BenCompilerNoOp(config)
        self.execute = BenExecute(config)
==block logic======
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = xWeights * x + (1 - xWeights) * machineOutput
==loss computation======
                x, metadata = block(x,print_weights=print_weights,step=i)
                _x_total = x
                _in = x.detach()
                _x, _ = block(_in,print_weights=False) # Do again... lol
                _xtraloss = _xtraloss + (1 - F.cosine_similarity(_x, _in, dim=-1).mean())
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
ATTENTION_SINK=True
IDENTITY_LOSS=True
```
![caption](img/17-1minus-mlpconcat-crazystuff-10.jpg)


Next steps: increase learning rate. What happens if i turn x to y?  Also, our hope was that removing residual would "clarify" signal (because that's how computation works). Do we see improvement in that regard (is perplexity the best way to measure that?).

## Randomness Extraction

First, we see whether, if we absolute-value the residual at every layer, does it change performance compared to vanilla GPT? (Yes, it does, kind of similar to reusing weights, I think. It is a consistent penalty.)

```
Transformer, max LR 0.0006 n_layer 8
Setting:
==details======
 machine_code
class BenExecute(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
        # self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
    def forward(self, program, attn):
        return self.mlp(program, attn)
----------------
 machine_modules
        self.compiler = BenCompilerNoOp(config)
        self.execute = VanillaExecute(config)
----------------
 block_logic
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = (x + machineOutput).abs()
----------------
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
ATTENTION_SINK=False
ATTENTION_MASK =False
IDENTITY_LOSS=False
```
![caption](img/18-vanilla-abs.jpg)



Compare to (somethign is off --- why is this worse than 13-baseline? Because we only use 8 layers instead of 12, lol.):

```
Transformer, max LR 0.0006 n_layer 8
Setting:
==details======
 machine_code
class BenExecute(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
        # self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
    def forward(self, program, attn):
        return self.mlp(program, attn)
----------------
 machine_modules
        self.compiler = BenCompilerNoOp(config)
        self.execute = VanillaExecute(config)
----------------
 block_logic
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = x + machineOutput
----------------
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
ATTENTION_SINK=False
ATTENTION_MASK =False
IDENTITY_LOSS=False
```
![caption](img/18-vanilla-noabs.jpg)

We can potentially think of our network as defining, recursively, a series of "contexts" -- where a context consists of an input string (i.e. comprised of T token embeddings), and an output target (usually computed by "inverting" the network, which is implicitly done during back-propagation). Each context thus defines a "gap", which is the loss, or distance between the input and the target. In a theory of life, for instance, we can posit that "organisms" would like to fill each gap.

Now, let's try to motivate the attention mechanism from this point of view. The MLP is easy to motivate: it is essentially memorizing a truth table of inputs and outputs. The issue with the MLP (and the universal approximation theorem) is that, of course, it does not scale. The attention mechanism, on the other hand, allows us to scale, in a computational way. It turns out that, by only memorizing a few things and having this "tape" or "graph structure", much like in a Turing Machine, we can simulate any computation in the world; the input embeddings then, in some sense, become the code, and the base rules become the interpreter.
The problem is that our interpreter must remain fairly complex: the language is very high level, and very expressive in an immediate way. Yet, at the same time, we don't want to memorize too much in order to build the interpreter. I posit that MLPs both map input tokens to more expressive "program" tokens and also memorize "base building block" functions, whereas the attention component "positions" related tokens together, like a "conditional application" of function tokens to data tokens. In other words, it is some expressive variant of a combinator calculus, which is somehow the perfect trade-off between memorization and expressiveness of the language that is being interpreted (if there is a trade-off at all...)

An organism then comprises an attention component ("where do I feed") and an MLP component ("what do I extract"). Conceivably, we can have an organism that feeds everywhere, but it should be out-competed by an organism that feeds selectively on topical tokens. (How do multiple attention heads build into this?) Similarly, if we have two organisms in sequence that are not attuned to the overall gap, there should be no stability and it is unlikely for any of them to fill the gap. Instead, one organism should try to fill the gap first the best it can, and the remaining gap (its output, and the target) can subsequently be filled by a second organism. (This may be the role of skip connections.)

We may even posit a much more fluid network dynamics, where organisms have not only a choice of which of T tokens to feed-on, but also where they are located in the network; moreover, some organisms may be copies of each other, and perhaps this would facilitate solutions to the Prisoner's Dilemma. (As far as I can imagine, gradient descent rewards selfish behavior only, by design. Cooperation seems only to arise by accident, when organisms are not yet fully attuned? Or does cooperation arise because behavior is never fully discrete, so with 1% chance both organisms cooperate, and this eventually dominates? We can think of it being computed in superposition. But the contribution of cooperation amongst multiple organisms is going to be miniscule, and hard to discover, I'm not sure. Perhaps once loss plateaus, it too will eventually have its time to shine, no matter how small... Indeed perhaps cooperation does arise by chance. But certainly it may not arise in unstable environments.)

If some organisms share code, perhaps cooperation arises faster and easier? This feels less fundamental. But in any case, let me implement some "block routing" mechanism for each layer, which sends the signal to one of n_layer organisms (and in the whole network there are only n_layer organisms) according to a softmax. To start, we skip the softmax, and just apply every block at every layer. Note that this is quite bad! (Note the bug with _xtota, should see 18-router-3 instead, likely won't be better than that, probably similar.)

```
Transformer, max LR 0.0006 n_layer 8
Setting:
==details======
 machine_code
class BenExecute(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
        # self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
    def forward(self, program, attn):
        return self.mlp(program, attn)
----------------
 machine_modules
        self.compiler = BenCompilerNoOp(config)
        self.execute = VanillaExecute(config)
----------------
 block_logic
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = x + machineOutput
----------------
 network_logic
                _finx, metadata = block(x,print_weights=print_weights,step=i)
                for j in range(self.config.n_layer):
                    if i == j:
                        continue
                    b = self.transformer.h[j]
                    _x, _metadata = b(x,print_weights=False,step=i)
                    _finx = _finx + _x
                x = _finx
                _xtotal = x
                # x, metadata = block(x,print_weights=print_weights,step=i)
                # _x_total = x
----------------
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
ATTENTION_SINK=False
ATTENTION_MASK =False
IDENTITY_LOSS=False
```
![caption](img/18-router.jpg)


Now, try to learn a weighted block, i.e., depending on a layer, a block may decide not to contribute. Still no good! (We didn't backprop properly...) (Note the bug, should see 18-router-3 instead, likely won't be better than that, probably similar.)

```
Transformer, max LR 0.0006 n_layer 8
Setting:
==details======
 machine_code
class BenExecute(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
        # self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
    def forward(self, program, attn):
        return self.mlp(program, attn)
----------------
 machine_modules
        self.compiler = BenCompilerNoOp(config)
        self.execute = VanillaExecute(config)
----------------
 block_logic
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = x + machineOutput
        newx = torch.sigmoid(self.routes[step]) * newx
----------------
 network_logic
                _finx, metadata = block(x,print_weights=print_weights,step=i)
                for j in range(self.config.n_layer):
                    if i == j:
                        continue
                    b = self.transformer.h[j]
                    _x, _metadata = b(x,print_weights=False,step=i)
                    _finx = _finx + _x
                x = _finx
                _xtotal = x
                # x, metadata = block(x,print_weights=print_weights,step=i)
                # _x_total = x
----------------
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
ATTENTION_SINK=False
ATTENTION_MASK =False
IDENTITY_LOSS=False
```
![caption](img/18-router-2.jpg)


What if we build a proper softmax router:

```
Transformer, max LR 0.0006 n_layer 8
Setting:
==details======
 machine_code
class BenExecute(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
        # self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
    def forward(self, program, attn):
        return self.mlp(program, attn)
----------------
 machine_modules
        self.compiler = BenCompilerNoOp(config)
        self.execute = VanillaExecute(config)
----------------
 block_logic
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = x + machineOutput
----------------
 network_logic
                routes = F.softmax(self.router[i], dim=-1)
                routes.requires_grad_(True)
                routes.retain_grad()
                x.requires_grad_(True)
                x.retain_grad()
                # print(f"routes grad {routes.grad}")
                # print(f"x grad {x.grad}")
                _finx, metadata = block(x,print_weights=print_weights,step=i)
                _finx = routes[i] * _finx
                for j in range(self.config.n_layer):
                    if i == j:
                        continue
                    b = self.transformer.h[j]
                    _x, _metadata = b(x,print_weights=False,step=i)
                    _finx = _finx + routes[j] * _x
                x = _finx
                _x_total = x
                metadata[f"routes_{i}"] = routes
                # x, metadata = block(x,print_weights=print_weights,step=i)
                # _x_total = x
----------------
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
ATTENTION_SINK=False
ATTENTION_MASK =False
IDENTITY_LOSS=False
```
![caption](img/18-router-3.jpg)

## Learning Programs

I am curious if prepending extra "learnable tokens" (i.e. playing the function of code) might help the network scale. With 768 prepended tokens, initialized with std 1, it seems to be a noop:

```
Transformer, max LR 0.0006 n_layer 12
Setting:
==details======
 machine_code
class BenExecute(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
        # self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
    def forward(self, program, attn):
        return self.mlp(program, attn)
----------------
 machine_modules
        self.compiler = BenCompilerNoOp(config)
        self.execute = VanillaExecute(config)
----------------
 block_logic
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = x + machineOutput
----------------
 code_logic
        # Now, concat our code (NOTE: shoudl we add positional embeddings)
        code_expanded = self.code.unsqueeze(0).expand(b, -1, -1)
        tok_emb = torch.cat((code_expanded, tok_emb), dim=1)
----------------
 network_logic
                x, metadata = block(x,print_weights=print_weights,step=i)
                _x_total = x
----------------
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
ATTENTION_SINK=False
ATTENTION_MASK =False
IDENTITY_LOSS=False
```
![caption](img/18-tokenprogram-2.jpg)

With 128 prepended tokens, initiated with std 5:

```
Transformer, max LR 0.0006 n_layer 12
Setting:
==details======
 machine_code
class BenExecute(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
        # self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
    def forward(self, program, attn):
        return self.mlp(program, attn)
----------------
 machine_modules
        self.compiler = BenCompilerNoOp(config)
        self.execute = VanillaExecute(config)
----------------
 block_logic
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = x + machineOutput
----------------
 code_logic
        # Now, concat our code (NOTE: shoudl we add positional embeddings)
        code_expanded = self.code.unsqueeze(0).expand(b, -1, -1)
        tok_emb = torch.cat((code_expanded, tok_emb), dim=1)
----------------
 network_logic
                x, metadata = block(x,print_weights=print_weights,step=i)
                _x_total = x
----------------
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
ATTENTION_SINK=False
ATTENTION_MASK =False
IDENTITY_LOSS=False
```
![caption](img/18-tokenprogram-3.jpg)


Instead of prepending, what if we directly add it to the input tokens? I set std back to 0.5 because otherwise it has trouble converging... This is definitely not great.

```
Transformer, max LR 0.0006 n_layer 12
Setting:
==details======
 machine_code
class BenExecute(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
        # self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
    def forward(self, program, attn):
        return self.mlp(program, attn)
----------------
 machine_modules
        self.compiler = BenCompilerNoOp(config)
        self.execute = VanillaExecute(config)
----------------
 block_logic
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = x + machineOutput
----------------
 code_logic
        # Now, concat our code (NOTE: shoudl we add positional embeddings)
        code_expanded = self.code.unsqueeze(0)  #.expand(b, -1, -1)
        tok_emb = tok_emb + code_expanded[:, -t:, :]
----------------
 network_logic
                x, metadata = block(x,print_weights=print_weights,step=i)
                _x_total = x
----------------
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
ATTENTION_SINK=False
ATTENTION_MASK =False
IDENTITY_LOSS=False
```
![caption](img/18-tokenprogram-4.jpg)

Just for fun, we run the original 128-code std=5 experiment in conjunction with MLPConcat. I suspect that for these code experiments, the gradient is pretty tiny, (should just print and check) because there just isn't really any immediate point in attending to them?

```
Transformer, max LR 0.0006 n_layer 12
Setting:
==details======
 machine_code
class BenExecute(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
    def forward(self, program, attn):
        return self.mlp(program, attn)
----------------
 machine_modules
        self.compiler = BenCompilerNoOp(config)
        self.execute = BenExecute(config)
----------------
 block_logic
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        newx = x + machineOutput
----------------
 code_logic
        # Now, concat our code (NOTE: shoudl we add positional embeddings)
        code_expanded = self.code.unsqueeze(0).expand(b, -1, -1)
        tok_emb = torch.cat((code_expanded, tok_emb), dim=1)
----------------
 network_logic
                x, metadata = block(x,print_weights=print_weights,step=i)
                _x_total = x
----------------
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
ATTENTION_SINK=False
ATTENTION_MASK =False
IDENTITY_LOSS=False
```
![caption](img/18-tokenprogram-5.jpg)


I decided to run the signal gate design again. It doesn't perform as well, but it's not fatal; I suspect that learning in "superposition" is important (and somehow the network learns to "dwarf" earlier residuals?). TODO investigate

```
Transformer, max LR 0.0006 n_layer 12
Setting:
==details======
 machine_code
class BenExecute(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
    def forward(self, program, attn):
        return self.mlp(program, attn)
----------------
 machine_modules
        self.compiler = BenCompilerNoOp(config)
        self.execute = BenExecute(config)
        self.throughput = nn.Parameter(torch.tensor(-2.0))
        torch.nn.init.normal_(self.throughput, mean=-2.0, std=0.02)
----------------
 block_logic
        y = self.ln_1(x)
        attn, xWeights, scores = self.attn(y, y, print_weights=print_weights)
        program = self.compiler(y)
        machineOutput = self.execute(program, attn)
        tr = torch.sigmoid(self.throughput)
        newx = (1 - tr) * x + tr * machineOutput
----------------
 init_logic CODE_MODE
        elif isinstance(module, GPT) and CODE_MODE:
            torch.nn.init.normal_(module.code, mean=0.0, std=5)
----------------
 code_logic CODE_MODE
        # Now, concat our code (NOTE: shoudl we add positional embeddings)
        if CODE_MODE:
            code_expanded = self.code.unsqueeze(0).expand(b, -1, -1)
            tok_emb = torch.cat((code_expanded, tok_emb), dim=1)
----------------
========
VALUEMATRIX=True
REUSE_WEIGHTS=False
MLP_SCALE=4
ATTENTION_SINK=False
ATTENTION_MASK =False
IDENTITY_LOSS=False
CODE_MODE=False
```
![caption](img/18-throughputgate.jpg)


## On Mixture of Experts

I finally read about MOE properly. A few questions:

What if we do routing as a function of just the pre-attention token x? And then we send in the attention into the expert. According to my principle, these lookup tables are probably best bucketed by applicator (though I imagine that having both also works).

Alternatively, if we route only the output of the attention layer, maybe we should send the output of each attention head to a different expert. This would be the opposite interpretation of the above paragraph; it would be odd if the look-ups went through together. Alternatively, we can code the previous paragraph, and then call that expert n_heads times, once for each of the attention heads. (If the attention heads compute orthogonal things, then this probably doesn't matter so much. I suspect it will usually compute orthogonal things, but we can't rule out "similar" attention pairs being placed together...)

For efficiency, we can also consider projecting down the keyspace into a smaller space. (This projection matrix is perhaps shared by all of the nodes?)

Honestly, it'd be nice to increase the learning rate for the MLP sections only, that would be nice.
